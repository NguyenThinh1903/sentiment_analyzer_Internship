# api.py (v1.5.0 - Tích hợp Product ID vào KB và Gemini)

import time, os, traceback, logging, asyncio, hashlib, json
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn
import google.generativeai as genai
import mysql.connector
from mysql.connector import Error as MySQLError, pooling

import config
try:
    from predict import SentimentPredictor
    PREDICTOR_LOADED = True
except ImportError:
    print("LỖI: Không tìm thấy module 'predict'. SentimentPredictor sẽ không hoạt động.")
    SentimentPredictor = None
    PREDICTOR_LOADED = False

# --- Cấu hình Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - RID=%(request_id)s - %(message)s')
logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'N/A')
        return True
logger.addFilter(RequestIdFilter())

# --- Pydantic Models ---
class SentimentRequest(BaseModel):
    comment: str = Field(..., description="Bình luận cần phân tích.")
    product_id: str | None = Field(None, description="(Tùy chọn) Mã hoặc tên sản phẩm.")  # Thêm product_id
    @field_validator('comment')
    @classmethod
    def strip_comment(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip()
        return v

class UnifiedResponse(BaseModel):
    sentiment: str | None = None
    confidence: float | None = None
    product_id_processed: str | None = Field(None, description="Mã sản phẩm đã được xử lý (nếu có).")  # Thêm product_id_processed
    ai_call_reason: str | None = None
    suggestions: list[str] | None = None
    generated_response: str | None = None
    processing_time_ms: float | None = None
    source: str = Field(..., description="'cache', 'cache_enriched', 'new_sentiment_only', 'new_full_process', 'error'")

# --- FastAPI App ---
app = FastAPI(
    title="API Phân Tích & Xử Lý Phản Hồi v1.5.0 (Product ID)",
    description="Endpoints: `/sentiment` (nhanh, đọc/lưu KB) và `/process` (AI, đọc/làm giàu KB, hỗ trợ product_id).",
    version="1.5.0"
)

# --- Quản lý Kết nối DB bằng Pooling ---
db_pool = None
db_connection_error = None

def create_db_pool():
    global db_pool, db_connection_error
    required_db_configs = [config.MYSQL_HOST, config.MYSQL_USER, config.MYSQL_PASSWORD, config.MYSQL_DATABASE]
    if not all(required_db_configs):
        db_connection_error = "Thiếu thông tin cấu hình MySQL."
        logger.error(f"DB Pool Error: {db_connection_error}")
        return False
    try:
        logger.info("Tạo MySQL Pool...")
        db_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="sentiment_pool",
            pool_size=5,
            pool_reset_session=True,
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
            connection_timeout=10
        )
        conn_test = db_pool.get_connection()
    except MySQLError as e:
        db_connection_error = f"Lỗi MySQL Pool ({e.errno}): {e.msg}"
        logger.error(f"Lỗi tạo MySQL Pool: {e}", exc_info=True)
        db_pool = None
        return False
    except Exception as e:
        db_connection_error = f"Lỗi không xác định Pool: {e}"
        logger.error(f"Lỗi tạo Pool: {e}", exc_info=True)
        db_pool = None
        return False
    if conn_test and conn_test.is_connected():
        logger.info(f"MySQL Pool OK cho DB '{config.MYSQL_DATABASE}'.")
        conn_test.close()
        db_connection_error = None
        return True
    else:
        db_connection_error = "Không thể tạo kết nối từ Pool."
        logger.error(db_connection_error)
        db_pool = None
        return False

async def get_db_connection():
    if db_pool is None:
        logger.error("Yêu cầu DB nhưng Pool lỗi/chưa tạo.")
        raise HTTPException(503, f"Lỗi DB Pool: {db_connection_error or 'Chưa khởi tạo'}.")
    connection = None
    cursor = None
    log_extra = {'request_id': 'DB_CONN'}
    try:
        connection = db_pool.get_connection()
    except MySQLError as pool_err:
        logger.error(f"Lỗi lấy connection: {pool_err}", exc_info=True, extra=log_extra)
        raise HTTPException(503, f"Lỗi DB Pool: {pool_err.msg}")
    if not connection or not connection.is_connected():
        raise HTTPException(503, "Lỗi connection từ pool.")
    try:
        cursor = connection.cursor(dictionary=True)
        logger.debug("Lấy connection DB OK.", extra=log_extra)
        yield connection, cursor
    finally:
        if cursor:
            cursor.close()
            logger.debug("Đóng DB cursor.", extra=log_extra)
        if connection and connection.is_connected():
            connection.close()
            logger.debug("Trả connection DB về pool.", extra=log_extra)

# --- Tải Model và Cấu hình Dependencies ---
predictor_instance: SentimentPredictor | None = None
gemini_configured = False
model_load_error: str | None = None

async def get_predictor():
    if predictor_instance is None:
        detail_msg = f"Model XLM-R lỗi: {model_load_error or 'Chưa tải'}"
        logger.error(f"Dependency Error: {detail_msg}")
        raise HTTPException(503, detail_msg)
    return predictor_instance

@app.on_event("startup")
async def startup_event():
    global predictor_instance, gemini_configured, model_load_error, db_pool, db_connection_error
    log_extra = {'request_id': 'STARTUP'}
    logger.info("--- API Startup Event ---", extra=log_extra)
    if PREDICTOR_LOADED:
        logger.info(f"Tải XLM-R từ: {config.MODEL_SAVE_PATH}", extra=log_extra)
        start_time = time.time()
        try:
            predictor_instance = SentimentPredictor(model_path=config.MODEL_SAVE_PATH)
        except Exception as e:
            model_load_error = str(e)
            logger.error(f"Lỗi tải model: {e}", exc_info=True, extra=log_extra)
            predictor_instance = None
        if not predictor_instance or not predictor_instance.model or not predictor_instance.label_map:
            model_load_error = f"Lỗi init Predictor từ '{config.MODEL_SAVE_PATH}'."
            predictor_instance = None
            logger.error(model_load_error if model_load_error else "Lỗi tải model", extra=log_extra)
        else:
            logger.info(f"Model XLM-R tải xong: {time.time() - start_time:.2f}s.", extra=log_extra)
    else:
        model_load_error = "Module 'predict' lỗi."
        logger.error(model_load_error, extra=log_extra)
    logger.info("Cấu hình Gemini...", extra=log_extra)
    if config.GEMINI_API_KEY:
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            gemini_configured = True
            logger.info("Gemini OK.", extra=log_extra)
        except Exception as e:
            logger.error(f"Lỗi cấu hình Gemini: {e}", exc_info=True, extra=log_extra)
            gemini_configured = False
    else:
        logger.warning("GEMINI_API_KEY chưa đặt.", extra=log_extra)
        gemini_configured = False
    if not create_db_pool():
        logger.error("!!! LỖI TẠO MYSQL POOL !!!", extra=log_extra)
    logger.info("--- API Startup Hoàn tất ---", extra=log_extra)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("--- API Shutdown ---")
    logger.info("--- API Shutdown Hoàn tất ---")

# --- Middleware ---
def generate_request_id():
    return os.urandom(4).hex()

@app.middleware("http")
async def add_process_time_header_and_handle_errors(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", generate_request_id())
    log_extra = {'request_id': request_id}
    logger.info(f"Nhận request: {request.method} {request.url.path}", extra=log_extra)
    start_time = time.time()
    request.state.request_id = request_id
    response = None
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Lỗi Server Nội bộ: {e}", exc_info=True, extra=log_extra)
        response = JSONResponse(status_code=500, content={"message": "Lỗi server nội bộ.", "request_id": request_id})
    
    process_time = (time.time() - start_time) * 1000
    if response:
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        response.headers["X-Request-ID"] = request_id
        try:
            status_code = response.status_code
        except AttributeError:
            status_code = 'N/A'
        logger.info(f"Hoàn thành request: Status {status_code} trong {process_time:.2f} ms", extra=log_extra)
    else:
        logger.error("Middleware không nhận được response từ call_next", extra=log_extra)
        response = JSONResponse(status_code=500, content={"message": "Lỗi xử lý nội bộ middleware."})
    
    return response

# --- Hàm Gọi Gemini API (Đã cập nhật để nhận product_id) ---
async def get_gemini_suggestions(comment: str, sentiment: str, product_id: str | None, request_id: str = "N/A") -> list[str]:
    task_id = f"{request_id}-sugg"
    log_extra = {'request_id': task_id}
    if not gemini_configured:
        logger.warning("Bỏ qua gợi ý: Gemini chưa cấu hình.", extra=log_extra)
        return ["Gemini chưa cấu hình."]
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Phân tích bình luận và cảm xúc liên quan đến sản phẩm (nếu có).\nĐề xuất 3 hành động cho CSKH.\nBình luận: \"{comment}\"\nCảm xúc: {sentiment}" + (f"\nSản phẩm: {product_id}" if product_id else "") + "\nGợi ý hành động:"
    except Exception as model_err:
        logger.error(f"Lỗi init Gemini Model (sugg): {model_err}", exc_info=True, extra=log_extra)
        return [f"Lỗi AI Init: {type(model_err).__name__}"]
    try:
        logger.info(f"Gửi yêu cầu gợi ý Gemini (Sent: {sentiment}, Prod: {product_id or 'N/A'})", extra=log_extra)
        start_gemini = time.time()
        response = await asyncio.wait_for(model.generate_content_async(prompt), timeout=60.0)
    except asyncio.TimeoutError:
        logger.error(f"Lỗi gọi Gemini (get_suggestions): Timeout", extra=log_extra)
        return ["Lỗi AI Suggestions: Timeout"]
    except Exception as e:
        logger.error(f"Lỗi gọi Gemini (get_suggestions): {e}", exc_info=True, extra=log_extra)
        return [f"Lỗi AI Suggestions: {type(e).__name__}"]
    logger.info(f"Nhận gợi ý Gemini sau {time.time() - start_gemini:.2f}s.", extra=log_extra)
    suggestions_text = response.text.strip()
    suggestions_list = [line.strip().lstrip('0123456789.*- ').strip() for line in suggestions_text.split('\n') if line.strip() and len(line.strip().lstrip('0123456789.*- ').strip()) > 3]
    return suggestions_list if suggestions_list else ["AI không đưa ra gợi ý cụ thể."]

async def generate_gemini_response(comment: str, sentiment: str, product_id: str | None, internal_suggestions: list[str] | None, request_id: str = "N/A") -> str:
    task_id = f"{request_id}-resp"
    log_extra = {'request_id': task_id}
    if not gemini_configured:
        logger.warning("Bỏ qua tạo phản hồi: Gemini chưa cấu hình.", extra=log_extra)
        return "Gemini chưa cấu hình."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt_context = f"Là nhân viên CSKH, soạn phản hồi chuyên nghiệp.\nBình luận: \"{comment}\"\nCảm xúc: {sentiment}" + (f"\nSản phẩm: {product_id}" if product_id else "")
    except Exception as model_err:
        logger.error(f"Lỗi init Gemini Model (resp): {model_err}", exc_info=True, extra=log_extra)
        return f"Lỗi AI Init: {type(model_err).__name__}"
    if internal_suggestions and isinstance(internal_suggestions, list) and not any("Lỗi" in s or "chưa cấu hình" in s for s in internal_suggestions):
        suggestions_text = "\n".join([f"- {s}" for s in internal_suggestions])
        prompt_context += f"\nGợi ý hành động nội bộ (tham khảo): \n{suggestions_text}"
    prompt_instruction = "\nViết nội dung phản hồi cho khách hàng:"
    full_prompt = prompt_context + prompt_instruction
    try:
        logger.info(f"Gửi yêu cầu tạo phản hồi Gemini (Sent: {sentiment}, Prod: {product_id or 'N/A'})", extra=log_extra)
        start_gemini = time.time()
        response = await asyncio.wait_for(model.generate_content_async(full_prompt), timeout=90.0)
    except asyncio.TimeoutError:
        logger.error(f"Lỗi gọi Gemini (generate_response): Timeout", extra=log_extra)
        return "Lỗi tạo phản hồi AI: Timeout"
    except Exception as e:
        logger.error(f"Lỗi gọi Gemini (generate_response): {e}", exc_info=True, extra=log_extra)
        return f"Lỗi tạo phản hồi AI: {type(e).__name__}"
    logger.info(f"Nhận phản hồi tự động Gemini sau {time.time() - start_gemini:.2f}s.", extra=log_extra)
    generated_text = response.text.strip()
    return generated_text if generated_text else "AI không tạo ra phản hồi."

# --- Hàm tương tác Knowledge Base (MySQL) - Đã cập nhật cho product_id ---
async def get_kb_entry_async(comment_hash: str, db_tuple: tuple) -> dict | None:
    connection, cursor = db_tuple
    log_extra = {'request_id': 'KB_GET_ASYNC'}
    if not cursor:
        logger.error("Lỗi DB Cursor (get).", extra=log_extra)
        return None
    try:
        query = "SELECT sentiment, confidence, product_id, suggestions, generated_response FROM knowledge_base WHERE comment_hash = %s"  # Thêm product_id
        def db_query():
            cursor.execute(query, (comment_hash,))
            return cursor.fetchone()
        result = await asyncio.to_thread(db_query)
        if result:
            if result.get('suggestions'):
                try:
                    result['suggestions'] = json.loads(result['suggestions']) if result['suggestions'] else None
                except:
                    result['suggestions'] = ["Lỗi parse suggestions"]
            logger.info(f"Tìm thấy KB entry: {comment_hash}", extra=log_extra)
            return result
        logger.debug(f"Không tìm thấy KB entry: {comment_hash}", extra=log_extra)
        return None
    except MySQLError as e:
        logger.error(f"Lỗi truy vấn KB (get): {e}", exc_info=True, extra=log_extra)
        return None
    except Exception as e:
        logger.error(f"Lỗi KB (get): {e}", exc_info=True, extra=log_extra)
        return None

async def save_or_update_kb_async(data: dict, db_tuple: tuple) -> bool:
    connection, cursor = db_tuple
    log_extra = {'request_id': data.get('request_id', 'KB_SAVE_ASYNC')}
    if not cursor or not connection:
        logger.error("Lỗi DB Conn/Cursor (save).", extra=log_extra)
        return False
    required = ['comment_text', 'comment_hash', 'sentiment']
    if not all(k in data for k in required):
        logger.error("Thiếu dữ liệu lưu KB.")
        return False
    suggestions_json = None
    if data.get('suggestions') is not None:
        try:
            suggestions_json = json.dumps(data['suggestions'], ensure_ascii=False)
        except:
            suggestions_json = json.dumps([str(s) for s in data.get('suggestions', [])], ensure_ascii=False)
    try:
        query = """INSERT INTO knowledge_base (comment_text, comment_hash, sentiment, confidence, product_id, suggestions, generated_response) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s) 
                   ON DUPLICATE KEY UPDATE 
                   sentiment = VALUES(sentiment), 
                   confidence = VALUES(confidence), 
                   product_id = IF(VALUES(product_id) IS NOT NULL, VALUES(product_id), product_id), 
                   suggestions = IF(VALUES(suggestions) IS NOT NULL, VALUES(suggestions), suggestions), 
                   generated_response = IF(VALUES(generated_response) IS NOT NULL, VALUES(generated_response), generated_response), 
                   last_used_at = CURRENT_TIMESTAMP;"""
        values = (
            data['comment_text'],
            data['comment_hash'],
            data['sentiment'],
            data.get('confidence'),
            data.get('product_id'),
            suggestions_json,
            data.get('generated_response')
        )
        def db_write():
            cursor.execute(query, values)
            connection.commit()
        await asyncio.to_thread(db_write)
        logger.info(f"Đã lưu/cập nhật KB: {data['comment_hash']}", extra=log_extra)
        return True
    except MySQLError as e:
        logger.error(f"Lỗi ghi/cập nhật KB: {e}", exc_info=True, extra=log_extra)
        connection.rollback()
        return False
    except Exception as e:
        logger.error(f"Lỗi KB (save/update): {e}", exc_info=True, extra=log_extra)
        connection.rollback()
        return False

# --- API Endpoints ---

@app.get("/", tags=["General"])
async def read_root(db: tuple = Depends(get_db_connection)):
    connection, _ = db
    model_status = "Sẵn sàng" if predictor_instance else f"Lỗi ({model_load_error or 'Unknown'})"
    gemini_status = "Đã cấu hình" if gemini_configured else "Chưa cấu hình"
    db_connected_status = "Đã kết nối" if connection and connection.is_connected() else f"Lỗi DB ({db_connection_error or 'Chưa tạo Pool'})"
    return {
        "message": "API Phân Tích & Xử Lý Phản Hồi",
        "model_status": model_status,
        "gemini_status": gemini_status,
        "db_status": db_connected_status
    }

@app.post("/sentiment/", response_model=UnifiedResponse, tags=["Sentiment Analysis Only"])
async def analyze_sentiment_only_kb_product(
    request: SentimentRequest,
    predictor: SentimentPredictor = Depends(get_predictor),
    db: tuple = Depends(get_db_connection),
    http_request: Request = None
):
    """Phân tích cảm xúc (local), đọc KB, lưu kết quả cơ bản vào KB nếu chưa có. Hỗ trợ product_id."""
    request_id = getattr(http_request.state, 'request_id', 'N/A')
    log_extra = {'request_id': request_id}
    logger.info(f"Nhận /sentiment/ req: {request.comment[:100]} (Prod: {request.product_id or 'N/A'})...", extra=log_extra)
    start_req_time = time.time()
    if not request.comment:
        raise HTTPException(400, "Bình luận trống.")
    comment_text = request.comment
    product_id = request.product_id
    try:
        comment_hash = hashlib.sha256(comment_text.encode('utf-8')).hexdigest()
    except Exception:
        raise HTTPException(500, "Lỗi hashing.")

    cached_entry = await get_kb_entry_async(comment_hash, db)
    if cached_entry:
        processing_time_ms = (time.time() - start_req_time) * 1000
        logger.info(f"Trả về từ Cache (/sentiment/) trong {processing_time_ms:.2f} ms.", extra=log_extra)
        return UnifiedResponse(
            sentiment=cached_entry['sentiment'],
            confidence=cached_entry.get('confidence'),
            product_id_processed=cached_entry.get('product_id'),
            ai_call_reason="Đã xử lý trước đó (Cache)",
            suggestions=cached_entry.get('suggestions'),
            generated_response=cached_entry.get('generated_response'),
            processing_time_ms=processing_time_ms,
            source="cache"
        )

    logger.info("Không tìm thấy trong KB (/sentiment/), phân tích mới...", extra=log_extra)
    sentiment_label, confidence = None, None
    try:
        sentiment_label, confidence, _ = predictor.predict_single(comment_text)
    except Exception as pred_err:
        logger.error(f"Lỗi predict_single: {pred_err}", exc_info=True, extra=log_extra)
        raise HTTPException(500, "Lỗi phân tích cảm xúc.")
    if sentiment_label is None:
        logger.error("Predict_single trả về None.")
        raise HTTPException(500, "Lỗi phân tích cảm xúc.")
    logger.info(f"Kết quả XLM-R (/sentiment/): {sentiment_label} (Conf: {confidence:.4f})", extra=log_extra)

    kb_data = {
        "comment_text": comment_text,
        "comment_hash": comment_hash,
        "sentiment": sentiment_label,
        "confidence": confidence,
        "product_id": product_id,
        "suggestions": None,
        "generated_response": None,
        "request_id": request_id
    }
    save_success = await save_or_update_kb_async(kb_data, db)
    if not save_success:
        logger.error("Không thể lưu /sentiment/ vào KB.", extra=log_extra)

    processing_time_ms = (time.time() - start_req_time) * 1000
    logger.info(f"Hoàn thành /sentiment/ (xử lý mới) trong {processing_time_ms:.2f} ms.", extra=log_extra)
    return UnifiedResponse(
        sentiment=sentiment_label,
        confidence=confidence,
        product_id_processed=product_id,
        ai_call_reason="Chỉ phân tích Sentiment",
        suggestions=None,
        generated_response=None,
        processing_time_ms=processing_time_ms,
        source="new_sentiment_only"
    )

@app.post("/process/", response_model=UnifiedResponse, tags=["Full Processing (KB + Gemini)"])
async def process_comment_kb_enrich_product(
    request: SentimentRequest,
    predictor: SentimentPredictor = Depends(get_predictor),
    db: tuple = Depends(get_db_connection),
    http_request: Request = None
):
    """Xử lý đầy đủ: Kiểm tra KB, nếu thiếu AI hoặc product_id không khớp -> XLM-R -> Gemini -> Cập nhật/Lưu KB. Hỗ trợ product_id."""
    request_id = getattr(http_request.state, 'request_id', 'N/A')
    log_extra = {'request_id': request_id}
    logger.info(f"Nhận /process/ req: {request.comment[:100]} (Prod: {request.product_id or 'N/A'})...", extra=log_extra)
    start_req_time = time.time()
    if not request.comment:
        raise HTTPException(400, "Bình luận trống.")
    comment_text = request.comment
    product_id = request.product_id
    try:
        comment_hash = hashlib.sha256(comment_text.encode('utf-8')).hexdigest()
    except Exception:
        raise HTTPException(500, "Lỗi hashing.")

    cached_entry = await get_kb_entry_async(comment_hash, db)
    sentiment_label, confidence = None, None
    internal_suggestions, auto_response = None, None
    ai_call_reason = "N/A"
    source = "new_full_process"

    if cached_entry:
        logger.info(f"Tìm thấy KB (/process/). Cache ProdID: {cached_entry.get('product_id')}, Req ProdID: {product_id}", extra=log_extra)
        sentiment_label = cached_entry['sentiment']
        confidence = cached_entry.get('confidence')
        has_ai_data = cached_entry.get('suggestions') is not None and cached_entry.get('generated_response') is not None
        product_id_match = (product_id == cached_entry.get('product_id')) or \
                           (product_id is None and cached_entry.get('product_id') is None)
        if has_ai_data and product_id_match:
            logger.info("KB đủ AI & product_id khớp. Trả cache.", extra=log_extra)
            internal_suggestions = cached_entry['suggestions']
            auto_response = cached_entry['generated_response']
            ai_call_reason = "Cache"
            source = "cache"
        else:
            if not product_id_match:
                ai_call_reason = f"Làm giàu KB (Product ID mới: {product_id})"
            else:
                ai_call_reason = "Làm giàu KB (Thiếu AI)"
            logger.info(f"KB chưa đủ/product_id khác. Gọi Gemini. Lý do: {ai_call_reason}", extra=log_extra)
            source = "cache_enriched"
    else:
        logger.info("Không tìm thấy trong KB (/process/), phân tích mới...", extra=log_extra)
        ai_call_reason = "Xử lý mới (Gọi AI)"
        source = "new_full_process"
        try:
            sentiment_label, confidence, _ = predictor.predict_single(comment_text)
        except Exception as pred_err:
            logger.error(f"Lỗi predict_single: {pred_err}", exc_info=True, extra=log_extra)
            raise HTTPException(500, "Lỗi phân tích cảm xúc.")
        if sentiment_label is None:
            logger.error("Predict_single trả về None.")
            raise HTTPException(500, "Lỗi phân tích cảm xúc.")
        logger.info(f"XLM-R Result: {sentiment_label} (Conf: {confidence:.4f})", extra=log_extra)

    if source != "cache":
        if gemini_configured:
            logger.info(f"Gọi Gemini. Lý do: {ai_call_reason}", extra=log_extra)
            try:
                task1 = get_gemini_suggestions(comment_text, sentiment_label, product_id, request_id)
                task2 = generate_gemini_response(comment_text, sentiment_label, product_id, None, request_id)
                results = await asyncio.gather(task1, task2, return_exceptions=True)
                internal_suggestions = results[0] if not isinstance(results[0], Exception) else [f"Lỗi AI Sugg: {type(results[0]).__name__}"]
                auto_response = results[1] if not isinstance(results[1], Exception) else f"Lỗi AI Resp: {type(results[1]).__name__}"
                if isinstance(results[0], Exception):
                    logger.error(f"Lỗi task gợi ý: {results[0]}", exc_info=results[0], extra=log_extra)
                if isinstance(results[1], Exception):
                    logger.error(f"Lỗi task phản hồi: {results[1]}", exc_info=results[1], extra=log_extra)
            except Exception as gather_err:
                logger.error(f"Lỗi gather Gemini: {gather_err}", exc_info=True, extra=log_extra)
                internal_suggestions = ['Lỗi hệ thống AI.']
                auto_response = 'Lỗi hệ thống AI.'
        else:
            logger.warning("Gemini chưa cấu hình.", extra=log_extra)
            ai_call_reason += " (Chưa cấu hình)"
            internal_suggestions = ["Gemini chưa được cấu hình."]
            auto_response = "Gemini chưa được cấu hình."

        if sentiment_label is not None:
            logger.info("Lưu/Cập nhật kết quả đầy đủ vào KB...", extra=log_extra)
            kb_data = {
                "comment_text": comment_text,
                "comment_hash": comment_hash,
                "sentiment": sentiment_label,
                "confidence": confidence,
                "product_id": product_id,
                "suggestions": internal_suggestions,
                "generated_response": auto_response,
                "request_id": request_id
            }
            save_success = await save_or_update_kb_async(kb_data, db)
            if not save_success:
                logger.error("Không thể lưu/cập nhật KB.", extra=log_extra)

    processing_time_ms = (time.time() - start_req_time) * 1000
    logger.info(f"Hoàn thành /process/ (source: {source}) trong {processing_time_ms:.2f} ms.", extra=log_extra)
    return UnifiedResponse(
        sentiment=sentiment_label,
        confidence=confidence,
        product_id_processed=product_id,
        ai_call_reason=ai_call_reason,
        suggestions=internal_suggestions,
        generated_response=auto_response,
        processing_time_ms=processing_time_ms,
        source=source
    )

# --- Chạy API Server ---
if __name__ == "__main__":
    logger.info("--- Khởi chạy FastAPI Server (KB Chung - Product ID) ---")
    if not PREDICTOR_LOADED or predictor_instance is None:
        logger.error("!!! Model XLM-R lỗi tải !!!")
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=False, log_level="info")