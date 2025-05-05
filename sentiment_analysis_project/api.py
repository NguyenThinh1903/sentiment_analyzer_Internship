# api.py (Phiên bản v1.3.2 - Luôn gọi Gemini, Logging chi tiết, Xử lý Edge Cases)

import time
import os
import traceback
import logging
from dotenv import load_dotenv
import asyncio

# Tải biến môi trường từ file .env (nếu có)
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn
import google.generativeai as genai

# Import các thành phần cần thiết từ dự án
import config
try:
    from predict import SentimentPredictor
    PREDICTOR_LOADED = True
except ImportError:
    print("LỖI: Không tìm thấy module 'predict'. SentimentPredictor sẽ không hoạt động.")
    SentimentPredictor = None # Đặt là None để kiểm tra sau
    PREDICTOR_LOADED = False

# --- Cấu hình Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - RID=%(request_id)s - %(message)s') # Thêm request_id vào format
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Xóa handler cũ nếu có để tránh log trùng lặp khi reload
if logger.hasHandlers():
    logger.handlers.clear()
# Handler mới
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
# Thêm filter để thêm request_id vào log record
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        # Cố gắng lấy request_id từ task local hoặc context (cần xử lý cẩn thận trong async)
        # Cách đơn giản hơn là truyền request_id vào logger khi gọi
        record.request_id = getattr(record, 'request_id', 'N/A') # Gán mặc định nếu chưa có
        return True
logger.addFilter(RequestIdFilter())


# --- Pydantic Models (Giữ nguyên) ---
class SentimentRequest(BaseModel):
    comment: str = Field(..., description="Bình luận cần phân tích.")
    @field_validator('comment')
    @classmethod
    def strip_comment(cls, value: str) -> str:
        return value.strip()

class SentimentOnlyResponse(BaseModel):
    sentiment: str = Field(...)
    confidence: float | None = Field(None)
    model_used: str = Field(...)
    processing_time_ms: float | None = Field(None)

class ProcessResponse(BaseModel):
    sentiment: str = Field(...)
    confidence: float | None = Field(None)
    ai_call_reason: str | None = Field(None) # Vẫn giữ để biết AI được gọi
    suggestions: list[str] | None = Field(None)
    generated_response: str | None = Field(None)
    processing_time_ms: float | None = Field(None)

# --- Khởi tạo FastAPI App ---
app = FastAPI(
    title="API Phân Tích & Xử Lý Phản Hồi v1.3.2",
    description="Endpoints: `/sentiment` (nhanh, chỉ XLM-R) và `/process` (luôn gọi Gemini).",
    version="1.3.2"
)

# --- Tải Model và Cấu hình Dependencies ---
predictor_instance: SentimentPredictor | None = None
gemini_configured = False
model_load_error: str | None = None

async def get_predictor():
    if predictor_instance is None:
        detail_msg = f"Model XLM-R chưa sẵn sàng. Lỗi: {model_load_error or 'Unknown'}"
        logger.error(f"Dependency Error: {detail_msg}")
        raise HTTPException(status_code=503, detail=detail_msg)
    return predictor_instance

async def check_gemini_config():
    if not gemini_configured:
        logger.warning("Dependency Error: Gemini API chưa cấu hình.")
        raise HTTPException(status_code=501, detail="Chức năng Gemini chưa cấu hình (thiếu API Key?).")
    return True

@app.on_event("startup")
async def startup_event():
    # ... (Code startup event giữ nguyên như trước) ...
    global predictor_instance, gemini_configured, model_load_error
    logger.info("--- API Startup Event ---")
    if PREDICTOR_LOADED:
        logger.info(f"Đang tải model XLM-R từ: {config.MODEL_SAVE_PATH}")
        start_time = time.time()
        try:
            predictor_instance = SentimentPredictor(model_path=config.MODEL_SAVE_PATH)
            if not predictor_instance or not predictor_instance.model or not predictor_instance.label_map:
                 model_load_error = f"Không thể khởi tạo Predictor từ '{config.MODEL_SAVE_PATH}'."
                 logger.error(f"Lỗi tải model XLM-R: {model_load_error}")
                 predictor_instance = None
            else: logger.info(f"Model XLM-R tải xong sau {time.time() - start_time:.2f} giây.")
        except Exception as e: model_load_error = str(e); logger.error(f"Lỗi nghiêm trọng khi tải model: {e}", exc_info=True); predictor_instance = None
    else: model_load_error = "Module 'predict' hoặc class 'SentimentPredictor' không tìm thấy."; logger.error(model_load_error)
    logger.info("Đang cấu hình Gemini API...")
    if config.GEMINI_API_KEY:
        try: genai.configure(api_key=config.GEMINI_API_KEY); gemini_configured = True; logger.info("Gemini API cấu hình thành công.")
        except Exception as e: logger.error(f"Lỗi cấu hình Gemini API: {e}", exc_info=True); gemini_configured = False
    else: logger.warning("GEMINI_API_KEY chưa đặt. Chức năng Gemini bị vô hiệu hóa."); gemini_configured = False
    logger.info("--- API Startup Hoàn tất ---")

# --- Middleware ---
# Tạo request ID đơn giản
def generate_request_id():
     return os.urandom(4).hex() # ID ngắn hơn

@app.middleware("http")
async def add_process_time_header_and_handle_errors(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", generate_request_id())
    # Tạo extra dict để truyền request_id vào logger
    log_extra = {'request_id': request_id}
    logger.info(f"Nhận request: {request.method} {request.url.path}", extra=log_extra)
    start_time = time.time()
    try:
        # Gán request_id vào state để các hàm khác có thể truy cập nếu cần (cẩn thận với async)
        request.state.request_id = request_id
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        response.headers["X-Request-ID"] = request_id
        logger.info(f"Hoàn thành request: Status {response.status_code} trong {process_time:.2f} ms", extra=log_extra)
        return response
    except HTTPException as http_err:
        logger.warning(f"HTTP Exception: {http_err.status_code} - {http_err.detail}", extra=log_extra)
        raise http_err
    except Exception as e:
        logger.error(f"Lỗi Server Nội bộ không mong muốn: {e}", exc_info=True, extra=log_extra)
        return JSONResponse(
            status_code=500,
            content={"message": "Lỗi server nội bộ.", "request_id": request_id},
            headers={"X-Request-ID": request_id}
        )


# --- Hàm Gọi Gemini API ---
# Thêm request_id vào log
async def get_gemini_suggestions(comment: str, sentiment: str, request_id: str = "N/A") -> list[str]:
    task_id = f"{request_id}-sugg"
    log_extra = {'request_id': task_id} # Dùng ID riêng cho task
    if not gemini_configured:
        logger.warning("Bỏ qua gợi ý: Gemini chưa cấu hình.", extra=log_extra)
        return ["Gemini chưa cấu hình."]
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Phân tích bình luận và cảm xúc sau, đề xuất 3 hành động cho CSKH, dạng danh sách đánh số.
Bình luận: "{comment}"
Cảm xúc: {sentiment}
Gợi ý hành động:"""
        logger.info(f"Gửi yêu cầu gợi ý đến Gemini (Sentiment: {sentiment})", extra=log_extra)
        start_gemini = time.time()
        response = await asyncio.wait_for(model.generate_content_async(prompt), timeout=60.0)
        logger.info(f"Nhận phản hồi gợi ý từ Gemini sau {time.time() - start_gemini:.2f} giây.", extra=log_extra)
        suggestions_text = response.text.strip()
        suggestions_list = [
            line.strip().lstrip('0123456789.*- ').strip()
            for line in suggestions_text.split('\n')
            if line.strip() and len(line.strip().lstrip('0123456789.*- ').strip()) > 3
        ]
        return suggestions_list if suggestions_list else ["AI không đưa ra gợi ý cụ thể."]
    except asyncio.TimeoutError:
        logger.error(f"Lỗi gọi Gemini (get_suggestions): Timeout", extra=log_extra)
        return ["Lỗi AI Suggestions: Timeout"]
    except Exception as e:
        logger.error(f"Lỗi gọi Gemini (get_suggestions): {e}", exc_info=True, extra=log_extra)
        return [f"Lỗi AI Suggestions: {type(e).__name__}"]

async def generate_gemini_response(comment: str, sentiment: str, internal_suggestions: list[str] | None, request_id: str = "N/A") -> str:
    task_id = f"{request_id}-resp"
    log_extra = {'request_id': task_id}
    if not gemini_configured:
        logger.warning("Bỏ qua tạo phản hồi: Gemini chưa cấu hình.", extra=log_extra)
        return "Gemini chưa cấu hình."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt_context = f"""Là nhân viên CSKH, soạn phản hồi chuyên nghiệp, đồng cảm cho khách hàng.
Bình luận khách hàng: "{comment}"
Cảm xúc phân tích: {sentiment}"""
        if internal_suggestions and isinstance(internal_suggestions, list) and not any("Lỗi" in s or "chưa cấu hình" in s for s in internal_suggestions):
             suggestions_text = "\n".join([f"- {s}" for s in internal_suggestions])
             prompt_context += f"\nGợi ý hành động nội bộ (tham khảo): \n{suggestions_text}"
        prompt_instruction = "\nViết nội dung phản hồi cho khách hàng:"
        full_prompt = prompt_context + prompt_instruction
        logger.info(f"Gửi yêu cầu tạo phản hồi đến Gemini (Sentiment: {sentiment})", extra=log_extra)
        start_gemini = time.time()
        response = await asyncio.wait_for(model.generate_content_async(full_prompt), timeout=90.0)
        logger.info(f"Nhận phản hồi tự động từ Gemini sau {time.time() - start_gemini:.2f} giây.", extra=log_extra)
        generated_text = response.text.strip()
        return generated_text if generated_text else "AI không tạo ra phản hồi."
    except asyncio.TimeoutError:
        logger.error(f"Lỗi gọi Gemini (generate_response): Timeout", extra=log_extra)
        return "Lỗi tạo phản hồi AI: Timeout"
    except Exception as e:
        logger.error(f"Lỗi gọi Gemini (generate_response): {e}", exc_info=True, extra=log_extra)
        return f"Lỗi tạo phản hồi AI: {type(e).__name__}"


# --- API Endpoints ---

@app.get("/", summary="Kiểm tra Trạng thái API", tags=["General"])
async def read_root():
    """Endpoint gốc kiểm tra API và trạng thái các model."""
    model_status = "Sẵn sàng" if predictor_instance else f"Lỗi ({model_load_error or 'Unknown'})"
    gemini_status = "Đã cấu hình" if gemini_configured else "Chưa cấu hình"
    return {"message": "API Phân Tích & Xử Lý Phản Hồi", "model_status": model_status, "gemini_status": gemini_status}

@app.post("/sentiment/", response_model=SentimentOnlyResponse, tags=["Sentiment Analysis Only"])
async def analyze_sentiment_only(
    request: SentimentRequest,
    predictor: SentimentPredictor = Depends(get_predictor),
    http_request: Request = None
):
    """Chỉ phân tích cảm xúc bằng model local (nhanh)."""
    request_id = getattr(http_request.state, 'request_id', 'N/A')
    log_extra = {'request_id': request_id}
    logger.info(f"Nhận yêu cầu /sentiment/ cho: {request.comment[:100]}...", extra=log_extra)
    start_req_time = time.time()
    try:
        if not request.comment: raise HTTPException(status_code=400, detail="Bình luận không được để trống.")
        sentiment_label, confidence, _ = predictor.predict_single(request.comment)
        if sentiment_label is None: raise HTTPException(status_code=500, detail="Lỗi nội bộ khi phân tích cảm xúc.")
        processing_time_ms = (time.time() - start_req_time) * 1000
        logger.info(f"Kết quả /sentiment/: {sentiment_label} (Conf: {confidence:.4f}) trong {processing_time_ms:.2f} ms", extra=log_extra)
        return SentimentOnlyResponse(
            sentiment=sentiment_label, confidence=confidence,
            model_used="local_xlmr", processing_time_ms=processing_time_ms
        )
    except HTTPException as http_err: raise http_err
    except Exception as e:
        logger.error(f"Lỗi trong /sentiment/: {e}", exc_info=True, extra=log_extra)
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi phân tích cảm xúc.")


@app.post("/process/", response_model=ProcessResponse, tags=["Full Processing (Gemini Always)"])
async def process_comment_full_ai( # Đổi tên hàm cho rõ
    request: SentimentRequest,
    predictor: SentimentPredictor = Depends(get_predictor),
    http_request: Request = None # Inject request để lấy request_id
):
    """
    Xử lý bình luận: Luôn gọi Gemini để lấy gợi ý và tạo phản hồi.
    """
    request_id = getattr(http_request.state, 'request_id', 'N/A')
    log_extra = {'request_id': request_id}
    logger.info(f"Nhận yêu cầu /process/ (luôn gọi AI) cho: {request.comment[:100]}...", extra=log_extra)
    start_req_time = time.time()

    # --- Bước 1: Phân tích Cảm xúc (Luôn chạy) ---
    sentiment_label, confidence, _ = None, None, None
    try:
        if not request.comment: raise HTTPException(status_code=400, detail="Bình luận trống.")
        sentiment_label, confidence, _ = predictor.predict_single(request.comment)
        if sentiment_label is None: raise ValueError("Predict_single trả về None")
        logger.info(f"XLM-R Result: {sentiment_label} (Conf: {confidence:.4f})", extra=log_extra)
    except HTTPException as http_err: raise http_err
    except Exception as pred_err:
        logger.error(f"Lỗi predict_single: {pred_err}", exc_info=True, extra=log_extra)
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi phân tích cảm xúc.")

    # --- Bước 2: Luôn Gọi Gemini (Nếu đã cấu hình) ---
    internal_suggestions = None
    auto_response = None
    ai_call_reason = "Luôn yêu cầu xử lý AI" # Lý do cố định

    if gemini_configured:
        logger.info(f"Luôn gọi Gemini...", extra=log_extra)
        try:
            # Chạy song song
            task1 = get_gemini_suggestions(request.comment, sentiment_label, request_id)
            task2 = generate_gemini_response(request.comment, sentiment_label, None, request_id) # Tạm thời không truyền sugg khi chạy song song
            results = await asyncio.gather(task1, task2, return_exceptions=True)
            logger.info(f"Hoàn thành gọi Gemini song song.", extra=log_extra)
            # Xử lý kết quả/lỗi
            internal_suggestions = results[0] if not isinstance(results[0], Exception) else [f"Lỗi AI Suggestions: {type(results[0]).__name__}"]
            auto_response = results[1] if not isinstance(results[1], Exception) else f"Lỗi tạo phản hồi AI: {type(results[1]).__name__}"
            if isinstance(results[0], Exception): logger.error(f"Lỗi task gợi ý: {results[0]}", exc_info=results[0], extra=log_extra)
            if isinstance(results[1], Exception): logger.error(f"Lỗi task phản hồi: {results[1]}", exc_info=results[1], extra=log_extra)
        except Exception as gather_err:
            logger.error(f"Lỗi nghiêm trọng khi gọi Gemini song song: {gather_err}", exc_info=True, extra=log_extra)
            internal_suggestions = ["Lỗi hệ thống gọi AI."]
            auto_response = "Lỗi hệ thống gọi AI."
    else:
        logger.warning("Gemini chưa cấu hình.", extra=log_extra)
        ai_call_reason += " (Gemini chưa cấu hình)"
        internal_suggestions = ["Gemini chưa được cấu hình."]
        auto_response = "Gemini chưa được cấu hình."


    end_req_time = time.time()
    processing_time_ms = (end_req_time - start_req_time) * 1000
    logger.info(f"Xử lý /process/ hoàn tất trong {processing_time_ms:.2f} ms.", extra=log_extra)

    # --- Bước 3: Trả về Kết quả ---
    return ProcessResponse(
        sentiment=sentiment_label,
        confidence=confidence,
        ai_call_reason=ai_call_reason,
        suggestions=internal_suggestions,
        generated_response=auto_response,
        processing_time_ms=processing_time_ms
    )

# --- Chạy API Server ---
if __name__ == "__main__":
    logger.info("--- Khởi chạy FastAPI Server (Luôn gọi Gemini) ---")
    if not os.path.isdir(config.MODEL_SAVE_PATH):
        logger.warning(f"!!! Không tìm thấy thư mục model '{config.MODEL_SAVE_PATH}'. !!!")
    # Chạy không reload để tránh lỗi re-init logger nhiều lần khi debug
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=False, log_level="info")