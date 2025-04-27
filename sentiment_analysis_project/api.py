# api.py (Phiên bản Lai ghép - Gọi Gemini có điều kiện)

import time
import os
import traceback
import logging
from dotenv import load_dotenv
import asyncio # Import asyncio để chạy song song nếu cần

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import google.generativeai as genai

import config
from predict import SentimentPredictor

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models (Giữ nguyên như trước) ---
class SentimentRequest(BaseModel):
    comment: str = Field(..., min_length=1, description="Nội dung bình luận cần phân tích cảm xúc.")

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Nhãn cảm xúc dự đoán.")
    confidence: float | None = Field(None, ge=0, le=1, description="Độ tin cậy của dự đoán (từ model local).")
    suggestions: list[str] | None = Field(None, description="Danh sách gợi ý hành động nội bộ (từ AI nếu được gọi).")
    generated_response: str | None = Field(None, description="Nội dung phản hồi tự động gợi ý (từ AI nếu được gọi).")
    processing_time_ms: float | None = Field(None, description="Thời gian xử lý yêu cầu (ms).")
    ai_call_reason: str | None = Field(None, description="Lý do gọi AI (nếu có).") # Thêm trường để biết tại sao gọi AI

# --- Khởi tạo FastAPI App ---
app = FastAPI(
    title="API Phân Tích & Xử Lý Phản Hồi (Lai ghép)",
    description="Sử dụng model XLM-R và gọi Gemini có điều kiện để phân tích, đề xuất hành động và tạo phản hồi.",
    version="1.2.0" # Tăng version
)

# --- Tải Model và Cấu hình Gemini ---
predictor_instance = None
gemini_configured = False

@app.on_event("startup")
async def startup_event():
    global predictor_instance, gemini_configured
    logger.info("--- Khởi động API ---")
    # 1. Tải model XLM-R
    logger.info(f"Đang tải model XLM-R từ: {config.MODEL_SAVE_PATH}")
    start_time = time.time()
    try:
        predictor_instance = SentimentPredictor(model_path=config.MODEL_SAVE_PATH)
        if not predictor_instance or not predictor_instance.model or not predictor_instance.label_map:
             logger.error(f"Lỗi tải model XLM-R từ '{config.MODEL_SAVE_PATH}'.", exc_info=True)
             predictor_instance = None
        else: logger.info(f"Model XLM-R tải xong sau {time.time() - start_time:.2f} giây.")
    except Exception as e: logger.error(f"Lỗi nghiêm trọng khi tải model: {e}", exc_info=True); predictor_instance = None

    # 2. Cấu hình Gemini
    logger.info("Đang cấu hình Gemini API...")
    if config.GEMINI_API_KEY:
        try: genai.configure(api_key=config.GEMINI_API_KEY); gemini_configured = True; logger.info("Gemini API cấu hình thành công.")
        except Exception as e: logger.error(f"Lỗi cấu hình Gemini: {e}", exc_info=True); gemini_configured = False
    else: logger.warning("GEMINI_API_KEY chưa đặt. Chức năng Gemini bị vô hiệu hóa."); gemini_configured = False

# --- Middleware (Giữ nguyên) ---
@app.middleware("http")
async def add_process_time_header_and_handle_errors(request: Request, call_next):
    # ... (Code middleware như trước) ...
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response
    except HTTPException as http_err:
        logger.warning(f"HTTP Exception: {http_err.status_code} - {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.error(f"Lỗi Server Nội bộ không mong muốn: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Lỗi server nội bộ."})


# --- Hàm Gọi Gemini API (Giữ nguyên như trước) ---
async def get_gemini_suggestions(comment: str, sentiment: str) -> list[str] | None:
    # ... (Code hàm get_gemini_suggestions như trước, kiểm tra gemini_configured) ...
    if not gemini_configured: return ["Gemini chưa cấu hình."]
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Phân tích bình luận và cảm xúc sau, đề xuất 3 hành động cho CSKH, dạng danh sách đánh số.
Bình luận: "{comment}"
Cảm xúc: {sentiment}
Gợi ý hành động:""" # Prompt rút gọn
        logger.info(f"Gửi yêu cầu gợi ý đến Gemini (Sentiment: {sentiment})")
        response = await model.generate_content_async(prompt, request_options={'timeout': 60}) # Thêm timeout
        suggestions_text = response.text.strip()
        suggestions_list = [
            line.strip().lstrip('0123456789.*- ').strip()
            for line in suggestions_text.split('\n')
            if line.strip() and len(line.strip().lstrip('0123456789.*- ').strip()) > 3 # Lọc kỹ hơn
        ]
        logger.info(f"Nhận được {len(suggestions_list)} gợi ý từ Gemini.")
        return suggestions_list if suggestions_list else ["AI không đưa ra gợi ý cụ thể."]
    except Exception as e:
        logger.error(f"Lỗi gọi Gemini (get_suggestions): {e}", exc_info=True)
        return [f"Lỗi AI Suggestions: {type(e).__name__}"]


async def generate_gemini_response(comment: str, sentiment: str, internal_suggestions: list[str] | None = None) -> str | None:
    # ... (Code hàm generate_gemini_response như trước, kiểm tra gemini_configured) ...
    if not gemini_configured: return "Gemini chưa cấu hình."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt_context = f"""Là nhân viên CSKH, soạn phản hồi chuyên nghiệp, đồng cảm cho khách hàng dựa trên bình luận và cảm xúc.
Bình luận khách hàng: "{comment}"
Cảm xúc phân tích: {sentiment}"""
        if internal_suggestions:
             suggestions_text = "\n".join([f"- {s}" for s in internal_suggestions])
             prompt_context += f"\nGợi ý hành động nội bộ (tham khảo): \n{suggestions_text}"
        prompt_instruction = "\nViết nội dung phản hồi cho khách hàng (không cần lời chào/kết quá trang trọng):"
        full_prompt = prompt_context + prompt_instruction
        logger.info(f"Gửi yêu cầu tạo phản hồi đến Gemini (Sentiment: {sentiment})")
        response = await model.generate_content_async(full_prompt, request_options={'timeout': 90}) # Thêm timeout
        logger.info("Nhận được phản hồi tự động từ Gemini.")
        generated_text = response.text.strip()
        # Có thể thêm bộ lọc/kiểm tra nội dung nhạy cảm ở đây nếu cần
        return generated_text if generated_text else "AI không tạo ra phản hồi."
    except Exception as e:
        logger.error(f"Lỗi gọi Gemini (generate_response): {e}", exc_info=True)
        return f"Lỗi tạo phản hồi AI: {type(e).__name__}"


# --- API Endpoint Chính (Logic Lai ghép) ---
@app.post("/process_comment_hybrid/", response_model=SentimentResponse, tags=["Comment Processing Hybrid"])
async def process_customer_comment_hybrid(request: SentimentRequest):
    """
    Xử lý Bình luận (Lai ghép):
    1. Phân tích cảm xúc (XLM-R).
    2. Gọi Gemini CHỌN LỌC dựa trên độ tin cậy/cảm xúc.
    3. Trả về kết quả.
    """
    start_req_time = time.time()
    logger.info(f"Nhận HYBRID request: {request.comment[:100]}...")

    if predictor_instance is None:
        logger.error("Model XLM-R chưa tải.")
        raise HTTPException(status_code=503, detail="Model phân tích cảm xúc chưa sẵn sàng.")

    # --- Bước 1: Phân tích Cảm xúc (Luôn chạy) ---
    sentiment_label, confidence, _ = None, None, None # Khởi tạo
    try:
        sentiment_label, confidence, _ = predictor_instance.predict_single(request.comment)
        if sentiment_label is None: raise ValueError("Predict_single trả về None") # Xử lý trường hợp này
        logger.info(f"XLM-R Result: {sentiment_label} (Conf: {confidence:.4f})")
    except Exception as pred_err:
        logger.error(f"Lỗi predict_single: {pred_err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi phân tích cảm xúc.")

    # --- Bước 2: Quyết định Gọi Gemini ---
    should_call_gemini = False
    ai_call_reason = "Độ tin cậy cao / Không thuộc TH đặc biệt" # Mặc định
    negative_label_value = config.TARGET_LABEL_MAP.get(0, "Tiêu cực") # Lấy đúng tên nhãn tiêu cực

    if confidence is not None and confidence < config.CONFIDENCE_THRESHOLD:
        should_call_gemini = True
        ai_call_reason = f"Độ tin cậy thấp ({confidence:.4f} < {config.CONFIDENCE_THRESHOLD})"
    elif config.ALWAYS_CHECK_NEGATIVE and sentiment_label == negative_label_value:
         should_call_gemini = True
         ai_call_reason = "Cảm xúc Tiêu cực"
    # Thêm điều kiện kiểm tra Trung tính nếu cần
    # elif config.ALWAYS_CHECK_NEUTRAL and sentiment_label == config.TARGET_LABEL_MAP.get(1, "Trung tính"):
    #    should_call_gemini = True
    #    ai_call_reason = "Cảm xúc Trung tính"

    # --- Bước 3: Gọi Gemini (Nếu cần và đã cấu hình) ---
    internal_suggestions = None
    auto_response = None

    if should_call_gemini:
        if gemini_configured:
            logger.info(f"Quyết định gọi Gemini. Lý do: {ai_call_reason}")
            # Chạy song song để tiết kiệm thời gian nếu có thể
            try:
                logger.info("Bắt đầu gọi Gemini song song (suggestions & response)...")
                suggestions_task = get_gemini_suggestions(request.comment, sentiment_label)
                # Tạo response có thể không cần suggestions, chạy song song được
                response_task = generate_gemini_response(request.comment, sentiment_label, None)
                internal_suggestions, auto_response = await asyncio.gather(suggestions_task, response_task, return_exceptions=True)
                logger.info("Hoàn thành gọi Gemini song song.")

                # Xử lý nếu có lỗi trong asyncio.gather
                if isinstance(internal_suggestions, Exception):
                     logger.error(f"Lỗi task gợi ý: {internal_suggestions}")
                     internal_suggestions = [f"Lỗi AI Suggestions: {type(internal_suggestions).__name__}"]
                if isinstance(auto_response, Exception):
                     logger.error(f"Lỗi task phản hồi: {auto_response}")
                     auto_response = f"Lỗi tạo phản hồi AI: {type(auto_response).__name__}"

            except Exception as gather_err: # Bắt lỗi chung của gather
                logger.error(f"Lỗi nghiêm trọng khi gọi Gemini song song: {gather_err}", exc_info=True)
                internal_suggestions = internal_suggestions or ["Lỗi hệ thống gọi AI."]
                auto_response = auto_response or "Lỗi hệ thống gọi AI."
        else:
            logger.warning("Muốn gọi Gemini nhưng chưa cấu hình API Key.")
            ai_call_reason += " (Nhưng Gemini chưa cấu hình)"
            internal_suggestions = ["Gemini chưa được cấu hình."]
            auto_response = "Gemini chưa được cấu hình."
    else:
         logger.info("Không gọi Gemini.")
         # Có thể thêm template phản hồi đơn giản ở đây nếu muốn
         # if sentiment_label == config.TARGET_LABEL_MAP[2]: auto_response = "Cảm ơn phản hồi của bạn!"

    end_req_time = time.time()
    processing_time_ms = (end_req_time - start_req_time) * 1000
    logger.info(f"Xử lý HYBRID hoàn tất trong {processing_time_ms:.2f} ms.")

    # --- Bước 4: Trả về Kết quả ---
    return SentimentResponse(
        sentiment=sentiment_label,
        confidence=confidence,
        suggestions=internal_suggestions,
        generated_response=auto_response,
        processing_time_ms=processing_time_ms,
        ai_call_reason=ai_call_reason # Trả về lý do gọi AI
    )

# --- Chạy API Server ---
if __name__ == "__main__":
    logger.info("--- Khởi chạy FastAPI Server (Hybrid) từ Command Line ---")
    if not os.path.isdir(config.MODEL_SAVE_PATH):
        logger.warning(f"!!! Không tìm thấy thư mục model '{config.MODEL_SAVE_PATH}'. !!!")
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=True, log_level="info")