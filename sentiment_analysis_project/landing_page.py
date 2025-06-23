# landing_page.py

import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Giải pháp Phân tích Phản hồi Khách hàng AI",
    page_icon="🌟",
    layout="wide"
)

st.markdown("""
<style>
/* Font hiện đại từ Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* Reset và style tổng quát */
* {
    font-family: 'Poppins', sans-serif;
    box-sizing: border-box;
}

/* Nền gradient cho toàn trang */
body {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: #ffffff;
}

/* Tiêu đề chính với hiệu ứng fade-in */
.stTitle {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00ddeb, #ff6f61);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeIn 2s ease-in-out;
}

/* Subheader với hiệu ứng nghiêng nhẹ */
.stSubheader {
    font-size: 1.5rem;
    font-weight: 400;
    color: #e0e0e0;
    font-style: italic;
}

/* Card cho tính năng nổi bật */
.feature-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
}

/* Nút CTA với hiệu ứng gradient */
.cta-button {
    background: linear-gradient(45deg, #ff6f61, #00ddeb);
    color: white !important;
    padding: 12px 25px;
    border-radius: 25px;
    text-decoration: none !important;
    display: inline-block;
    transition: transform 0.2s ease, background 0.3s ease;
}
.cta-button:hover {
    transform: scale(1.05);
    background: linear-gradient(45deg, #00ddeb, #ff6f61);
}

/* Hiệu ứng fade-in */
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Định dạng đoạn văn Markdown */
.stMarkdown p {
    color: #d1d1d1;
    font-size: 1.1rem;
    line-height: 1.6;
}

/* Code block với nền tối */
.stCodeBlock {
    background: #2d2d2d !important;
    border-radius: 10px;
    padding: 15px;
}

/* Footer */
.footer {
    text-align: center;
    color: #a0a0a0;
    font-size: 0.9rem;
    margin-top: 50px;
}

/* Highlight cho Product ID */
.product-highlight {
    color: #00ddeb;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
current_time = datetime.now().strftime("%I:%M %p +07, %d/%m/%Y")
st.title("🌟 Giải pháp Phân tích & Xử lý Phản hồi Khách hàng Thông minh")
st.subheader("Tự động hóa hiểu biết khách hàng và tối ưu hóa tương tác bằng AI")
st.caption(f"Cập nhật lần cuối: {current_time} (10:54 AM +07, 14/05/2025)")
st.markdown("---")

# --- Giới thiệu Vấn đề & Giải pháp ---
col1, col2 = st.columns([2, 1])
with col1:
    st.header("❓ Vấn đề Doanh nghiệp Thường Gặp")
    st.markdown("""
    - Khối lượng lớn phản hồi từ khách hàng (email, chat, review, mạng xã hội...) khiến việc xử lý thủ công trở nên quá tải.  
    - Bỏ lỡ những thông tin chi tiết quan trọng về cảm xúc, nhu cầu, và các vấn đề khách hàng gặp phải theo <span class='product-highlight'>mỗi sản phẩm</span>.  
    - Phản hồi chậm trễ hoặc không nhất quán làm giảm sự hài lòng của khách hàng.  
    - Khó khăn trong việc tổng hợp và đánh giá xu hướng chung từ dữ liệu văn bản phi cấu trúc.
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 20px; text-align: center;">
        <p style="color: #e0e0e0;">📧 Nhân viên quá tải với email và phản hồi</p>
    </div>
    """, unsafe_allow_html=True)

st.header("🚀 Giải pháp của Chúng tôi")
st.markdown("""
Hệ thống của chúng tôi cung cấp một giải pháp toàn diện dựa trên Trí tuệ Nhân tạo (AI) để giúp bạn:  
1. **Tự động phân tích cảm xúc** của từng phản hồi (Tích cực, Tiêu cực, Trung tính) bằng mô hình học sâu tiên tiến (XLM-RoBERTa).  
2. **Tận dụng AI tạo sinh (Google Gemini)** để nhận gợi ý hành động nội bộ và soạn thảo nội dung phản hồi tự động, phù hợp với từng <span class='product-highlight'>sản phẩm</span>.  
3. **Xây dựng Knowledge Base (Cơ sở Tri thức)**: Tự động lưu trữ và tái sử dụng các kết quả đã xử lý, bao gồm thông tin <span class='product-highlight'>Product ID</span>, giúp tăng tốc độ và tiết kiệm chi phí theo thời gian.  
4. **Cung cấp API linh hoạt** để dễ dàng tích hợp vào các hệ thống hiện có của bạn.  
5. **Giao diện demo trực quan** để bạn trải nghiệm và kiểm thử nhanh chóng.
""", unsafe_allow_html=True)
st.markdown("---")

# --- Tính năng Nổi bật ---
st.header("✨ Tính năng Nổi bật")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 Phân tích Cảm xúc Chính xác</h3>
        <p>- Sử dụng model XLM-RoBERTa đa ngôn ngữ đã được fine-tune.</p>
        <p>- Phân loại theo <span class='product-highlight'>Product ID</span> với 3 mức: Tích cực, Trung tính, Tiêu cực với độ tin cậy.</p>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="feature-card">
        <h3>🧠 Hỗ trợ bởi AI Tạo sinh</h3>
        <p>- Tích hợp Google Gemini để đưa ra gợi ý hành động cụ thể cho từng <span class='product-highlight'>sản phẩm</span>.</p>
        <p>- Tự động tạo nội dung phản hồi phù hợp với từng trường hợp.</p>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="feature-card">
        <h3>📚 Knowledge Base Thông minh</h3>
        <p>- Lưu trữ kết quả theo <span class='product-highlight'>Product ID</span>, tránh xử lý lặp lại.</p>
        <p>- Hệ thống 'học hỏi' và tối ưu dần theo thời gian sử dụng.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Lợi ích ---
st.header("📈 Lợi ích cho Doanh nghiệp")
st.markdown("""
- **Tiết kiệm thời gian và chi phí:** Giảm thiểu công việc thủ công trong việc đọc và phân loại phản hồi theo <span class='product-highlight'>Product ID</span>.  
- **Hiểu sâu sắc khách hàng:** Nắm bắt nhanh chóng cảm xúc và các vấn đề chính mà khách hàng quan tâm theo từng sản phẩm.  
- **Cải thiện Chất lượng Dịch vụ:** Đưa ra hành động và phản hồi kịp thời, chuyên nghiệp.  
- **Tăng cường sự Hài lòng và Trung thành:** Giải quyết vấn đề hiệu quả, thể hiện sự quan tâm đến khách hàng.  
- **Quyết định Dựa trên Dữ liệu:** Có được thông tin tổng hợp để cải tiến sản phẩm/dịch vụ.
""", unsafe_allow_html=True)
st.markdown("---")

# --- Cách Sử dụng API (Cập nhật với product_id) ---
st.header("🔌 Cách Sử dụng API")
st.markdown("Hệ thống cung cấp các API endpoint đơn giản để tích hợp, hỗ trợ tham số <span class='product-highlight'>product_id</span> để phân tích theo sản phẩm:")
st.subheader("1. Phân tích Cảm xúc Nhanh (`/sentiment/`)")
st.markdown("Chỉ phân tích cảm xúc bằng model local, nhanh chóng và tiết kiệm. Kết quả được lưu vào KB theo <span class='product-highlight'>product_id</span>.")
st.code("""
POST http://your-api-host:8000/sentiment/
Body:
{
  "comment": "Nội dung bình luận của bạn",
  "product_id": "AO-001"  // Mã sản phẩm (tùy chọn)
}

Response (Ví dụ):
{
  "sentiment": "Tích cực",
  "confidence": 0.95,
  "product_id_processed": "AO-001",
  "model_used": "local_xlmr",
  "processing_time_ms": 150,
  "source": "new_sentiment_only" // Hoặc "cache" nếu đã có trong KB
}
""", language="json")

st.subheader("2. Xử lý Chi tiết với AI (`/process/`)")
st.markdown("Phân tích cảm xúc, đồng thời gọi AI (Gemini) để lấy gợi ý hành động và tạo phản hồi tự động. Kết quả đầy đủ được lưu/cập nhật vào KB theo <span class='product-highlight'>product_id</span>.")
st.code("""
POST http://your-api-host:8000/process/
Body:
{
  "comment": "Sản phẩm này thật tuyệt vời!",
  "product_id": "AO-001"  // Mã sản phẩm (tùy chọn)
}

Response (Ví dụ):
{
  "sentiment": "Tích cực",
  "confidence": 0.98,
  "product_id_processed": "AO-001",
  "ai_call_reason": "Xử lý mới (Luôn gọi AI)",
  "suggestions": [
    "Gửi lời cảm ơn chân thành đến khách hàng.",
    "Khuyến khích khách hàng chia sẻ trải nghiệm này với bạn bè.",
    "Ghi nhận đây là một điểm mạnh của sản phẩm/dịch vụ."
  ],
  "generated_response": "Cảm ơn bạn rất nhiều vì đã yêu thích sản phẩm của chúng tôi! Chúng tôi rất vui khi bạn hài lòng và hy vọng sẽ tiếp tục mang đến những trải nghiệm tuyệt vời cho bạn.",
  "processing_time_ms": 3500,
  "source": "new_full_process" // Hoặc "cache", "cache_enriched"
}
""", language="json")
st.markdown("""
<div style="text-align: center;">
    <a href="http://localhost:8000/docs" class="cta-button">👉 Xem Tài liệu API Chi tiết (Swagger UI)</a>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# --- Demo và Liên hệ ---
st.header("🚀 Trải nghiệm Demo & Liên hệ")
st.markdown("""
Bạn có thể trải nghiệm trực tiếp các tính năng của hệ thống, bao gồm phân tích theo <span class='product-highlight'>Product ID</span>, qua ứng dụng demo của chúng tôi.
""", unsafe_allow_html=True)

link_to_app = "http://localhost:8502" 
st.markdown(f"""
<div style="text-align: center;">
    <a href="{link_to_app}" target="_blank" class="cta-button">🔗 Chạy Ứng dụng Demo Chính</a>
</div>
""", unsafe_allow_html=True)
st.markdown("Để biết thêm thông tin chi tiết hoặc yêu cầu tích hợp, vui lòng liên hệ: thinhnguyen190304@gmail.com")

# --- Footer ---
st.markdown("""
<div class="footer">
    Dự án Thực tập - Nguyễn Trần Hoàng Thịnh - Trường Đại học Gia Định | Được hướng dẫn bởi Ths. Đặng Quốc Phong
</div>
""", unsafe_allow_html=True)