# app.py (vFinal v2 - Tab 2 Nhận xét rõ ràng & Thống kê AI chỉn chu)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import time
import requests
import json
import traceback
from collections import Counter

import config
try: from visualization import plot_confusion_matrix, plot_training_history; VIZ_AVAILABLE = True
except ImportError: VIZ_AVAILABLE = False; print("Cảnh báo: Module 'visualization' không tìm thấy.")

# Import thư viện Gemini và kiểm tra cấu hình
gemini_configured_app = False
if config.GEMINI_API_KEY:
    try: import google.generativeai as genai; genai.configure(api_key=config.GEMINI_API_KEY); gemini_configured_app = True; print("Gemini OK (Streamlit).")
    except ImportError: print("Cảnh báo: google-generativeai chưa cài.")
    except Exception as e: print(f"Cảnh báo: Lỗi cấu hình Gemini (Streamlit): {e}")
else: print("Cảnh báo: GEMINI_API_KEY chưa đặt (Streamlit).")


# --- Cấu hình Trang ---
st.set_page_config(page_title="Xử lý Phản hồi vFinal", page_icon="💡", layout="wide")

# --- Địa chỉ API Backend ---
API_HOST = getattr(config, 'API_HOST', '127.0.0.1')
API_PORT = getattr(config, 'API_PORT', 8000)
BACKEND_API_URL_SENTIMENT = f"http://{API_HOST}:{API_PORT}/sentiment/"
BACKEND_API_URL_PROCESS = f"http://{API_HOST}:{API_PORT}/process/"

# --- Giao diện Chính ---
st.title("💡 Hệ thống Phân tích & Xử lý Phản hồi Khách hàng")
st.markdown("""
**Chọn cách xử lý:**
- **Phân tích Nhanh:** Chỉ lấy cảm xúc (nhanh, đọc/lưu vào KB).
- **Xử lý Chi tiết:** Lấy cảm xúc, gợi ý & phản hồi AI (đọc/làm giàu KB & gọi Gemini).
- **Xử lý Hàng loạt:** Phân tích nhanh toàn bộ file CSV (làm nóng KB), nhận **nhận xét/xu hướng** và **gợi ý chung** từ AI.
""")

# --- Các Tab chức năng ---
tab1, tab2, tab3 = st.tabs(["📝 Xử lý Đơn lẻ", "📂 Xử lý Hàng loạt (Nhanh + AI Tổng hợp)", "📈 Thông tin Model"])

# --- Tab 1: Xử lý Đơn lẻ (Giữ nguyên) ---
with tab1:
    # ... (Code Tab 1 giữ nguyên) ...
    st.header("Nhập phản hồi cần xử lý:")
    user_input_single = st.text_area("Nhập văn bản...", height=150, key="single_input_tab1_final", placeholder="...")
    col_btn1, col_btn2 = st.columns(2)
    def display_results(api_response, start_time, end_time, endpoint_name):
        # ... (Hàm display_results giữ nguyên) ...
        st.markdown("---"); st.subheader(f"Kết quả từ {endpoint_name}:")
        if not api_response: st.error(f"Không nhận được phản hồi hợp lệ từ API {endpoint_name}."); return
        total_time = (end_time - start_time) * 1000; api_time = api_response.get('processing_time_ms'); source = api_response.get('source', 'N/A'); ai_reason = api_response.get('ai_call_reason')
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("**Phân tích Cảm xúc:**"); sentiment = api_response.get('sentiment', 'N/A'); confidence = api_response.get('confidence')
            try: label_map = getattr(config, 'TARGET_LABEL_MAP', {}); positive_label = label_map.get(2, "Tích cực"); negative_label = label_map.get(0, "Tiêu cực");
            except: label_map={}; positive_label="Tích cực"; negative_label="Tiêu cực"
            if sentiment == positive_label: st.success(f"**Cảm xúc:** {sentiment}")
            elif sentiment == negative_label: st.error(f"**Cảm xúc:** {sentiment}")
            else: st.warning(f"**Cảm xúc:** {sentiment}")
            if confidence is not None: st.metric(label="Độ tin cậy", value=f"{confidence:.2%}")
            st.caption(f"T.gian: {total_time:.0f}ms | API T.gian: {api_time:.0f}ms" if api_time else f"T.gian: {total_time:.0f}ms")
            source_text = {'cache':'Cache KB', 'cache_enriched':'Làm giàu KB', 'new_sentiment_only':'Mới (Chỉ Sentiment)', 'new_full_process':'Mới (Full AI)', 'error':'Lỗi Xử lý'}.get(source, source)
            st.caption(f"Nguồn: {source_text}")
            if ai_reason and source != 'cache': st.caption(f"Trạng thái AI: {ai_reason}")
        with col_res2:
            st.markdown("**Gợi ý Phản hồi Tự động (AI/Cache):**"); generated_response = api_response.get('generated_response')
            is_valid_response = generated_response and isinstance(generated_response, str) and "Lỗi" not in generated_response and "chưa cấu hình" not in generated_response and "không tạo ra" not in generated_response
            if is_valid_response: st.text_area("Nội dung:", value=generated_response, height=120, key=f"gen_resp_{source}_{int(time.time())}", disabled=False)
            elif generated_response: st.info(generated_response)
            else: st.info("Không có.")
        st.markdown("---"); st.markdown("**Gợi ý Hành động Nội bộ (AI/Cache):**"); suggestions = api_response.get('suggestions')
        is_valid_suggestions = suggestions and isinstance(suggestions, list) and not any("Lỗi" in s or "chưa cấu hình" in s for s in suggestions)
        if is_valid_suggestions: st.markdown("\n".join(f"- {s}" for s in suggestions))
        elif suggestions and isinstance(suggestions, list): st.info(suggestions[0])
        else: st.info("Không có.")

    with col_btn1: # Nút Phân tích nhanh
        if st.button("⚡ Phân tích Nhanh (Đọc/Lưu KB)", key="analyze_fast_kb_final", help="Lấy cảm xúc (local), đọc/lưu KB."):
            if user_input_single and user_input_single.strip():
                start_time = time.time(); api_response = None; error_message = None
                with st.spinner('⚡ Đang phân tích nhanh & kiểm tra KB...'):
                    try: response = requests.post(BACKEND_API_URL_SENTIMENT, json={"comment": user_input_single}, timeout=30); response.raise_for_status(); api_response = response.json()
                    except Exception as e: error_message = f"Lỗi gọi API /sentiment/: {e}"; traceback.print_exc()
                end_time = time.time()
                if error_message: st.error(error_message)
                display_results(api_response, start_time, end_time, "/sentiment/")
            else: st.warning("⚠️ Vui lòng nhập văn bản.")

    with col_btn2: # Nút Xử lý chi tiết
        if st.button("✨ Xử lý Chi tiết (KB + AI)", key="analyze_detailed_kb_final", help="Đọc KB, nếu thiếu AI -> Model + Gemini -> Lưu/Cập nhật KB."):
            if user_input_single and user_input_single.strip():
                start_time = time.time(); api_response = None; error_message = None
                with st.spinner('✨ Đang xử lý chi tiết...'):
                    try: response = requests.post(BACKEND_API_URL_PROCESS, json={"comment": user_input_single}, timeout=180); response.raise_for_status(); api_response = response.json()
                    except Exception as e: error_message = f"Lỗi gọi API /process/: {e}"; traceback.print_exc()
                end_time = time.time()
                if error_message: st.error(error_message); st.info("Mẹo: Kiểm tra server API & GEMINI_API_KEY.")
                display_results(api_response, start_time, end_time, "/process/")
            else: st.warning("⚠️ Vui lòng nhập văn bản.")


# --- Tab 2: Xử lý Hàng loạt (CSV - Cập nhật Thống kê & Nhận xét) ---
with tab2:
    st.header("Tải lên file CSV để Phân tích Nhanh & Nhận Gợi ý Chung")
    st.markdown("Chức năng này sẽ phân tích cảm xúc từng dòng bằng model local (lưu vào KB), sau đó **gọi AI một lần** để đưa ra nhận xét, xu hướng và gợi ý hành động chung dựa trên kết quả tổng hợp.")
    text_col_name = getattr(config, 'TEXT_COLUMN', 'comment')
    uploaded_file_batch = st.file_uploader(f"Chọn file CSV (cột '{text_col_name}')", type=["csv"], key="csv_local_kb_final_v3")
    limit_rows_batch = st.number_input(
        "Giới hạn số dòng xử lý (Nhập 0 để xử lý tất cả):", min_value=0, value=0,  # Mặc định 0
        step=50, key="limit_rows_batch_final_v3", help="Để 0 nếu muốn xử lý toàn bộ file."
    )

    if uploaded_file_batch is not None:
        try:
            df_batch = None
            with st.spinner("Đang đọc CSV..."):
                try:
                    df_batch = pd.read_csv(uploaded_file_batch, encoding='utf-8-sig', low_memory=False)
                except:
                    df_batch = pd.read_csv(uploaded_file_batch, encoding='utf-8', low_memory=False)
            st.success(f"✅ Đã tải file '{uploaded_file_batch.name}' ({len(df_batch)} dòng).")
            if text_col_name not in df_batch.columns:
                st.error(f"Lỗi: Không tìm thấy cột '{text_col_name}'.")
                st.stop()

            if st.button("📊 Phân tích & Nhận Gợi ý Chung (Hàng loạt)", key="analyze_csv_local_kb_final_v3"):
                if limit_rows_batch > 0 and limit_rows_batch < len(df_batch):
                    process_df_batch = df_batch.head(limit_rows_batch)
                    limit_info_batch = f"{limit_rows_batch} dòng đầu"
                else:
                    process_df_batch = df_batch
                    limit_info_batch = "tất cả các dòng"
                total_to_process_batch = len(process_df_batch)
                if total_to_process_batch == 0:
                    st.warning("Không có dòng nào để xử lý.")
                    st.stop()

                st.info(f"Bắt đầu phân tích nhanh & lưu KB cho {limit_info_batch}...")
                results_list_batch = []
                error_count_batch = 0
                cache_hit_count = 0
                potential_ai_call_count = 0
                start_batch_run_time = time.time()
                progress_text_batch = f"Đang xử lý 0/{total_to_process_batch} dòng..."
                progress_bar_batch = st.progress(0, text=progress_text_batch)

                # Lấy cấu hình kiểm tra AI
                conf_threshold_batch = float(getattr(config, 'CONFIDENCE_THRESHOLD', 0.80))
                check_negative_batch = bool(getattr(config, 'ALWAYS_CHECK_NEGATIVE', True))
                label_map_batch = getattr(config, 'TARGET_LABEL_MAP', {})
                negative_label_value_batch = label_map_batch.get(0, "Tiêu cực")

                # --- Vòng lặp Xử lý Từng Dòng ---
                for index, row in process_df_batch.iterrows():
                    comment_text_batch = str(row[text_col_name]) if pd.notna(row[text_col_name]) else ""
                    result_row_batch = {"original_comment": comment_text_batch, "sentiment": None, "confidence": None, "source": None, "status": None, "would_call_ai": False}
                    if comment_text_batch:
                        try:
                            response = requests.post(BACKEND_API_URL_SENTIMENT, json={"comment": comment_text_batch}, timeout=60)
                            response.raise_for_status()
                            api_data = response.json()
                            sentiment_res = api_data.get('sentiment')
                            conf_res = api_data.get('confidence')
                            source_res = api_data.get('source')
                            result_row_batch.update({'sentiment': sentiment_res, 'confidence': conf_res, 'source': source_res, 'status': 'Thành công'})
                            if source_res == 'cache':
                                cache_hit_count += 1
                            # Ước tính gọi AI (ngay cả khi đọc từ cache nhưng thiếu AI)
                            would_call = False
                            if source_res != 'cache':
                                if conf_res is not None and conf_res < conf_threshold_batch:
                                    would_call = True
                                elif check_negative_batch and sentiment_res == negative_label_value_batch:
                                    would_call = True
                            elif source_res == 'cache' and (api_data.get('suggestions') is None or api_data.get('generated_response') is None):
                                would_call = True  # Cần làm giàu KB
                            result_row_batch['would_call_ai'] = would_call
                            if would_call:
                                potential_ai_call_count += 1
                        except requests.exceptions.Timeout:
                            result_row_batch['status'] = 'Lỗi API: Timeout'
                            error_count_batch += 1
                        except requests.exceptions.RequestException as e:
                            result_row_batch['status'] = f'Lỗi API: {type(e).__name__}'
                            error_count_batch += 1
                        except Exception as e:
                            result_row_batch['status'] = f'Lỗi khác: {type(e).__name__}'
                            error_count_batch += 1
                    else:
                        result_row_batch['status'] = 'Bỏ qua (rỗng)'
                    results_list_batch.append(result_row_batch)
                    progress_percentage = (index + 1) / total_to_process_batch
                    progress_text_batch = f"Đang xử lý {index + 1}/{total_to_process_batch} dòng..."
                    progress_bar_batch.progress(progress_percentage, text=progress_text_batch)
                progress_bar_batch.empty()
                end_batch_run_time = time.time()
                st.success(f"✅ Phân tích {total_to_process_batch} dòng hoàn tất sau {end_batch_run_time - start_batch_run_time:.2f} giây.")

                # --- Hiển thị Kết quả Tổng hợp ---
                if results_list_batch:
                    results_df_batch = pd.DataFrame(results_list_batch)
                    st.markdown("---")
                    st.subheader("📊 Thống kê Chung")

                    # *** CẬP NHẬT CỘT THỐNG KÊ ***
                    col_b_stat1, col_b_stat2, col_b_stat3, col_b_stat4 = st.columns(4)
                    with col_b_stat1:
                        st.metric("Tổng dòng xử lý", total_to_process_batch)
                    with col_b_stat2:
                        st.metric("Số dòng gặp lỗi", error_count_batch)
                    with col_b_stat3:
                        st.metric("Số lần dùng Cache KB", cache_hit_count)
                    with col_b_stat4:
                        st.metric("Ước tính cần Gọi AI*", potential_ai_call_count, help="Số dòng có độ tin cậy thấp/tiêu cực hoặc cần làm giàu KB, sẽ gọi Gemini nếu dùng 'Xử lý Chi tiết'.")

                    # Biểu đồ + Nhận xét
                    if 'sentiment' in results_df_batch.columns and not results_df_batch['sentiment'].empty:
                        valid_sentiments_b = results_df_batch.dropna(subset=['sentiment'])
                        sentiment_counts_b = valid_sentiments_b['sentiment'].value_counts()
                        all_labels_b = list(getattr(config, 'TARGET_LABEL_MAP', {}).values())
                        sentiment_counts_b = sentiment_counts_b.reindex(all_labels_b, fill_value=0)
                        color_map_stats_b = {"Tiêu cực": '#DC143C', "Trung tính": '#FFD700', "Tích cực": '#32CD32'}
                        counts_to_plot_b = sentiment_counts_b[sentiment_counts_b.index.isin(color_map_stats_b.keys())]
                        if not counts_to_plot_b.empty:
                            st.markdown("---")
                            st.subheader("📈 Phân phối & Nhận xét Tổng quan")
                            col_chart, col_commentary = st.columns([2, 1])
                            with col_chart:
                                fig_bar_b = px.bar(counts_to_plot_b, x=counts_to_plot_b.index, y=counts_to_plot_b.values, labels={'x': 'Cảm xúc', 'y': 'Số lượng'}, color=counts_to_plot_b.index, color_discrete_map=color_map_stats_b, text=counts_to_plot_b.values, height=350)
                                fig_bar_b.update_layout(showlegend=False, title_text="Biểu đồ Cảm xúc", title_x=0.5)
                                st.plotly_chart(fig_bar_b, use_container_width=True)
                            with col_commentary:
                                st.subheader("📝 Nhận xét & Gợi ý")
                                total_valid = counts_to_plot_b.sum()
                                if total_valid > 0:
                                    pos_count = counts_to_plot_b.get("Tích cực", 0)
                                    neg_count = counts_to_plot_b.get("Tiêu cực", 0)
                                    neu_count = counts_to_plot_b.get("Trung tính", 0)
                                    positive_perc = (pos_count / total_valid) * 100
                                    negative_perc = (neg_count / total_valid) * 100
                                    neutral_perc = (neu_count / total_valid) * 100

                                    # *** NHẬN XÉT RÕ RÀNG HƠN ***
                                    st.markdown("**Phân phối Cảm xúc:**")
                                    st.markdown(f"- **Tích cực:** {pos_count} ({positive_perc:.1f}%)")
                                    st.markdown(f"- **Trung tính:** {neu_count} ({neutral_perc:.1f}%)")
                                    st.markdown(f"- **Tiêu cực:** {neg_count} ({negative_perc:.1f}%)")
                                    st.markdown("---")
                                    st.markdown("**Xu hướng chính:**")
                                    if positive_perc >= 65:
                                        st.success("-> Phần lớn phản hồi là Tích cực.")
                                    elif negative_perc >= 35:
                                        st.error("-> Tỷ lệ Tiêu cực rất cao, cần hành động ngay.")
                                    elif negative_perc >= 20:
                                        st.warning("-> Tỷ lệ Tiêu cực đáng chú ý, cần xem xét.")
                                    else:
                                        st.info("-> Tỷ lệ cảm xúc tương đối cân bằng.")
                                    st.markdown("---")

                                    # Gọi Gemini 1 lần lấy gợi ý chung
                                    st.markdown("**Gợi ý Hành động Tổng thể (AI):**")
                                    if gemini_configured_app:
                                        with st.spinner("Đang lấy gợi ý từ AI..."):
                                            prompt_summary = f"""Phân tích tỷ lệ cảm xúc từ phản hồi khách hàng: Tích cực {positive_perc:.1f}%, Trung tính {neutral_perc:.1f}%, Tiêu cực {negative_perc:.1f}%. Đề xuất 3-5 hành động chiến lược tổng thể. Định dạng: danh sách gạch đầu dòng."""
                                            try:
                                                model_gen = genai.GenerativeModel('gemini-1.5-flash')
                                                response_gen = model_gen.generate_content(prompt_summary)
                                                summary_suggestions = response_gen.text.strip().split('\n')
                                            except Exception as gemini_e:
                                                st.warning(f"Lỗi gọi AI: {gemini_e}")
                                                summary_suggestions = []
                                            if summary_suggestions:
                                                for sugg in summary_suggestions:
                                                    if sugg.strip():
                                                        st.markdown(f"- {sugg.strip().lstrip('-* ')}")
                                            else:
                                                st.info("AI không đưa ra gợi ý.")
                                    else:
                                        st.info("Gemini chưa cấu hình. Sử dụng gợi ý mẫu:")
                                        # Gợi ý template thay thế
                                        if positive_perc >= 65:
                                            st.markdown("- Tiếp tục phát huy điểm mạnh.\n- Lan tỏa phản hồi tốt.")
                                        elif negative_perc >= 35:
                                            st.markdown("- Ưu tiên phân tích nguyên nhân gốc rễ của các phản hồi tiêu cực.\n- Lên kế hoạch hành động khắc phục ngay.")
                                        else:
                                            st.markdown("- Theo dõi sát sao phản hồi.\n- Tìm cơ hội cải thiện từ nhóm trung tính/tiêu cực.")

                                    st.caption(f"(Dựa trên {total_valid} phản hồi hợp lệ)")
                                else:
                                    st.info("Không đủ dữ liệu nhận xét.")
                        else:
                            st.info("Không có dữ liệu cảm xúc hợp lệ.")
                    else:
                        st.warning("Không có dữ liệu cảm xúc.")

                    # --- Nút Tải xuống ---
                    st.markdown("---")
                    st.subheader("💾 Tải xuống Kết quả Phân tích Nhanh")
                    @st.cache_data
                    def convert_sentiment_batch_df(df):
                        cols = ["original_comment", "sentiment", "confidence", "source", "status", "would_call_ai"]  # Thêm cột ước tính AI
                        existing = [c for c in cols if c in df.columns]
                        try:
                            return df[existing].to_csv(index=False, encoding='utf-8-sig')
                        except:
                            return None
                    csv_sentiment_output = convert_sentiment_batch_df(results_df_batch)
                    if csv_sentiment_output:
                        st.download_button(label="📥 Tải Kết quả (CSV)", data=csv_sentiment_output, file_name=f'ket_qua_sentiment_{uploaded_file_batch.name}.csv', mime='text/csv')
                    else:
                        st.error("Lỗi tạo file CSV.")
                else:
                    st.warning("Không có dòng nào được xử lý.")
        except Exception as e:
            st.error(f"⚠️ Lỗi khi xử lý file CSV: {e}")
            traceback.print_exc()

# --- Tab 3: Thông tin Model (Giữ nguyên) ---
with tab3:
    st.header("Thông tin Đánh giá Model (XLM-RoBERTa)")
    # ... (Code hiển thị metrics, report, cm, curves, error analysis giữ nguyên) ...

# --- Footer ---
st.markdown("---")
st.caption("Dự án Thực tập - Xử lý Phản hồi Khách hàng - [Tên của bạn]")