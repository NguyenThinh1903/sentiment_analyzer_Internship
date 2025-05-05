# app.py (Gọi Endpoint /process/ mới của Backend API)

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
# Import visualization chỉ dùng cho Tab 3
try: from visualization import plot_confusion_matrix, plot_training_history; VISUALIZATION_AVAILABLE = True
except ImportError: VISUALIZATION_AVAILABLE = False; print("Cảnh báo: Không tìm thấy module 'visualization'.")

# --- Cấu hình Trang ---
st.set_page_config(page_title="Xử lý Phản hồi (Lai ghép)", page_icon="🚀", layout="wide")

# --- Địa chỉ API Backend ---
# *** SỬA ĐƯỜNG DẪN ENDPOINT ***
BACKEND_API_URL_PROCESS = f"http://{getattr(config, 'API_HOST', '127.0.0.1')}:{getattr(config, 'API_PORT', 8000)}/process/" # Trỏ đến endpoint mới
# Có thể thêm URL cho endpoint /sentiment/ nếu muốn dùng ở đâu đó
# BACKEND_API_URL_SENTIMENT = f"http://{getattr(config, 'API_HOST', '127.0.0.1')}:{getattr(config, 'API_PORT', 8000)}/sentiment/"


# --- Giao diện Chính ---
st.title("🚀 Hệ thống Xử lý Phản hồi Khách hàng (Lai ghép AI)")
st.markdown("Nhập phản hồi hoặc tải file CSV. Hệ thống sẽ dùng model local và gọi AI (Gemini) khi cần thiết.")

# --- Các Tab chức năng ---
tab1, tab2, tab3 = st.tabs(["📝 Xử lý Đơn lẻ", "📂 Xử lý Hàng loạt (CSV)", "📈 Thông tin Model XLM-R"])

# --- Tab 1: Xử lý Đơn lẻ ---
with tab1:
    st.header("Nhập phản hồi cần xử lý:")
    user_input = st.text_area("Nhập văn bản...", height=150, key="single_hybrid", placeholder="Ví dụ: Sản phẩm tốt nhưng giao hàng hơi chậm.")

    if st.button("🚀 Xử lý Ngay!", key="analyze_single_hybrid"):
        if user_input and user_input.strip():
            start_time = time.time()
            with st.spinner('🧠 Đang xử lý (Model local + AI nếu cần)...'):
                api_response = None; error_message = None
                try:
                    # *** GỌI ENDPOINT MỚI ***
                    response = requests.post(BACKEND_API_URL_PROCESS, json={"comment": user_input}, timeout=120)
                    response.raise_for_status(); api_response = response.json()
                except requests.exceptions.RequestException as e: error_message = f"Lỗi kết nối API Backend ({BACKEND_API_URL_PROCESS}): {e}"
                except json.JSONDecodeError: error_message = f"Lỗi đọc JSON từ API. Status: {response.status_code}. Response: {response.text[:500]}"
                except Exception as e: error_message = f"Lỗi không xác định: {e}"; traceback.print_exc()

            end_time = time.time()

            if error_message:
                st.error(error_message)
                st.info("Mẹo: Đảm bảo server API Backend (uvicorn api:app --reload) đang chạy.")
            elif api_response:
                # --- Hiển thị kết quả từ /process/ ---
                st.subheader("Kết quả Xử lý:")
                total_time = (end_time - start_time) * 1000
                api_time = api_response.get('processing_time_ms')
                ai_reason = api_response.get('ai_call_reason', 'N/A')

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown("**Phân tích Cảm xúc (Model Local):**")
                    sentiment = api_response.get('sentiment', 'N/A')
                    confidence = api_response.get('confidence')
                    try: # Tô màu
                        label_map = getattr(config, 'TARGET_LABEL_MAP', {})
                        positive_label = label_map.get(2, "Tích cực")
                        negative_label = label_map.get(0, "Tiêu cực")
                        if sentiment == positive_label: st.success(f"**Cảm xúc:** {sentiment}")
                        elif sentiment == negative_label: st.error(f"**Cảm xúc:** {sentiment}")
                        else: st.warning(f"**Cảm xúc:** {sentiment}")
                    except Exception: st.write(f"**Cảm xúc:** {sentiment}")
                    if confidence is not None: st.metric(label="Độ tin cậy", value=f"{confidence:.2%}")
                    st.caption(f"Tổng T.gian: {total_time:.0f}ms | API T.gian: {api_time:.0f}ms" if api_time else f"Tổng T.gian: {total_time:.0f}ms")
                    st.caption(f"Lý do gọi AI: {ai_reason}")

                with col_res2:
                    st.markdown("**Gợi ý Phản hồi Tự động (AI):**")
                    generated_response = api_response.get('generated_response')
                    # Hiển thị nếu có và không phải lỗi/thông báo
                    if generated_response and isinstance(generated_response, str) and "Lỗi" not in generated_response and "chưa cấu hình" not in generated_response:
                        st.text_area("Nội dung:", value=generated_response, height=150, key="gen_resp_area_h", disabled=False)
                    elif generated_response: # Hiển thị thông báo lỗi/chưa cấu hình từ API
                         st.info(generated_response)
                    else: # Trường hợp không gọi AI và không có template
                        st.info("Không có gợi ý phản hồi tự động (do không cần gọi AI hoặc AI không tạo ra).")

                st.markdown("---")
                st.markdown("**Gợi ý Hành động Nội bộ (AI):**")
                suggestions = api_response.get('suggestions')
                # Hiển thị nếu có và không phải lỗi/thông báo
                if suggestions and isinstance(suggestions, list) and not any("Lỗi" in s or "chưa cấu hình" in s for s in suggestions):
                    for i, suggestion in enumerate(suggestions): st.markdown(f"{i+1}. {suggestion}")
                elif suggestions and isinstance(suggestions, list): # Hiển thị thông báo lỗi/chưa cấu hình
                    st.info(suggestions[0])
                else: # Trường hợp không gọi AI
                    st.info("Không có gợi ý hành động nội bộ (do không cần gọi AI hoặc AI không tạo ra).")
            else:
                 st.error("Không nhận được phản hồi hợp lệ từ API.")
        else:
            st.warning("⚠️ Vui lòng nhập văn bản.")


# --- Tab 2: Xử lý Hàng loạt (CSV) ---
with tab2:
    st.header("Tải lên file CSV để xử lý hàng loạt (Lai ghép):")
    text_col_name = getattr(config, 'TEXT_COLUMN', 'comment')
    uploaded_file = st.file_uploader(f"Chọn file CSV (cột '{text_col_name}')", type=["csv"], key="csv_hybrid", help=f"Cột '{text_col_name}' sẽ được xử lý.")

    col_limit1, col_limit2 = st.columns([1, 3])
    with col_limit1:
        limit_enabled = st.checkbox("Giới hạn số dòng?", key="limit_checkbox", value=False)
    with col_limit2:
        limit_rows_hybrid = st.number_input("Số dòng tối đa xử lý:", min_value=1, value=50, step=10, key="limit_rows_input_conditional", disabled=not limit_enabled, help="Tick ô bên cạnh.")

    if uploaded_file is not None:
        try:
            with st.spinner("Đang đọc CSV..."):
                try: df = pd.read_csv(uploaded_file, encoding='utf-8-sig', low_memory=False)
                except: df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
            st.success(f"✅ Đã tải file '{uploaded_file.name}' ({len(df)} dòng).")
            if text_col_name not in df.columns: st.error(f"Lỗi: Không tìm thấy cột '{text_col_name}'."); st.stop()

            if st.button("📊 Xử lý File CSV (Lai ghép)", key="analyze_csv_hybrid"):
                if limit_enabled: process_df = df.head(limit_rows_hybrid); limit_info = f"{limit_rows_hybrid} dòng đầu"
                else: process_df = df; limit_info = "tất cả các dòng"
                total_to_process = len(process_df)
                if total_to_process == 0: st.warning("Không có dòng nào để xử lý.")
                else:
                    st.info(f"Bắt đầu xử lý {limit_info}...")
                    results_list = [] ; error_count = 0 ; ai_call_count = 0
                    start_batch_time = time.time()
                    progress_text = f"Đang xử lý 0/{total_to_process} dòng..."
                    progress_bar = st.progress(0, text=progress_text)

                    # --- Vòng lặp Xử lý (Gọi Endpoint /process/)---
                    for index, row in process_df.iterrows():
                        comment_text = str(row[text_col_name]) if pd.notna(row[text_col_name]) else ""
                        # Chỉ lưu những gì cần cho thống kê/tải về
                        result_row = {"original_comment": comment_text, "sentiment": None, "confidence": None, "ai_call_reason": None, "status": None}
                        if comment_text:
                            try:
                                # *** GỌI ENDPOINT MỚI ***
                                response = requests.post(BACKEND_API_URL_PROCESS, json={"comment": comment_text}, timeout=180)
                                response.raise_for_status()
                                api_data = response.json()
                                result_row['sentiment'] = api_data.get('sentiment')
                                result_row['confidence'] = api_data.get('confidence') # Lấy confidence để tải về
                                ai_reason = api_data.get('ai_call_reason', '')
                                result_row['ai_call_reason'] = ai_reason
                                if ai_reason and "Độ tin cậy cao" not in ai_reason and "Không thuộc TH đặc biệt" not in ai_reason:
                                    ai_call_count += 1
                                result_row['status'] = 'Thành công'
                            except requests.exceptions.Timeout: result_row['status'] = 'Lỗi API: Timeout'; error_count += 1
                            except requests.exceptions.RequestException as e: result_row['status'] = f'Lỗi API: {type(e).__name__}'; error_count += 1
                            except Exception as e: result_row['status'] = f'Lỗi khác: {type(e).__name__}'; error_count += 1
                        else: result_row['status'] = 'Bỏ qua (rỗng)'
                        results_list.append(result_row)
                        progress_percentage = (index + 1) / total_to_process
                        progress_text = f"Đang xử lý {index + 1}/{total_to_process} dòng..."
                        progress_bar.progress(progress_percentage, text=progress_text)

                    end_batch_time = time.time()
                    progress_bar.empty()
                    st.success(f"✅ Xử lý {total_to_process} dòng hoàn tất sau {end_batch_time - start_batch_time:.2f} giây.")

                    # --- Hiển thị Kết quả Tổng hợp Tối giản (Giữ nguyên) ---
                    if results_list:
                        results_df = pd.DataFrame(results_list)
                        st.markdown("---"); st.subheader("📊 Thống kê Chung")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1: st.metric("Tổng dòng xử lý", total_to_process)
                        with col_stat2: st.metric("Số dòng lỗi", error_count)
                        with col_stat3: st.metric("Số dòng gọi AI", ai_call_count)
                        # ... (Code vẽ biểu đồ + Nhận xét giữ nguyên) ...
                        if 'sentiment' in results_df.columns and not results_df['sentiment'].empty:
                             valid_sentiments = results_df.dropna(subset=['sentiment']); sentiment_counts = valid_sentiments['sentiment'].value_counts()
                             all_labels = list(getattr(config, 'TARGET_LABEL_MAP', {}).values()); sentiment_counts = sentiment_counts.reindex(all_labels, fill_value=0)
                             color_map_stats = {"Tiêu cực": '#DC143C', "Trung tính": '#FFD700', "Tích cực": '#32CD32'}
                             counts_to_plot = sentiment_counts[sentiment_counts.index.isin(color_map_stats.keys())]
                             if not counts_to_plot.empty:
                                  st.markdown("---"); st.subheader("📈 Phân phối & Nhận xét Cảm xúc")
                                  col_chart, col_commentary = st.columns([2, 1])
                                  with col_chart:
                                      fig_bar_batch = px.bar(counts_to_plot, x=counts_to_plot.index, y=counts_to_plot.values, labels={'x': 'Cảm xúc', 'y': 'Số lượng'}, color=counts_to_plot.index, color_discrete_map=color_map_stats, text=counts_to_plot.values, height=350)
                                      fig_bar_batch.update_layout(showlegend=False, title_text="Biểu đồ Cảm xúc", title_x=0.5); st.plotly_chart(fig_bar_batch, use_container_width=True)
                                  with col_commentary:
                                      st.subheader("📝 Nhận xét"); total_valid = counts_to_plot.sum()
                                      if total_valid > 0:
                                          pos_count = counts_to_plot.get("Tích cực", 0); neg_count = counts_to_plot.get("Tiêu cực", 0); neu_count = counts_to_plot.get("Trung tính", 0)
                                          positive_perc = (pos_count / total_valid) * 100; negative_perc = (neg_count / total_valid) * 100; neutral_perc = (neu_count / total_valid) * 100
                                          st.markdown(f"- Tích cực: {pos_count} ({positive_perc:.1f}%)"); st.markdown(f"- Trung tính: {neu_count} ({neutral_perc:.1f}%)"); st.markdown(f"- Tiêu cực: {neg_count} ({negative_perc:.1f}%)"); st.markdown("---")
                                          if positive_perc >= 65: st.success("**Xu hướng:** Rất tích cực!"); st.markdown("**Gợi ý:** Phát huy điểm mạnh.")
                                          elif negative_perc >= 35: st.error("**Xu hướng:** Cần cải thiện!"); st.markdown("**Gợi ý:** Phân tích kỹ bình luận tiêu cực.")
                                          elif negative_perc >= 20: st.warning("**Xu hướng:** Có điểm cần chú ý."); st.markdown("**Gợi ý:** Xem xét phản hồi tiêu cực/trung tính.")
                                          else: st.info("**Xu hướng:** Cân bằng."); st.markdown("**Gợi ý:** Duy trì và theo dõi.")
                                          st.caption(f"(Trên {total_valid} phản hồi hợp lệ)")
                                      else: st.info("Không đủ dữ liệu nhận xét.")
                             else: st.info("Không có dữ liệu cảm xúc hợp lệ.")
                        else: st.warning("Không có dữ liệu cảm xúc để thống kê.")

                        # --- Nút Tải xuống ---
                        st.markdown("---"); st.subheader("💾 Tải xuống Kết quả")
                        @st.cache_data
                        def convert_minimal_batch_df(df_to_convert):
                            cols_to_save = ["original_comment", "sentiment", "confidence", "ai_call_reason", "status"]
                            existing_cols = [col for col in cols_to_save if col in df_to_convert.columns]
                            try: return df_to_convert[existing_cols].to_csv(index=False, encoding='utf-8-sig')
                            except: return None
                        csv_minimal_output = convert_minimal_batch_df(results_df)
                        if csv_minimal_output: st.download_button(label="📥 Tải Kết quả Xử lý (CSV)", data=csv_minimal_output, file_name=f'ket_qua_xu_ly_{uploaded_file.name}.csv', mime='text/csv')
                        else: st.error("Lỗi tạo file CSV.")

        except Exception as e:
            st.error(f"⚠️ Lỗi khi xử lý file CSV: {e}")
            traceback.print_exc()


# --- Tab 3: Thông tin Model (Giữ nguyên) ---
with tab3:
    # ... (Code Tab 3 giữ nguyên) ...
     st.header("Thông tin Đánh giá Model (XLM-RoBERTa)")
     # ... hiển thị metrics, report, cm, curves, error analysis ...


# --- Footer ---
st.markdown("---")
st.caption("Dự án Thực tập - Xử lý Phản hồi Khách hàng (Lai ghép) - [Tên của bạn]")