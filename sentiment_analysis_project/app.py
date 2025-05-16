# app.py (vFinal v2 - Tab 2 Phân tích & Gợi ý AI theo Product ID)

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
try:
    from visualization import plot_confusion_matrix, plot_training_history
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    print("Cảnh báo: Module 'visualization' không tìm thấy.")

# Import thư viện Gemini và kiểm tra cấu hình
gemini_configured_app = False
if config.GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        gemini_configured_app = True
        print("Gemini OK (Streamlit).")
    except ImportError:
        print("Cảnh báo: google-generativeai chưa cài.")
    except Exception as e:
        print(f"Cảnh báo: Lỗi cấu hình Gemini (Streamlit): {e}")
else:
    print("Cảnh báo: GEMINI_API_KEY chưa đặt (Streamlit).")

# --- Cấu hình Trang ---
st.set_page_config(page_title="Xử lý Phản hồi vFinal v2", page_icon="💡", layout="wide")

# --- Địa chỉ API Backend ---
API_HOST = getattr(config, 'API_HOST', '127.0.0.1')
API_PORT = getattr(config, 'API_PORT', 8000)
BACKEND_API_URL_SENTIMENT = f"http://{API_HOST}:{API_PORT}/sentiment/"
BACKEND_API_URL_PROCESS = f"http://{API_HOST}:{API_PORT}/process/"

# --- Giao diện Chính ---
st.title("💡 Hệ thống Phân tích & Xử lý Phản hồi Khách hàng (Product Aware) vFinal v2")
st.markdown("""
**Chọn cách xử lý:**
- **Phân tích Nhanh:** Chỉ lấy cảm xúc (nhanh, đọc/lưu vào KB). *Có thể kèm Product ID.*
- **Xử lý Chi tiết:** Lấy cảm xúc, gợi ý & phản hồi AI (đọc/làm giàu KB & gọi Gemini). *Có thể kèm Product ID.*
- **Xử lý Hàng loạt:** Phân tích nhanh file CSV (làm nóng KB), nhận **phân tích cảm xúc** và **gợi ý AI** chi tiết theo từng Product ID.
""")

# --- Các Tab chức năng ---
tab1, tab2, tab3 = st.tabs(["📝 Xử lý Đơn lẻ", "📂 Xử lý Hàng loạt (Theo Sản phẩm + AI)", "📈 Thông tin Model"])

# --- Tab 1: Xử lý Đơn lẻ (Giữ nguyên, hỗ trợ Product ID) ---
with tab1:
    st.header("Nhập phản hồi cần xử lý:")
    user_input_single = st.text_area("Nội dung bình luận:", height=120, key="single_input_tab1_prod_final", placeholder="Ví dụ: Chiếc áo này màu rất đẹp!")
    product_id_input = st.text_input("Mã/Tên Sản phẩm (Tùy chọn):", key="product_id_single_final", placeholder="Ví dụ: AO-001")

    col_btn1, col_btn2 = st.columns(2)

    def display_results_tab1(api_response, start_time, end_time, endpoint_name):
        st.markdown("---")
        st.subheader(f"Kết quả từ {endpoint_name}:")
        if not api_response:
            st.error(f"Không nhận được phản hồi hợp lệ từ API {endpoint_name}.")
            return
        total_time = (end_time - start_time) * 1000
        api_time = api_response.get('processing_time_ms')
        source = api_response.get('source', 'N/A')
        ai_reason = api_response.get('ai_call_reason')
        product_id_rcv = api_response.get('product_id_processed')
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("**Phân tích Cảm xúc:**")
            sentiment = api_response.get('sentiment', 'N/A')
            confidence = api_response.get('confidence')
            try:
                label_map = getattr(config, 'TARGET_LABEL_MAP', {})
                positive_label = label_map.get(2, "Tích cực")
                negative_label = label_map.get(0, "Tiêu cực")
            except:
                label_map = {}
                positive_label = "Tích cực"
                negative_label = "Tiêu cực"
            if sentiment == positive_label:
                st.success(f"**Cảm xúc:** {sentiment}")
            elif sentiment == negative_label:
                st.error(f"**Cảm xúc:** {sentiment}")
            else:
                st.warning(f"**Cảm xúc:** {sentiment}")
            if confidence is not None:
                st.metric(label="Độ tin cậy", value=f"{confidence:.2%}")
            if product_id_rcv:
                st.caption(f"Sản phẩm đã xử lý: {product_id_rcv}")
            st.caption(f"T.gian: {total_time:.0f}ms | API T.gian: {api_time:.0f}ms" if api_time else f"T.gian: {total_time:.0f}ms")
            source_text = {
                'cache': 'Cache KB',
                'cache_enriched': 'Làm giàu KB',
                'new_sentiment_only': 'Mới (Chỉ Sentiment)',
                'new_full_process': 'Mới (Full AI)',
                'error': 'Lỗi Xử lý'
            }.get(source, source)
            st.caption(f"Nguồn: {source_text}")
            if ai_reason and source != 'cache':
                st.caption(f"Trạng thái AI: {ai_reason}")
        with col_res2:
            st.markdown("**Gợi ý Phản hồi Tự động (AI/Cache):**")
            generated_response = api_response.get('generated_response')
            is_valid_response = generated_response and isinstance(generated_response, str) and "Lỗi" not in generated_response and "chưa cấu hình" not in generated_response and "không tạo ra" not in generated_response
            if is_valid_response:
                st.text_area("Nội dung:", value=generated_response, height=120, key=f"gen_resp_{source}_{int(time.time())}", disabled=False)
            elif generated_response:
                st.info(generated_response)
            else:
                st.info("Không có.")
        st.markdown("---")
        st.markdown("**Gợi ý Hành động Nội bộ (AI/Cache):**")
        suggestions = api_response.get('suggestions')
        is_valid_suggestions = suggestions and isinstance(suggestions, list) and not any("Lỗi" in s or "chưa cấu hình" in s for s in suggestions)
        if is_valid_suggestions:
            st.markdown("\n".join(f"- {s}" for s in suggestions))
        elif suggestions and isinstance(suggestions, list):
            st.info(suggestions[0])
        else:
            st.info("Không có.")

    with col_btn1:
        if st.button("⚡ Phân tích Nhanh (Đọc/Lưu KB)", key="analyze_fast_kb_final_prod", help="Lấy cảm xúc, đọc/lưu KB. Kèm Product ID nếu có."):
            if user_input_single and user_input_single.strip():
                start_time = time.time()
                api_response = None
                error_message = None
                payload = {"comment": user_input_single}
                if product_id_input and product_id_input.strip():
                    payload["product_id"] = product_id_input.strip()
                with st.spinner('⚡ Đang phân tích nhanh & kiểm tra KB...'):
                    try:
                        response = requests.post(BACKEND_API_URL_SENTIMENT, json=payload, timeout=30)
                        response.raise_for_status()
                        api_response = response.json()
                    except Exception as e:
                        error_message = f"Lỗi API /sentiment/: {e}"
                        traceback.print_exc()
                end_time = time.time()
                if error_message:
                    st.error(error_message)
                display_results_tab1(api_response, start_time, end_time, "/sentiment/")
            else:
                st.warning("⚠️ Vui lòng nhập bình luận.")

    with col_btn2:
        if st.button("✨ Xử lý Chi tiết (KB + AI)", key="analyze_detailed_kb_final_prod", help="Đọc KB, nếu thiếu -> XLM-R + Gemini -> Lưu/Cập nhật KB. Kèm Product ID nếu có."):
            if user_input_single and user_input_single.strip():
                start_time = time.time()
                api_response = None
                error_message = None
                payload = {"comment": user_input_single}
                if product_id_input and product_id_input.strip():
                    payload["product_id"] = product_id_input.strip()
                with st.spinner('✨ Đang xử lý chi tiết...'):
                    try:
                        response = requests.post(BACKEND_API_URL_PROCESS, json=payload, timeout=180)
                        response.raise_for_status()
                        api_response = response.json()
                    except Exception as e:
                        error_message = f"Lỗi API /process/: {e}"
                        traceback.print_exc()
                end_time = time.time()
                if error_message:
                    st.error(error_message)
                    st.info("Mẹo: Kiểm tra server API & GEMINI_API_KEY.")
                display_results_tab1(api_response, start_time, end_time, "/process/")
            else:
                st.warning("⚠️ Vui lòng nhập bình luận.")

# --- Tab 2: Xử lý Hàng loạt (Phân tích & Gợi ý AI theo Product ID) ---
with tab2:
    st.header("Phân tích Hàng loạt và Gợi ý AI theo Sản phẩm")
    st.markdown("Tải lên file CSV có cột chứa **bình luận** và cột chứa **Mã/Tên Sản phẩm** để phân tích cảm xúc và nhận gợi ý AI chi tiết cho từng sản phẩm.")

    # Lấy tên cột từ config
    comment_col_name_cfg = getattr(config, 'TEXT_COLUMN', 'comment')
    # Thêm ô nhập tên cột product_id, để trống ban đầu nhưng giữ placeholder
    product_id_col_name_input = st.text_input(
        "Nhập tên cột chứa Mã/Tên Sản phẩm trong file CSV của bạn:",
        placeholder="Ví dụ: product_id, Product Name, MaSP,...",
        key="product_id_col_csv"
    )

    uploaded_file_batch = st.file_uploader(
        f"Chọn file CSV (cần cột '{comment_col_name_cfg}' và cột Sản phẩm bạn vừa nhập)",
        type=["csv"],
        key="csv_product_analysis"
    )
    limit_rows_batch_prod = st.number_input(
        "Giới hạn số dòng xử lý (Nhập 0 để xử lý tất cả):",
        min_value=0,
        value=0,
        step=50,
        key="limit_rows_batch_prod",
        help="Để 0 nếu muốn xử lý toàn bộ file."
    )

    if uploaded_file_batch is not None and product_id_col_name_input.strip():
        product_id_col_actual = product_id_col_name_input.strip()
        try:
            df_batch = None
            with st.spinner("Đang đọc CSV..."):
                try:
                    df_batch = pd.read_csv(uploaded_file_batch, encoding='utf-8-sig', low_memory=False)
                except UnicodeDecodeError:
                    df_batch = pd.read_csv(uploaded_file_batch, encoding='utf-8', low_memory=False)
                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")
                    st.write("Nội dung file (dạng text):", uploaded_file_batch.getvalue().decode('utf-8', errors='ignore'))
                    st.stop()

            st.success(f"✅ File '{uploaded_file_batch.name}' đã được tải thành công! (Tổng {len(df_batch)} dòng)")

            # Kiểm tra sự tồn tại của cả 2 cột
            if comment_col_name_cfg not in df_batch.columns:
                st.error(f"Lỗi: Không tìm thấy cột bình luận '{comment_col_name_cfg}' trong file CSV.")
                st.stop()
            if product_id_col_actual not in df_batch.columns:
                st.error(f"Lỗi: Không tìm thấy cột sản phẩm '{product_id_col_actual}' trong file CSV.")
                st.stop()

            if st.button("📊 Phân tích theo Sản phẩm & Nhận Gợi ý AI", key="analyze_csv_by_product"):
                # Xác định số dòng xử lý
                if limit_rows_batch_prod > 0 and limit_rows_batch_prod < len(df_batch):
                    process_df_batch = df_batch.head(limit_rows_batch_prod)
                    limit_info_batch = f"{limit_rows_batch_prod} dòng đầu"
                else:
                    process_df_batch = df_batch
                    limit_info_batch = "tất cả các dòng"
                total_to_process_batch = len(process_df_batch)
                if total_to_process_batch == 0:
                    st.warning("Không có dòng nào để xử lý.")
                    st.stop()

                st.info(f"Bắt đầu phân tích cảm xúc cho {limit_info_batch} (lưu vào KB)...")
                results_list_batch = []
                error_count_batch = 0
                cache_hit_count = 0
                potential_ai_call_count = 0
                start_batch_run_time = time.time()
                progress_text_batch = f"Phân tích cảm xúc 0/{total_to_process_batch} dòng..."
                progress_bar_batch = st.progress(0, text=progress_text_batch)

                # Lấy cấu hình kiểm tra AI
                conf_threshold_batch = float(getattr(config, 'CONFIDENCE_THRESHOLD', 0.80))
                check_negative_batch = bool(getattr(config, 'ALWAYS_CHECK_NEGATIVE', True))
                label_map_batch = getattr(config, 'TARGET_LABEL_MAP', {})
                negative_label_value_batch = label_map_batch.get(0, "Tiêu cực")

                # --- Bước 1: Phân tích cảm xúc hàng loạt bằng API /sentiment/ ---
                for index, row in process_df_batch.iterrows():
                    comment_text = str(row[comment_col_name_cfg]) if pd.notna(row[comment_col_name_cfg]) else ""
                    product_id = str(row[product_id_col_actual]) if pd.notna(row[product_id_col_actual]) else "N/A"
                    result_row = {
                        "original_comment": comment_text,
                        "product_id": product_id,
                        "sentiment": None,
                        "confidence": None,
                        "source": None,
                        "status": None,
                        "would_call_ai": False
                    }
                    if comment_text:
                        try:
                            payload = {"comment": comment_text, "product_id": product_id}
                            response = requests.post(BACKEND_API_URL_SENTIMENT, json=payload, timeout=60)
                            response.raise_for_status()
                            api_data = response.json()
                            result_row.update({k: api_data.get(k) for k in ['sentiment', 'confidence', 'source']})
                            result_row['status'] = 'Thành công'
                            if api_data.get('source') == 'cache':
                                cache_hit_count += 1
                            # Ước tính gọi AI
                            would_call = False
                            if api_data.get('source') != 'cache':
                                if api_data.get('confidence') is not None and api_data.get('confidence') < conf_threshold_batch:
                                    would_call = True
                                elif check_negative_batch and api_data.get('sentiment') == negative_label_value_batch:
                                    would_call = True
                            elif api_data.get('source') == 'cache' and (api_data.get('suggestions') is None or api_data.get('generated_response') is None):
                                would_call = True
                            result_row['would_call_ai'] = would_call
                            if would_call:
                                potential_ai_call_count += 1
                        except requests.exceptions.Timeout:
                            result_row['status'] = 'Lỗi API: Timeout'
                            error_count_batch += 1
                        except requests.exceptions.RequestException as e:
                            result_row['status'] = f'Lỗi API: {type(e).__name__}'
                            error_count_batch += 1
                        except Exception as e:
                            result_row['status'] = f'Lỗi khác: {type(e).__name__}'
                            error_count_batch += 1
                    else:
                        result_row['status'] = 'Bỏ qua (rỗng)'
                    results_list_batch.append(result_row)
                    progress_percentage = (index + 1) / total_to_process_batch
                    progress_text_batch = f"Phân tích cảm xúc {index + 1}/{total_to_process_batch} dòng..."
                    progress_bar_batch.progress(progress_percentage, text=progress_text_batch)

                end_batch_run_time = time.time()
                progress_bar_batch.empty()
                st.success(f"✅ Phân tích cảm xúc {total_to_process_batch} dòng hoàn tất sau {end_batch_run_time - start_batch_run_time:.2f} giây.")

                # --- Bước 2: Tổng hợp kết quả và Gọi Gemini cho từng Product ID ---
                if results_list_batch:
                    results_df_batch = pd.DataFrame(results_list_batch)
                    st.markdown("---")
                    st.subheader("📊 Thống kê Chung (Toàn bộ File)")
                    col_b_stat1, col_b_stat2, col_b_stat3, col_b_stat4 = st.columns(4)
                    with col_b_stat1:
                        st.metric("Tổng dòng xử lý", total_to_process_batch)
                    with col_b_stat2:
                        st.metric("Số dòng gặp lỗi", error_count_batch)
                    with col_b_stat3:
                        st.metric("Số lần dùng Cache KB", cache_hit_count)
                    with col_b_stat4:
                        st.metric("Ước tính cần Gọi AI*", potential_ai_call_count, help="Số dòng có độ tin cậy thấp/tiêu cực hoặc cần làm giàu KB, sẽ gọi Gemini nếu dùng 'Xử lý Chi tiết'.")

                    # --- Dashboard Tổng hợp: Thống kê phản hồi tốt/xấu/trung tính ---
                    st.markdown("---")
                    st.subheader("🌟 Dashboard Tổng hợp: Phân tích Cảm xúc Toàn bộ File")
                    
                    # Lọc các dòng phân tích thành công
                    valid_df = results_df_batch[results_df_batch['status'] == 'Thành công']
                    total_valid = len(valid_df)
                    
                    if total_valid > 0:
                        # Tính số lượng từng loại cảm xúc
                        sentiment_counts_total = valid_df['sentiment'].value_counts()
                        all_labels_cfg = list(getattr(config, 'TARGET_LABEL_MAP', {}).values())
                        sentiment_counts_total = sentiment_counts_total.reindex(all_labels_cfg, fill_value=0)
                        color_map_cfg = {"Tiêu cực": '#DC143C', "Trung tính": '#FFD700', "Tích cực": '#32CD32'}
                        counts_to_plot_total = sentiment_counts_total[sentiment_counts_total.index.isin(color_map_cfg.keys())]

                        # Hiển thị số liệu và biểu đồ
                        col_total_chart, col_total_stats = st.columns([2, 1])
                        
                        with col_total_chart:
                            # Vẽ biểu đồ tròn
                            fig_pie_total = px.pie(
                                names=counts_to_plot_total.index,
                                values=counts_to_plot_total.values,
                                title="Tỷ lệ Cảm xúc Toàn bộ File",
                                color=counts_to_plot_total.index,
                                color_discrete_map=color_map_cfg,
                                height=300
                            )
                            st.plotly_chart(fig_pie_total, use_container_width=True)
                        
                        with col_total_stats:
                            # Tính phần trăm
                            pos_count = counts_to_plot_total.get("Tích cực", 0)
                            neg_count = counts_to_plot_total.get("Tiêu cực", 0)
                            neu_count = counts_to_plot_total.get("Trung tính", 0)
                            pos_p = (pos_count / total_valid) * 100 if total_valid > 0 else 0
                            neg_p = (neg_count / total_valid) * 100 if total_valid > 0 else 0
                            neu_p = (neu_count / total_valid) * 100 if total_valid > 0 else 0

                            st.markdown("**Thống kê Phản hồi:**")
                            st.markdown(f"- **Phản hồi Tốt (Tích cực):** {pos_count} ({pos_p:.1f}%)")
                            st.markdown(f"- **Phản hồi Xấu (Tiêu cực):** {neg_count} ({neg_p:.1f}%)")
                            st.markdown(f"- **Phản hồi Trung tính:** {neu_count} ({neu_p:.1f}%)")
                            st.markdown("---")
                            st.markdown("**Nhận xét Tổng quan:**")
                            if pos_p >= 65:
                                st.success("Phần lớn phản hồi là Tích cực, cho thấy khách hàng hài lòng.")
                            elif neg_p >= 35:
                                st.error("Tỷ lệ phản hồi Tiêu cực cao, cần xem xét và cải thiện ngay.")
                            elif neg_p >= 20:
                                st.warning("Tỷ lệ phản hồi Tiêu cực đáng chú ý, nên kiểm tra chi tiết.")
                            else:
                                st.info("Phản hồi tương đối cân bằng, cần theo dõi thêm.")
                    else:
                        st.warning("Không có dữ liệu hợp lệ để hiển thị Dashboard Tổng hợp.")

                    # --- Hiển thị Phân tích theo từng Product ID ---
                    st.markdown("---")
                    st.subheader("💎 Phân tích & Gợi ý AI theo từng Sản phẩm")
                    all_products = results_df_batch[results_df_batch['status'] == 'Thành công']['product_id'].unique()

                    if not all_products.size:
                        st.warning("Không có sản phẩm nào được xử lý thành công để phân tích.")
                    else:
                        for prod_id in all_products:
                            with st.expander(f"Kết quả cho Sản phẩm: **{prod_id}**"):
                                prod_df = results_df_batch[(results_df_batch['product_id'] == prod_id) & (results_df_batch['status'] == 'Thành công')]
                                if prod_df.empty:
                                    st.write("Không có dữ liệu hợp lệ cho sản phẩm này.")
                                    continue

                                st.markdown(f"**Tổng số phản hồi cho sản phẩm này:** {len(prod_df)}")
                                sentiment_counts_prod = prod_df['sentiment'].value_counts()
                                all_labels_cfg = list(getattr(config, 'TARGET_LABEL_MAP', {}).values())
                                sentiment_counts_prod = sentiment_counts_prod.reindex(all_labels_cfg, fill_value=0)
                                color_map_cfg = {"Tiêu cực": '#DC143C', "Trung tính": '#FFD700', "Tích cực": '#32CD32'}
                                counts_to_plot_prod = sentiment_counts_prod[sentiment_counts_prod.index.isin(color_map_cfg.keys())]

                                if not counts_to_plot_prod.empty:
                                    col_p_chart, col_p_stats = st.columns([2, 1])
                                    with col_p_chart:
                                        fig_bar_prod = px.bar(
                                            counts_to_plot_prod,
                                            x=counts_to_plot_prod.index,
                                            y=counts_to_plot_prod.values,
                                            labels={'x': 'Cảm xúc', 'y': 'Số lượng'},
                                            color=counts_to_plot_prod.index,
                                            color_discrete_map=color_map_cfg,
                                            text=counts_to_plot_prod.values,
                                            height=300
                                        )
                                        fig_bar_prod.update_layout(showlegend=False, title_text=f"Cảm xúc cho SP: {prod_id}", title_x=0.5)
                                        st.plotly_chart(fig_bar_prod, use_container_width=True)
                                    with col_p_stats:
                                        total_valid_prod = counts_to_plot_prod.sum()
                                        if total_valid_prod > 0:
                                            pos_p = (counts_to_plot_prod.get("Tích cực", 0) / total_valid_prod) * 100
                                            neg_p = (counts_to_plot_prod.get("Tiêu cực", 0) / total_valid_prod) * 100
                                            neu_p = (counts_to_plot_prod.get("Trung tính", 0) / total_valid_prod) * 100
                                            st.markdown("**Phân phối Cảm xúc:**")
                                            st.markdown(f"- **Tích cực:** {counts_to_plot_prod.get('Tích cực', 0)} ({pos_p:.1f}%)")
                                            st.markdown(f"- **Trung tính:** {counts_to_plot_prod.get('Trung tính', 0)} ({neu_p:.1f}%)")
                                            st.markdown(f"- **Tiêu cực:** {counts_to_plot_prod.get('Tiêu cực', 0)} ({neg_p:.1f}%)")
                                            st.markdown("---")
                                            st.markdown("**Xu hướng chính:**")
                                            if pos_p >= 65:
                                                st.success("-> Phần lớn phản hồi là Tích cực.")
                                            elif neg_p >= 35:
                                                st.error("-> Tỷ lệ Tiêu cực rất cao, cần hành động ngay.")
                                            elif neg_p >= 20:
                                                st.warning("-> Tỷ lệ Tiêu cực đáng chú ý, cần xem xét.")
                                            else:
                                                st.info("-> Tỷ lệ cảm xúc tương đối cân bằng.")
                                            st.markdown("---")

                                            # Gọi Gemini cho từng sản phẩm
                                            st.markdown("**Gợi ý Hành động (AI):**")
                                            if gemini_configured_app:
                                                with st.spinner(f"Đang lấy gợi ý AI cho sản phẩm {prod_id}..."):
                                                    prompt_prod_summary = f"""Phân tích cảm xúc cho sản phẩm '{prod_id}': Tích cực {pos_p:.1f}%, Trung tính {neu_p:.1f}%, Tiêu cực {neg_p:.1f}%.
Đề xuất 2-3 hành động cụ thể cho sản phẩm này. Định dạng: danh sách gạch đầu dòng."""
                                                    try:
                                                        model_gen_prod = genai.GenerativeModel('gemini-1.5-flash')
                                                        response_gen_prod = model_gen_prod.generate_content(prompt_prod_summary)
                                                        prod_suggestions = response_gen_prod.text.strip().split('\n')
                                                        if prod_suggestions:
                                                            for sugg in prod_suggestions:
                                                                if sugg.strip():
                                                                    st.markdown(f"- {sugg.strip().lstrip('-* ')}")
                                                        else:
                                                            st.info("AI không đưa ra gợi ý.")
                                                    except Exception as gemini_e_prod:
                                                        st.warning(f"Lỗi gọi AI cho SP {prod_id}: {gemini_e_prod}")
                                            else:
                                                st.info("Gemini chưa cấu hình. Sử dụng gợi ý mẫu.")
                                        else:
                                            st.info("Không có dữ liệu cảm xúc hợp lệ cho sản phẩm này.")
                                else:
                                    st.info("Không có dữ liệu cảm xúc hợp lệ để vẽ biểu đồ cho sản phẩm này.")
                    # --- Nút Tải xuống ---
                    st.markdown("---")
                    st.subheader("💾 Tải xuống Kết quả Phân tích Hàng loạt")
                    @st.cache_data
                    def convert_batch_product_df(df_to_convert):
                        cols = ["original_comment", "product_id", "sentiment", "confidence", "source", "status", "would_call_ai"]
                        existing = [c for c in cols if c in df_to_convert.columns]
                        try:
                            return df_to_convert[existing].to_csv(index=False, encoding='utf-8-sig')
                        except:
                            return None
                    csv_batch_prod_output = convert_batch_product_df(results_df_batch)
                    if csv_batch_prod_output:
                        st.download_button(
                            label="📥 Tải Kết quả (CSV)",
                            data=csv_batch_prod_output,
                            file_name=f'ket_qua_batch_{uploaded_file_batch.name}.csv',
                            mime='text/csv'
                        )
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
    st.markdown("Phần này hiển thị các số liệu đánh giá và biểu đồ liên quan đến hiệu suất của mô hình XLM-RoBERTa được sử dụng để phân tích cảm xúc.")
    if VIZ_AVAILABLE:
        st.subheader("📊 Biểu đồ Hiệu suất")
        try:
            confusion_fig = plot_confusion_matrix()
            if confusion_fig:
                st.plotly_chart(confusion_fig, use_container_width=True)
            history_fig = plot_training_history()
            if history_fig:
                st.plotly_chart(history_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi khi tạo biểu đồ: {e}")
    else:
        st.warning("Không thể tải module visualization. Vui lòng kiểm tra file visualization.py.")

    st.subheader("📈 Số liệu Hiệu suất")
    st.markdown("Dưới đây là các số liệu hiệu suất mẫu (cần thay thế bằng dữ liệu thực tế từ mô hình):")
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    with col_metrics1:
        st.metric("Accuracy", "85%")
    with col_metrics2:
        st.metric("F1-Score (Tích cực)", "0.88")
    with col_metrics3:
        st.metric("F1-Score (Tiêu cực)", "0.82")

# --- Footer ---
st.markdown("---")
st.caption("Dự án Thực tập - Xử lý Phản hồi Khách hàng - [Nguyễn Trần Hoàng Thịnh]")