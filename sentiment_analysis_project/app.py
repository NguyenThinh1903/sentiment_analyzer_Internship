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
from datetime import datetime # Thêm datetime

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
             st.info(suggestions[0] if suggestions else "Không có.")
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
                    except requests.exceptions.Timeout:
                        error_message = f"Lỗi API /sentiment/: Timeout. Máy chủ có thể đang bận."
                    except requests.exceptions.HTTPError as http_err:
                        error_message = f"Lỗi HTTP API /sentiment/: {http_err}. Nội dung: {response.text if response else 'N/A'}"
                    except requests.exceptions.RequestException as req_err:
                        error_message = f"Lỗi kết nối API /sentiment/: {req_err}"
                    except Exception as e:
                        error_message = f"Lỗi không xác định khi gọi API /sentiment/: {e}"
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
                with st.spinner('✨ Đang xử lý chi tiết... (Có thể mất vài chục giây)'):
                    try:
                        response = requests.post(BACKEND_API_URL_PROCESS, json=payload, timeout=180)
                        response.raise_for_status()
                        api_response = response.json()
                    except requests.exceptions.Timeout:
                        error_message = f"Lỗi API /process/: Timeout. Máy chủ hoặc Gemini có thể đang bận."
                    except requests.exceptions.HTTPError as http_err:
                        error_message = f"Lỗi HTTP API /process/: {http_err}. Nội dung: {response.text if response else 'N/A'}"
                    except requests.exceptions.RequestException as req_err:
                        error_message = f"Lỗi kết nối API /process/: {req_err}"
                    except Exception as e:
                        error_message = f"Lỗi không xác định khi gọi API /process/: {e}"
                        traceback.print_exc()
                end_time = time.time()
                if error_message:
                    st.error(error_message)
                    st.info("Mẹo: Kiểm tra kết nối mạng, server API backend và cấu hình GEMINI_API_KEY.")
                display_results_tab1(api_response, start_time, end_time, "/process/")
            else:
                st.warning("⚠️ Vui lòng nhập bình luận.")

# --- Tab 2: Xử lý Hàng loạt (Phân tích & Gợi ý AI theo Product ID) ---
with tab2:
    st.header("Phân tích Hàng loạt và Gợi ý AI theo Sản phẩm")
    st.markdown("Tải lên file CSV có cột chứa **bình luận** và cột chứa **Mã/Tên Sản phẩm** để phân tích cảm xúc và nhận gợi ý AI chi tiết cho từng sản phẩm.")

    comment_col_name_cfg = getattr(config, 'TEXT_COLUMN', 'comment')
    product_id_col_name_input = st.text_input(
        "Nhập tên cột chứa Mã/Tên Sản phẩm trong file CSV của bạn:",
        placeholder="Ví dụ: product_id, Product Name, MaSP,... (phân biệt chữ hoa/thường)",
        key="product_id_col_csv"
    )

    uploaded_file_batch = st.file_uploader(
        f"Chọn file CSV (cần cột '{comment_col_name_cfg}' và cột Sản phẩm bạn vừa nhập)",
        type=["csv"],
        key="csv_product_analysis_v2"
    )
    limit_rows_batch_prod = st.number_input(
        "Giới hạn số dòng xử lý (Nhập 0 để xử lý tất cả):",
        min_value=0,
        value=50, 
        step=50,
        key="limit_rows_batch_prod_v2",
        help="Để 0 nếu muốn xử lý toàn bộ file. Cẩn thận với file lớn có thể tốn thời gian."
    )

    if uploaded_file_batch is not None and product_id_col_name_input.strip():
        product_id_col_actual = product_id_col_name_input.strip()
        try:
            df_batch_original = None
            with st.spinner("Đang đọc CSV..."):
                try:
                    try:
                        df_batch_original = pd.read_csv(uploaded_file_batch, encoding='utf-8-sig', low_memory=False)
                    except UnicodeDecodeError:
                        try:
                            df_batch_original = pd.read_csv(uploaded_file_batch, encoding='utf-8', low_memory=False)
                        except UnicodeDecodeError:
                            df_batch_original = pd.read_csv(uploaded_file_batch, encoding='latin-1', low_memory=False) 
                except Exception as e:
                    st.error(f"Lỗi đọc file CSV: {e}. Vui lòng kiểm tra định dạng file.")
                    # In ra một phần nội dung file để debug
                    uploaded_file_batch.seek(0) 
                    st.text_area("Nội dung đầu file (để debug):", uploaded_file_batch.read(1000).decode('utf-8', errors='ignore'), height=150)
                    st.stop()
            
            if df_batch_original is None:
                st.error("Không thể đọc được nội dung file CSV.")
                st.stop()

            st.success(f"✅ File '{uploaded_file_batch.name}' đã được tải thành công! (Tổng {len(df_batch_original)} dòng)")
            st.dataframe(df_batch_original.head())

            # Kiểm tra sự tồn tại của cả 2 cột (phân biệt chữ hoa/thường)
            if comment_col_name_cfg not in df_batch_original.columns:
                st.error(f"Lỗi: Không tìm thấy cột bình luận '{comment_col_name_cfg}' trong file CSV. Các cột hiện có: {', '.join(df_batch_original.columns)}")
                st.stop()
            if product_id_col_actual not in df_batch_original.columns:
                st.error(f"Lỗi: Không tìm thấy cột sản phẩm '{product_id_col_actual}' trong file CSV. Các cột hiện có: {', '.join(df_batch_original.columns)}")
                st.stop()

            if st.button("📊 Phân tích theo Sản phẩm & Nhận Gợi ý AI", key="analyze_csv_by_product_v2"):
                if limit_rows_batch_prod > 0 and limit_rows_batch_prod < len(df_batch_original):
                    process_df_batch = df_batch_original.head(limit_rows_batch_prod).copy()
                    limit_info_batch = f"{limit_rows_batch_prod} dòng đầu"
                else:
                    process_df_batch = df_batch_original.copy()
                    limit_info_batch = "tất cả các dòng"
                
                total_to_process_batch = len(process_df_batch)
                if total_to_process_batch == 0:
                    st.warning("Không có dòng nào để xử lý.")
                    st.stop()

                st.info(f"Bắt đầu phân tích cảm xúc cho {limit_info_batch} (tương tác với KB)...")
                results_list_batch = []
                error_count_batch = 0
                cache_hit_count = 0
                
                start_batch_run_time = time.time()
                progress_bar_batch = st.progress(0)
                progress_text_container = st.empty()

                # Lấy cấu hình kiểm tra AI (nếu dùng cho would_call_ai)
                conf_threshold_batch = float(getattr(config, 'CONFIDENCE_THRESHOLD', 0.80))
                check_negative_batch = bool(getattr(config, 'ALWAYS_CHECK_NEGATIVE', True)) 
                label_map_batch = getattr(config, 'TARGET_LABEL_MAP', {})
                negative_label_value_batch = ""
                for k, v in label_map_batch.items():
                    if k == 0:
                        negative_label_value_batch = v
                        break
                
                # --- Bước 1: Phân tích cảm xúc hàng loạt bằng API /sentiment/ ---
                for index, row in process_df_batch.iterrows():
                    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    comment_text = str(row[comment_col_name_cfg]) if pd.notna(row[comment_col_name_cfg]) else ""
                    product_id_val = str(row[product_id_col_actual]) if pd.notna(row[product_id_col_actual]) and str(row[product_id_col_actual]).strip() else "N/A"

                    result_row = {
                        "original_comment": comment_text,
                        "product_id": product_id_val,
                        "sentiment": None,
                        "confidence": None,
                        "source": None,
                        "kb_has_ai_details": False,
                        "status": "Chưa xử lý",
                        "would_call_ai": False,
                        "processing_timestamp": current_time_str
                    }

                    if comment_text.strip():
                        try:
                            payload = {"comment": comment_text, "product_id": product_id_val}
                            response = requests.post(BACKEND_API_URL_SENTIMENT, json=payload, timeout=60)
                            response.raise_for_status()
                            api_data = response.json()
                            
                            result_row.update({
                                'sentiment': api_data.get('sentiment'),
                                'confidence': api_data.get('confidence'),
                                'source': api_data.get('source')
                            })
                            result_row['status'] = 'Thành công'

                            if api_data.get('source') == 'cache':
                                cache_hit_count += 1
                            
                            if api_data.get('suggestions') is not None or api_data.get('generated_response') is not None:
                                result_row['kb_has_ai_details'] = True

                            # Ước tính gọi AI (would_call_ai)
                            would_call = False
                            # Trường hợp 1: Mới phân tích (không phải cache) VÀ (độ tin cậy thấp HOẶC là tiêu cực)
                            if api_data.get('source') != 'cache':
                                if (api_data.get('confidence') is not None and api_data.get('confidence') < conf_threshold_batch) or \
                                   (check_negative_batch and api_data.get('sentiment') == negative_label_value_batch):
                                    would_call = True
                            # Trường hợp 2: Từ cache NHƯNG chưa có chi tiết AI
                            elif api_data.get('source') == 'cache' and not result_row['kb_has_ai_details']:
                                would_call = True
                            result_row['would_call_ai'] = would_call

                        except requests.exceptions.Timeout:
                            result_row['status'] = 'Lỗi API: Timeout'
                            error_count_batch += 1
                        except requests.exceptions.HTTPError as http_err:
                            result_row['status'] = f'Lỗi HTTP API ({http_err.response.status_code if http_err.response else "N/A"})'
                            error_count_batch += 1
                        except requests.exceptions.RequestException as e:
                            result_row['status'] = f'Lỗi kết nối API: {type(e).__name__}'
                            error_count_batch += 1
                        except Exception as e:
                            result_row['status'] = f'Lỗi khác: {type(e).__name__}'
                            error_count_batch += 1
                    else:
                        result_row['status'] = 'Bỏ qua (bình luận rỗng)'
                    
                    results_list_batch.append(result_row)

                    progress_percentage = (index + 1) / total_to_process_batch
                    progress_text_container.text(f"Đang xử lý dòng {index + 1}/{total_to_process_batch}...")
                    progress_bar_batch.progress(progress_percentage)

                end_batch_run_time = time.time()
                progress_text_container.text(f"Hoàn thành phân tích {total_to_process_batch} dòng!")
                st.success(f"✅ Phân tích cảm xúc {total_to_process_batch} dòng hoàn tất sau {end_batch_run_time - start_batch_run_time:.2f} giây.")

                # --- Bước 2: Tổng hợp kết quả và Gọi Gemini cho từng Product ID ---
                if results_list_batch:
                    results_df_batch = pd.DataFrame(results_list_batch)
                  # --- Đoạn code mới để tạo giao diện 2x2 và hiển thị ô gọi AI ---

                    # Tính toán số lượt cần gọi AI
                    ai_call_count = results_df_batch['would_call_ai'].sum()

                    st.markdown("---")
                    st.subheader("📊 Thống kê Chung (Toàn bộ File)")

                    # Tạo hàng thứ nhất với 2 cột
                    col_b_stat1, col_b_stat2 = st.columns(2)
                    with col_b_stat1:
                        st.metric("Tổng dòng đã xử lý", total_to_process_batch)
                    with col_b_stat2:
                        st.metric("Số dòng gặp lỗi API", error_count_batch, delta_color="inverse")

                    # Tạo hàng thứ hai với 2 cột
                    col_b_stat3, col_b_stat4 = st.columns(2)
                    with col_b_stat3:
                        st.metric("Số lần dùng Cache KB", cache_hit_count)
                    with col_b_stat4:
                        # Đây là ô mới sẽ hiển thị số lần gọi AI
                        st.metric(
                            "Số dòng cần gọi AI (Ước tính)",
                            value=int(ai_call_count), 
                            help="Số dòng này được ước tính dựa trên các điều kiện: bình luận mới (chưa có trong KB) và là tiêu cực/độ tin cậy thấp, hoặc bình luận đã có trong KB nhưng thiếu gợi ý AI chi tiết."
                        )
                        
                    # --- Dashboard Tổng hợp ---
                    st.markdown("---")
                    st.subheader("🌟 Dashboard Tổng hợp: Phân tích Cảm xúc Toàn bộ File")
                    valid_sentiment_df = results_df_batch[results_df_batch['status'] == 'Thành công'].copy()
                    
                    if not valid_sentiment_df.empty:
                        sentiment_counts_total = valid_sentiment_df['sentiment'].value_counts()
                        all_labels_cfg = list(getattr(config, 'TARGET_LABEL_MAP', {0:"Tiêu cực", 1:"Trung tính", 2:"Tích cực"}).values())
                        for label in all_labels_cfg:
                            if label not in sentiment_counts_total:
                                sentiment_counts_total[label] = 0
                        
                        color_map_cfg = {"Tiêu cực": '#DC143C', "Trung tính": '#FFD700', "Tích cực": '#32CD32', "Không xác định": "#808080"} # Thêm màu cho không xác định
                        
                        col_total_chart, col_total_stats = st.columns([2, 1])
                        with col_total_chart:
                            fig_pie_total = px.pie(
                                sentiment_counts_total,
                                names=sentiment_counts_total.index,
                                values=sentiment_counts_total.values,
                                title="Tỷ lệ Cảm xúc Toàn bộ File",
                                color=sentiment_counts_total.index,
                                color_discrete_map=color_map_cfg,
                                height=350
                            )
                            fig_pie_total.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_pie_total, use_container_width=True)
                        
                        with col_total_stats:
                            total_valid_sentiments = sentiment_counts_total.sum()
                            st.markdown("**Thống kê Phản hồi:**")
                            for label, count in sentiment_counts_total.items():
                                percentage = (count / total_valid_sentiments) * 100 if total_valid_sentiments > 0 else 0
                                st.markdown(f"- **{label}:** {count} ({percentage:.1f}%)")

                    else:
                        st.warning("Không có dữ liệu cảm xúc hợp lệ để hiển thị Dashboard Tổng hợp.")

                    # --- Phân tích theo từng Product ID ---
                    st.markdown("---")
                    st.subheader("💎 Phân tích & Gợi ý AI theo từng Sản phẩm")
                    
                    # Lấy danh sách sản phẩm duy nhất từ các dòng xử lý thành công
                    unique_products = valid_sentiment_df['product_id'].unique()
                    
                    if not unique_products.size: # Kiểm tra xem mảng có rỗng không
                        st.warning("Không có sản phẩm nào được xử lý thành công để phân tích chi tiết.")
                    else:
                        for prod_id in unique_products:
                            if prod_id == "N/A": # Có thể bỏ qua N/A hoặc xử lý riêng
                                st.markdown(f"**Kết quả cho các bình luận không có Product ID (N/A)**")
                            else:
                                st.markdown(f"**Kết quả cho Sản phẩm: `{prod_id}`**")

                            with st.expander(f"Xem chi tiết và gợi ý AI cho '{prod_id}'", expanded=(prod_id != "N/A")):
                                prod_specific_df = valid_sentiment_df[valid_sentiment_df['product_id'] == prod_id]
                                if prod_specific_df.empty:
                                    st.write("Không có dữ liệu cảm xúc hợp lệ cho sản phẩm này.")
                                    continue

                                st.markdown(f"**Tổng số phản hồi hợp lệ cho sản phẩm này:** {len(prod_specific_df)}")
                                sentiment_counts_prod = prod_specific_df['sentiment'].value_counts()
                                # Đảm bảo tất cả các nhãn đều có
                                for label in all_labels_cfg:
                                    if label not in sentiment_counts_prod:
                                        sentiment_counts_prod[label] = 0
                                
                                col_p_chart, col_p_stats_ai = st.columns([1, 1])
                                with col_p_chart:
                                    fig_bar_prod = px.bar(
                                        sentiment_counts_prod,
                                        x=sentiment_counts_prod.index,
                                        y=sentiment_counts_prod.values,
                                        labels={'x': 'Cảm xúc', 'y': 'Số lượng'},
                                        color=sentiment_counts_prod.index,
                                        color_discrete_map=color_map_cfg,
                                        text_auto=True, 
                                        height=300
                                    )
                                    fig_bar_prod.update_layout(showlegend=False, title_text=f"Cảm xúc SP: {prod_id}", title_x=0.5, xaxis_title=None, yaxis_title="Số lượng")
                                    st.plotly_chart(fig_bar_prod, use_container_width=True)

                                with col_p_stats_ai:
                                    total_prod_sentiments = sentiment_counts_prod.sum()
                                    st.markdown("**Phân phối:**")
                                    if total_prod_sentiments > 0:
                                        for label, count in sentiment_counts_prod.items():
                                            percentage = (count / total_prod_sentiments) * 100
                                            st.markdown(f"- {label}: {count} ({percentage:.1f}%)")
                                        
                                        st.markdown("---")
                                        st.markdown("**Gợi ý Hành động (AI):**")
                                        if gemini_configured_app:
                                            # Tạo prompt tóm tắt
                                            summary_parts = []
                                            for label, count in sentiment_counts_prod.items():
                                                if count > 0:
                                                    percentage = (count / total_prod_sentiments) * 100
                                                    summary_parts.append(f"{label} {percentage:.0f}% ({count} bình luận)")
                                            sentiment_summary_for_prompt = ", ".join(summary_parts)
                                            
                                            prompt_prod_summary = f"""Sản phẩm: '{prod_id}'.
Tóm tắt cảm xúc khách hàng: {sentiment_summary_for_prompt}.
Dựa trên tóm tắt này, hãy đề xuất 2-3 hành động cụ thể và ưu tiên mà bộ phận Chăm sóc Khách hàng hoặc Phát triển Sản phẩm nên thực hiện để cải thiện trải nghiệm khách hàng hoặc sản phẩm. Trình bày dưới dạng gạch đầu dòng ngắn gọn.
Ví dụ Gợi ý:
- [Hành động cụ thể 1]
- [Hành động cụ thể 2]"""
                                            
                                            with st.spinner(f"Đang lấy gợi ý AI cho sản phẩm {prod_id}..."):
                                                try:
                                                    model_gen_prod = genai.GenerativeModel('gemini-1.5-flash')
                                                    response_gen_prod = model_gen_prod.generate_content(prompt_prod_summary)
                                                    prod_suggestions_text = response_gen_prod.text.strip()
                                                    if prod_suggestions_text:
                                                        # Xử lý để hiển thị đẹp hơn
                                                        prod_suggestions_list = [s.strip().lstrip("-* ") for s in prod_suggestions_text.split('\n') if s.strip()]
                                                        for sugg_item in prod_suggestions_list:
                                                            st.markdown(f"- {sugg_item}")
                                                    else:
                                                        st.info("AI không đưa ra gợi ý cụ thể.")
                                                except Exception as gemini_e_prod:
                                                    st.warning(f"Lỗi gọi AI cho SP {prod_id}: {gemini_e_prod}")
                                        else:
                                            st.info("Gemini chưa được cấu hình để đưa ra gợi ý AI.")
                                    else:
                                        st.info("Không có dữ liệu cảm xúc hợp lệ cho sản phẩm này để tạo gợi ý.")
                    
                    # --- Nút Tải xuống ---
                    st.markdown("---")
                    st.subheader("💾 Tải xuống Kết quả Phân tích Hàng loạt")
                    
                    # Sử dụng @st.cache_data cho hàm convert_df
                    @st.cache_data
                    def convert_df_to_csv_sig(df_to_convert):
                        # Chọn các cột cần xuất và đúng thứ tự mong muốn
                        cols_to_export = [
                            "original_comment", "product_id", "sentiment", "confidence", 
                            "source", "kb_has_ai_details", "status", "would_call_ai", "processing_timestamp"
                        ]
                        # Lấy các cột thực sự tồn tại trong df_to_convert để tránh KeyError
                        existing_cols = [col for col in cols_to_export if col in df_to_convert.columns]
                        try:
                            return df_to_convert[existing_cols].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                        except Exception as e:
                            st.error(f"Lỗi khi chuyển đổi DataFrame sang CSV: {e}")
                            return None

                    csv_data_to_download = convert_df_to_csv_sig(results_df_batch)
                    
                    if csv_data_to_download:
                        st.download_button(
                            label="📥 Tải Kết quả (CSV)",
                            data=csv_data_to_download, 
                            file_name=f'ket_qua_phan_tich_cam_xuc_{uploaded_file_batch.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv'
                        )
                    else:
                        st.error("Không thể tạo file CSV để tải xuống.")
                else:
                    st.warning("Không có dòng nào được xử lý, không có kết quả để hiển thị hoặc tải xuống.")
        except Exception as e:
            st.error(f"⚠️ Đã xảy ra lỗi không mong muốn trong quá trình xử lý file: {e}")
            traceback.print_exc()

# --- Tab 3: Thông tin Model (Giữ nguyên) ---
with tab3:
    st.header("Thông tin Đánh giá Model (XLM-RoBERTa)")
    st.markdown("Phần này hiển thị các số liệu đánh giá và biểu đồ liên quan đến hiệu suất của mô hình XLM-RoBERTa được sử dụng để phân tích cảm xúc.")
    
    # Cập nhật: Tải dữ liệu từ evaluation_summary.json và hiển thị
    eval_summary_path = os.path.join(config.VISUALIZATION_DIR, 'evaluation_summary.json')
    classification_report_path = os.path.join(config.VISUALIZATION_DIR, 'classification_report.txt')
    confusion_matrix_img_path = config.CONFUSION_MATRIX_FILE
    training_curves_img_path = config.TRAINING_CURVES_FILE

    if os.path.exists(eval_summary_path):
        try:
            with open(eval_summary_path, 'r', encoding='utf-8') as f:
                eval_summary = json.load(f)
            
            st.subheader("📈 Số liệu Hiệu suất Tổng thể")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Test Accuracy", f"{eval_summary.get('test_accuracy', 0)*100:.2f}%" if eval_summary.get('test_accuracy') is not None else "N/A")
            with col_m2:
                st.metric("Weighted F1-Score", f"{eval_summary.get('weighted_f1', 0):.4f}" if eval_summary.get('weighted_f1') is not None else "N/A")
            with col_m3:
                st.metric("Macro F1-Score", f"{eval_summary.get('macro_f1', 0):.4f}" if eval_summary.get('macro_f1') is not None else "N/A")
            with col_m4:
                st.metric("Test Loss", f"{eval_summary.get('test_loss', 0):.4f}" if eval_summary.get('test_loss') is not None else "N/A")

            if 'classification_report_dict' in eval_summary:
                st.subheader("📊 Báo cáo Phân loại Chi tiết (Test Set)")
                # Hiển thị đẹp hơn classification report dạng dict
                report_df_data = []
                for label, metrics in eval_summary['classification_report_dict'].items():
                    if isinstance(metrics, dict):
                        report_df_data.append({
                            'Cảm xúc': label.capitalize(),
                            'Precision': f"{metrics.get('precision',0):.4f}",
                            'Recall': f"{metrics.get('recall',0):.4f}",
                            'F1-Score': f"{metrics.get('f1-score',0):.4f}",
                            'Support': metrics.get('support',0)
                        })
                if report_df_data:
                    report_display_df = pd.DataFrame(report_df_data)
                    st.dataframe(report_display_df.set_index('Cảm xúc'))
            elif os.path.exists(classification_report_path):
                 st.subheader("📊 Báo cáo Phân loại (Từ File)")
                 with open(classification_report_path, 'r', encoding='utf-8') as f:
                     st.text(f.read())
            
        except Exception as e:
            st.error(f"Lỗi khi tải hoặc hiển thị tóm tắt đánh giá: {e}")
    else:
        st.warning(f"Không tìm thấy file tóm tắt đánh giá tại: {eval_summary_path}. Vui lòng chạy 'evaluate.py' trước.")

    st.subheader("🖼️ Biểu đồ Hiệu suất")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists(confusion_matrix_img_path):
            st.image(confusion_matrix_img_path, caption="Ma trận Nhầm lẫn (Test Set)", use_column_width=True)
        else:
            st.caption(f"Không tìm thấy hình ảnh Ma trận nhầm lẫn. Chạy 'evaluate.py'. ({confusion_matrix_img_path})")
    
    with col_img2:
        if os.path.exists(training_curves_img_path):
            st.image(training_curves_img_path, caption="Biểu đồ Huấn luyện (Loss & Accuracy)", use_column_width=True)
        else:
            st.caption(f"Không tìm thấy hình ảnh Biểu đồ huấn luyện. Chạy 'evaluate.py'. ({training_curves_img_path})")
    
    # Hiển thị một vài ví dụ lỗi nếu có
    if os.path.exists(eval_summary_path) and eval_summary.get('error_samples_examples'):
        st.subheader("🧐 Ví dụ các Mẫu Dự đoán Sai (Test Set)")
        error_examples_df = pd.DataFrame(eval_summary['error_samples_examples'])
        st.dataframe(error_examples_df[['cleaned_text', 'true_label_name', 'predicted_label_name']].rename(columns={
            'cleaned_text': 'Bình luận đã làm sạch',
            'true_label_name': 'Nhãn thực tế',
            'predicted_label_name': 'Nhãn dự đoán'
        }))


# --- Footer ---
st.markdown("---")
st.caption("Dự án Thực tập - Xử lý Phản hồi Khách hàng - [Nguyễn Trần Hoàng Thịnh]")