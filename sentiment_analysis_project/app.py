# app.py (Đã Việt hóa giao diện người dùng)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import time

import config
from predict import SentimentPredictor
# Giả sử các file ảnh đã được tạo bởi evaluate.py
# from visualization import plot_confusion_matrix

# --- Cấu hình Trang ---
st.set_page_config(
    page_title="Phân tích Cảm xúc Khách hàng",
    page_icon="😊",
    layout="wide"
)

# --- Tải Predictor (Sử dụng cache) ---
@st.cache_resource # Cache việc tải model
def load_predictor(model_path=config.MODEL_SAVE_PATH):
    """Tải instance của SentimentPredictor."""
    print("Đang thử tải Sentiment Predictor...")
    predictor = SentimentPredictor(model_path=model_path)
    if not predictor.model or not predictor.tokenizer:
        # Hiển thị lỗi ngay trên UI nếu không tải được model
        st.error(f"Lỗi nghiêm trọng: Không thể tải model từ '{model_path}'. Đảm bảo model đã được huấn luyện và lưu đúng chỗ.")
        return None # Trả về None để báo hiệu lỗi
    print("Sentiment Predictor đã được tải thành công.")
    return predictor

predictor = load_predictor()

# --- Hàm trợ giúp ---
def display_probabilities_pie(probabilities_dict):
    """Hiển thị biểu đồ tròn thể hiện xác suất các cảm xúc."""
    if probabilities_dict:
        # Lấy nhãn và giá trị từ dict, đảm bảo đúng thứ tự nếu cần
        labels = list(probabilities_dict.keys())
        values = list(probabilities_dict.values())
        # Sắp xếp theo thứ tự mong muốn (ví dụ: Tiêu cực, Trung tính, Tích cực)
        sorted_labels = ["Tiêu cực", "Trung tính", "Tích cực"]
        try:
            # Cố gắng sắp xếp theo thứ tự trên, bỏ qua nếu nhãn không tồn tại
            label_map_inv = {v: k for k, v in config.LABEL_MAP.items()} # Map ngược để lấy index
            values_sorted = sorted(zip(labels, values), key=lambda item: label_map_inv.get(item[0], 99)) # Sắp xếp theo index, nhãn lạ cuối cùng
            labels_sorted = [item[0] for item in values_sorted]
            values_final = [item[1] for item in values_sorted]
            labels_final = labels_sorted
        except Exception: # Nếu có lỗi sắp xếp, dùng thứ tự gốc
             labels_final = labels
             values_final = values

        # Định nghĩa màu sắc tương ứng
        color_map = {"Tiêu cực": '#DC143C', "Trung tính": '#FFD700', "Tích cực": '#32CD32', "Không xác định": '#808080'}
        colors = [color_map.get(label, '#808080') for label in labels_final]


        fig = go.Figure(data=[go.Pie(labels=labels_final, values=values_final, hole=.3,
                                     marker_colors=colors,
                                     pull=[0.05 if v == max(values_final) else 0 for v in values_final] # Kéo miếng lớn nhất
                                     )])
        fig.update_layout(
            title_text='Phân bổ Xác suất Cảm xúc',
            legend_title_text='Cảm xúc',
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Không có dữ liệu xác suất để hiển thị.")

# --- Giao diện Chính của Ứng dụng ---
st.title("📊 Web App Phân Tích Cảm Xúc Phản Hồi Khách Hàng")
st.markdown("""
Chào mừng bạn đến với ứng dụng phân tích cảm xúc!
Ứng dụng này sử dụng mô hình học sâu (Deep Learning) dựa trên *Transformers* để dự đoán cảm xúc
(**Tích cực**, **Tiêu cực**, **Trung tính**) từ văn bản phản hồi của khách hàng.
""")

# --- Kiểm tra Model đã tải được chưa ---
if predictor is None:
    st.warning("Model chưa sẵn sàng. Vui lòng kiểm tra lỗi ở trên hoặc đợi quá trình tải hoàn tất.")
    st.stop() # Dừng thực thi nếu model không tải được

# --- Các Tab chức năng ---
tab1, tab2, tab3 = st.tabs(["🔍 Phân tích Văn bản Đơn lẻ", "📄 Phân tích File CSV", "📈 Đánh giá Model"])

# --- Tab 1: Phân tích Đơn lẻ ---
with tab1:
    st.header("Nhập phản hồi cần phân tích:")
    user_input = st.text_area("Nhập văn bản vào đây...", height=150, key="single_text_input", placeholder="Ví dụ: Chất lượng sản phẩm rất tốt, tôi rất hài lòng!")

    if st.button("🚀 Phân tích Ngay!", key="analyze_single"):
        if user_input and user_input.strip():
            start_time = time.time()
            # Hiển thị spinner trong khi chờ dự đoán
            with st.spinner('🧠 Đang phân tích, vui lòng chờ...'):
                label, confidence, probabilities = predictor.predict_single(user_input)
            end_time = time.time()

            if label is not None:
                st.subheader("Kết quả Phân tích:")
                col1, col2 = st.columns([1, 2]) # Chia cột để hiển thị gọn hơn
                with col1:
                    # Hiển thị nhãn với màu sắc tương ứng
                    if label == config.LABEL_MAP[2]: # Tích cực
                        st.success(f"**Cảm xúc:** {label}")
                    elif label == config.LABEL_MAP[0]: # Tiêu cực
                        st.error(f"**Cảm xúc:** {label}")
                    else: # Trung tính hoặc khác
                        st.warning(f"**Cảm xúc:** {label}")
                    # Hiển thị độ tin cậy dạng %
                    st.metric(label="Độ tin cậy", value=f"{confidence:.2%}")
                    st.caption(f"Thời gian xử lý: {end_time - start_time:.2f} giây")

                with col2:
                    # Hiển thị biểu đồ tròn xác suất
                    display_probabilities_pie(probabilities)
            else:
                st.error("⚠️ Có lỗi xảy ra trong quá trình dự đoán hoặc đầu vào không hợp lệ. Vui lòng thử lại.")
        else:
            st.warning("⚠️ Vui lòng nhập văn bản để phân tích.")

# --- Tab 2: Phân tích Hàng loạt (CSV) ---
with tab2:
    st.header("Tải lên file CSV để phân tích hàng loạt:")
    uploaded_file = st.file_uploader(
        f"Chọn file CSV (phải có cột tên là '{config.TEXT_COLUMN}')",
        type=["csv"],
        key="csv_uploader",
        help=f"File CSV của bạn cần có ít nhất một cột chứa văn bản phản hồi. Hãy đảm bảo tên cột đó là '{config.TEXT_COLUMN}' như đã cấu hình."
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Đang đọc file CSV..."):
                # Cố gắng đọc với encoding utf-8-sig để xử lý BOM nếu có
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    st.warning("Không thể đọc bằng UTF-8-SIG, thử UTF-8...")
                    df = pd.read_csv(uploaded_file, encoding='utf-8')

            st.success(f"✅ Đã tải lên file '{uploaded_file.name}' với {len(df)} dòng.")

            # Kiểm tra xem cột text có tồn tại không
            if config.TEXT_COLUMN not in df.columns:
                st.error(f"Lỗi: Không tìm thấy cột '{config.TEXT_COLUMN}' trong file CSV đã tải lên.")
                st.info(f"Vui lòng đảm bảo file CSV của bạn có cột tên chính xác là '{config.TEXT_COLUMN}'.")
            else:
                # Hiển thị bản xem trước
                st.write("Xem trước dữ liệu (5 dòng đầu):")
                st.dataframe(df.head(), use_container_width=True)

                if st.button("📊 Phân tích File CSV", key="analyze_csv"):
                    start_time = time.time()
                    progress_bar = st.progress(0, text="Bắt đầu phân tích...") # Thêm thanh tiến trình
                    status_text = st.empty() # Vị trí để cập nhật trạng thái

                    # Hàm callback để cập nhật tiến trình (ví dụ, nếu predict_batch_df hỗ trợ)
                    # Hiện tại, chúng ta sẽ mô phỏng tiến trình
                    results_df = None
                    total_rows = len(df)
                    try:
                        # --- Bắt đầu dự đoán ---
                        # Lưu ý: predict_batch_df trong ví dụ hiện tại xử lý từng dòng,
                        # nên việc cập nhật progress bar chính xác cần sửa đổi hàm đó
                        # Ở đây, chúng ta chỉ hiển thị spinner và thông báo chung
                        with st.spinner(f"⏳ Đang phân tích cột '{config.TEXT_COLUMN}'... Quá trình này có thể mất vài phút."):
                             results_df = predictor.predict_batch_df(df.copy(), config.TEXT_COLUMN)
                             # Giả lập hoàn thành progress bar sau khi xong
                             progress_bar.progress(100, text="Phân tích hoàn tất!")

                    except Exception as batch_error:
                         st.error(f"Lỗi nghiêm trọng trong quá trình phân tích hàng loạt: {batch_error}")
                         progress_bar.progress(100, text="Phân tích thất bại!") # Cập nhật thanh tiến trình khi lỗi


                    end_time = time.time()

                    if results_df is not None:
                        st.success(f"✅ Phân tích hoàn tất sau {end_time - start_time:.2f} giây!")

                        # Hiển thị Thống kê Tổng hợp
                        st.subheader("Thống kê Cảm xúc Tổng hợp:")
                        # Đảm bảo xử lý trường hợp cột dự đoán không tồn tại hoặc rỗng
                        if 'predicted_label' in results_df.columns and not results_df['predicted_label'].empty:
                            sentiment_counts = results_df['predicted_label'].value_counts()
                            # Đảm bảo dùng đúng tên nhãn từ config
                            valid_labels = list(config.LABEL_MAP.values()) + ["Lỗi Dự đoán"] # Bao gồm cả nhãn lỗi
                            sentiment_counts = sentiment_counts.reindex(valid_labels, fill_value=0) # Đảm bảo đủ 3 nhãn + lỗi

                            color_map_stats = {"Tiêu cực": '#DC143C', "Trung tính": '#FFD700', "Tích cực": '#32CD32', "Lỗi Dự đoán": '#808080'}
                            colors_stats = [color_map_stats.get(label, '#808080') for label in sentiment_counts.index]


                            fig_bar = px.bar(
                                sentiment_counts,
                                x=sentiment_counts.index,
                                y=sentiment_counts.values,
                                labels={'x': 'Cảm xúc', 'y': 'Số lượng'},
                                title='Phân phối Số lượng Cảm xúc',
                                color=sentiment_counts.index,
                                color_discrete_map=color_map_stats,
                                text=sentiment_counts.values
                            )
                            fig_bar.update_layout(showlegend=False)

                            # Chỉ vẽ pie chart nếu có dữ liệu hợp lệ (không chỉ có lỗi)
                            valid_counts = sentiment_counts.drop("Lỗi Dự đoán", errors='ignore') # Bỏ qua nhãn lỗi
                            if valid_counts.sum() > 0:
                                fig_pie = go.Figure(data=[go.Pie(
                                    labels=valid_counts.index,
                                    values=valid_counts.values,
                                    hole=.3,
                                    marker_colors=[color_map_stats.get(label, '#808080') for label in valid_counts.index],
                                )])
                                fig_pie.update_layout(title_text='Tỷ lệ Phần trăm Cảm xúc (Không tính lỗi)')
                            else:
                                fig_pie = None # Không vẽ pie nếu toàn lỗi

                            col_stats1, col_stats2 = st.columns(2)
                            with col_stats1:
                                st.plotly_chart(fig_bar, use_container_width=True)
                            with col_stats2:
                                if fig_pie:
                                     st.plotly_chart(fig_pie, use_container_width=True)
                                else:
                                     st.info("Không có dữ liệu cảm xúc hợp lệ để vẽ biểu đồ tròn.")
                        else:
                             st.warning("Không tìm thấy cột 'predicted_label' hoặc không có kết quả để thống kê.")

                        # Hiển thị Kết quả Chi tiết (có thể phân trang hoặc giới hạn nếu cần)
                        st.subheader("Kết quả Chi tiết:")
                        # Tùy chọn: Giới hạn số dòng hiển thị ban đầu
                        # st.dataframe(results_df.head(100), use_container_width=True)
                        # if len(results_df) > 100:
                        #    st.caption(f"Hiển thị 100/{len(results_df)} dòng đầu tiên.")
                        st.dataframe(results_df, use_container_width=True)

                        # Thêm nút tải xuống
                        @st.cache_data # Cache việc chuyển đổi DF sang CSV
                        def convert_df_to_csv(df_to_convert):
                            try:
                                # Sử dụng encoding utf-8-sig để Excel đọc tiếng Việt tốt hơn
                                return df_to_convert.to_csv(index=False).encode('utf-8-sig')
                            except Exception as e:
                                print(f"Lỗi khi chuyển đổi DataFrame sang CSV: {e}")
                                return None

                        csv_output = convert_df_to_csv(results_df)
                        if csv_output:
                            st.download_button(
                                label="📥 Tải xuống Kết quả (CSV)",
                                data=csv_output,
                                file_name=f'phan_tich_cam_xuc_{uploaded_file.name}.csv', # Tên file tiếng Việt
                                mime='text/csv',
                            )
                        else:
                            st.error("Không thể tạo file CSV để tải xuống.")

                    # Không cần else ở đây vì lỗi đã được xử lý trong khối try-except predict_batch_df

        except UnicodeDecodeError:
            st.error("Lỗi: Không thể đọc file CSV. File có thể không được mã hóa đúng dạng UTF-8. Vui lòng kiểm tra và lưu lại file với mã hóa UTF-8.")
        except pd.errors.EmptyDataError:
             st.error("Lỗi: File CSV bị rỗng hoặc không có dữ liệu.")
        except Exception as e:
            st.error(f"⚠️ Lỗi không xác định khi xử lý file CSV: {e}")
            st.warning("Hãy đảm bảo file CSV của bạn hợp lệ.")


# --- Tab 3: Đánh giá Model ---
with tab3:
    st.header("Thông tin Đánh giá Model")
    st.markdown("Kết quả đánh giá hiệu năng của model trên tập dữ liệu kiểm thử (test set):")

    # Tải và hiển thị các chỉ số từ file báo cáo
    report_path = config.CLASSIFICATION_REPORT_FILE
    cm_path = config.CONFUSION_MATRIX_FILE
    curves_path = config.TRAINING_CURVES_FILE

    if os.path.exists(report_path):
        try:
            with open(report_path, 'r', encoding='utf-8') as f: # Thêm encoding='utf-8'
                lines = f.readlines()
                accuracy_line = next((line for line in lines if "Độ chính xác trên tập Test:" in line), None) # Tìm dòng accuracy tiếng Việt
                accuracy = float(accuracy_line.split(":")[1].strip()) if accuracy_line else None

                report_content = "".join(lines) # Lấy toàn bộ nội dung báo cáo

            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                 if accuracy is not None:
                    # Hiển thị accuracy dưới dạng metric
                    st.metric("Accuracy Tổng thể (trên tập Test)", f"{accuracy:.2%}")
                 else:
                     st.info("Không tìm thấy thông tin Accuracy trong file báo cáo.")
            # Bạn có thể thêm các metric khác nếu parse được từ report (ví dụ F1-score)
            # with col_metric2:
            #    st.metric("F1-score (Weighted - nếu có)", "...")

            st.subheader("Báo cáo Phân loại Chi tiết:")
            st.text(report_content) # Hiển thị nội dung file report

        except FileNotFoundError:
             st.warning(f"Không tìm thấy file báo cáo phân loại tại: {report_path}")
        except Exception as e:
            st.warning(f"Không thể đọc hoặc phân tích file báo cáo ({report_path}): {e}")
    else:
        st.info(f"Chưa có file đánh giá ({report_path}). Hãy chạy script `python evaluate.py` trước.")

    # Hiển thị Ma trận Nhầm lẫn
    st.subheader("Ma trận Nhầm lẫn (Confusion Matrix):")
    if os.path.exists(cm_path):
        try:
             st.image(cm_path, caption="Ma trận Nhầm lẫn trên tập Test")
        except Exception as e:
             st.warning(f"Không thể tải ảnh ma trận nhầm lẫn ({cm_path}): {e}")
    else:
        st.info(f"Chưa có ảnh ma trận nhầm lẫn ({cm_path}). Hãy chạy script `python evaluate.py`.")

    # Hiển thị Biểu đồ Huấn luyện
    st.subheader("Biểu đồ Quá trình Huấn luyện:")
    if os.path.exists(curves_path):
         try:
            st.image(curves_path, caption="Biểu đồ Loss và Accuracy trong quá trình Huấn luyện/Kiểm định")
         except Exception as e:
             st.warning(f"Không thể tải ảnh biểu đồ huấn luyện ({curves_path}): {e}")
    else:
        st.info(f"Chưa có ảnh biểu đồ huấn luyện ({curves_path}). Hãy chạy script `train.py` và `evaluate.py`.")


# --- Footer ---
st.markdown("---")
st.caption("Dự án Thực tập 8 Tuần - Phân tích Cảm xúc - Xây dựng bởi Nguyễn Trần Hoàng Thịnh")