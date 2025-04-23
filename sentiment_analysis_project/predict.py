# predict.py (Đã Việt hóa thông báo)

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import time
import traceback # Để in chi tiết lỗi

import config
from data_processing import clean_text

class SentimentPredictor:
    """Lớp để tải model đã huấn luyện và thực hiện dự đoán cảm xúc."""

    def __init__(self, model_path=config.MODEL_SAVE_PATH):
        """
        Khởi tạo predictor bằng cách tải model và tokenizer.

        Args:
            model_path (str): Đường dẫn đến thư mục chứa model và tokenizer đã lưu.
        """
        self.device = torch.device(config.DEVICE)
        print(f"Predictor đang sử dụng thiết bị: {self.device}")

        try:
            print(f"Đang tải model và tokenizer từ: {model_path}")
            if not os.path.isdir(model_path):
                 raise OSError(f"Không tìm thấy thư mục model tại {model_path}")

            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval() # Đặt model vào chế độ đánh giá
            print("Đã tải model và tokenizer thành công.")
            self.label_map = config.LABEL_MAP # Sử dụng bản đồ nhãn từ config
            self.max_len = config.MAX_LENGTH # Sử dụng độ dài tối đa từ config

        except OSError as e:
             print(f"Lỗi khi tải model/tokenizer từ {model_path}: {e}")
             print("Hãy đảm bảo model đã được huấn luyện và lưu đúng cách.")
             self.model = None
             self.tokenizer = None
        except Exception as e:
             print(f"Đã xảy ra lỗi không mong muốn trong quá trình khởi tạo: {e}")
             self.model = None
             self.tokenizer = None

    def predict_single(self, text):
        """
        Dự đoán cảm xúc cho một chuỗi văn bản đơn lẻ.

        Args:
            text (str): Văn bản đầu vào cần phân tích.

        Returns:
            tuple: (predicted_label, confidence_score, probabilities_dict)
                   - predicted_label (str): "Tích cực", "Tiêu cực", hoặc "Trung tính".
                   - confidence_score (float): Điểm tin cậy của nhãn dự đoán.
                   - probabilities_dict (dict): Dictionary ánh xạ tên nhãn với xác suất của nó.
                   Trả về (None, None, None) nếu khởi tạo thất bại hoặc có lỗi dự đoán.
        """
        if not self.model or not self.tokenizer:
            print("Lỗi: Model hoặc tokenizer chưa được tải. Không thể dự đoán.")
            return None, None, None

        if not text or not isinstance(text, str):
            # print("Lỗi: Văn bản đầu vào phải là một chuỗi không rỗng.")
            # Trả về None để báo hiệu input không hợp lệ
            return None, None, None

        try:
            cleaned_text = clean_text(text)
            if not cleaned_text:
                # print("Cảnh báo: Văn bản trở nên rỗng sau khi làm sạch.")
                # Trả về dự đoán Trung tính mặc định cho chuỗi rỗng
                neutral_label_index = next((k for k, v in self.label_map.items() if v == "Trung tính"), 1)
                default_probs = {label: 1.0/len(self.label_map) for label in self.label_map.values()}
                return self.label_map.get(neutral_label_index, "Trung tính"), 1.0/len(self.label_map), default_probs

            encoding = self.tokenizer.encode_plus(
                cleaned_text, add_special_tokens=True, max_length=self.max_len,
                return_token_type_ids=False, padding='max_length', truncation=True,
                return_attention_mask=True, return_tensors='pt',
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            probabilities = F.softmax(logits, dim=1).squeeze()
            confidence, predicted_idx_tensor = torch.max(probabilities, dim=0)
            predicted_idx = predicted_idx_tensor.item()
            confidence_score = confidence.item()

            predicted_label = self.label_map.get(predicted_idx, "Không xác định")

            probs_list = probabilities.cpu().numpy().tolist()
            probabilities_dict = {self.label_map.get(i, f"Không_xác_định_{i}"): prob for i, prob in enumerate(probs_list)}

            return predicted_label, confidence_score, probabilities_dict

        except Exception as e:
            print(f"Đã xảy ra lỗi trong quá trình dự đoán đơn lẻ cho văn bản '{text[:50]}...': {e}")
            print(traceback.format_exc()) # In chi tiết lỗi
            return None, None, None

    def predict_batch_df(self, df, text_column):
        """
        Thêm các cột dự đoán ('predicted_label', 'confidence') vào DataFrame.

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
            text_column (str): Tên cột chứa văn bản cần phân tích.

        Returns:
            pd.DataFrame: DataFrame gốc với các cột dự đoán được thêm vào.
                          Trả về None nếu có lỗi.
        """
        if not self.model or not self.tokenizer:
            print("Lỗi: Model hoặc tokenizer chưa được tải. Không thể dự đoán.")
            return None
        if df is None or df.empty:
            print("Lỗi: DataFrame đầu vào rỗng hoặc None.")
            return None
        if text_column not in df.columns:
            print(f"Lỗi: Không tìm thấy cột văn bản '{text_column}' trong DataFrame.")
            return None

        print(f"Đang dự đoán cảm xúc cho {len(df)} văn bản từ cột '{text_column}'...")
        results = []
        start_time = time.time()

        for index, row in df.iterrows():
            text = row[text_column]
            if pd.isna(text) or not isinstance(text, str):
                 label, confidence, _ = None, None, None
            else:
                 label, confidence, _ = self.predict_single(str(text))

            results.append({
                'predicted_label': label if label is not None else "Lỗi Dự đoán",
                'confidence': confidence if confidence is not None else 0.0
            })

            if (index + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Đã xử lý {index + 1}/{len(df)} dòng... ({elapsed:.2f} giây)")

        end_time = time.time()
        print(f"Dự đoán hàng loạt hoàn thành sau {end_time - start_time:.2f} giây")

        try:
             results_df = pd.DataFrame(results, index=df.index)
             df_out = pd.concat([df, results_df], axis=1)
             return df_out
        except Exception as e:
             print(f"Lỗi khi ghép kết quả vào DataFrame: {e}")
             return None

# --- Phần Test (chỉ chạy khi thực thi file predict.py trực tiếp) ---
if __name__ == '__main__':
    print("--- Kiểm tra Sentiment Predictor (Chạy trực tiếp predict.py) ---")
    predictor_test_instance = SentimentPredictor()

    if predictor_test_instance.model:
        print("\n--- Kiểm tra Dự đoán Đơn lẻ ---")
        test_texts = [
            "Sản phẩm này thật tuyệt vời! Tôi yêu nó.",
            "Bộ phim cũng được, không hay nhưng cũng không tệ.",
            "Rất thất vọng về chất lượng, nó hỏng sau một lần dùng.",
            "Dịch vụ khách hàng hoàn toàn vô dụng.",
            "",
            None,
            123
        ]
        for text in test_texts:
            print(f"\nĐầu vào: '{text}'")
            label, conf, probs = predictor_test_instance.predict_single(text)
            if label is not None:
                print(f"Đầu ra -> Nhãn: {label}, Độ tin cậy: {conf:.4f}")
                print(f"Xác suất: {probs}")
            else:
                print("Dự đoán thất bại hoặc đầu vào không hợp lệ.")

        print("\n--- Kiểm tra Dự đoán Hàng loạt (DataFrame) ---")
        data = {'review_text': ["Dịch vụ tuyệt vời!", "Đồ ăn tạm được.", None, "Tôi sẽ không bao giờ quay lại đây.", "Cũng ổn.", ""],
                'other_col': [1, 2, 3, 4, 5, 6]}
        test_df = pd.DataFrame(data)
        print("\nDataFrame Đầu vào:")
        print(test_df)

        results_df = predictor_test_instance.predict_batch_df(test_df, text_column='review_text')

        if results_df is not None:
            print("\nDataFrame Đầu ra với Dự đoán:")
            print(results_df)
        else:
            print("Dự đoán hàng loạt thất bại.")
    else:
        print("Khởi tạo Predictor thất bại. Không thể chạy kiểm tra.")