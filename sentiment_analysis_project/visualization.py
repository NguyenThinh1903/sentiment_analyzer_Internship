import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

import config 

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Vẽ ma trận nhầm lẫn sử dụng Seaborn.

    Args:
        cm (numpy.ndarray): Mảng ma trận nhầm lẫn.
        class_names (list): Danh sách tên các lớp cho nhãn (đã Việt hóa).
        save_path (str, optional): Đường dẫn để lưu hình ảnh biểu đồ.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    try:
         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                     xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
    except Exception as e:
        print(f"Lỗi khi vẽ heatmap: {e}. Kiểm tra dữ liệu đầu vào.")
        plt.close(fig)
        return 

    ax.set_xlabel('Nhãn Dự đoán', fontsize=12) 
    ax.set_ylabel('Nhãn Thực tế', fontsize=12)  
    ax.set_title('Ma trận Nhầm lẫn', fontsize=14, fontweight='bold') 
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10, rotation=0) 

    plt.tight_layout()
    if save_path:
        try:
             os.makedirs(os.path.dirname(save_path), exist_ok=True)
             plt.savefig(save_path, dpi=300, bbox_inches='tight')
             print(f"Ma trận nhầm lẫn đã được lưu vào {save_path}")
        except Exception as e:
            print(f"Lỗi khi lưu ma trận nhầm lẫn: {e}")
    else:
        print("Cảnh báo: Không có đường dẫn lưu, ma trận nhầm lẫn sẽ không được lưu.")
    plt.close(fig) 

def plot_training_history(history_path, save_path=None):
    """
    Vẽ biểu đồ loss và accuracy huấn luyện/kiểm định từ file lịch sử.

    Args:
        history_path (str): Đường dẫn đến file JSON chứa lịch sử huấn luyện.
        save_path (str, optional): Đường dẫn để lưu hình ảnh biểu đồ.
    """
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file lịch sử huấn luyện tại {history_path}")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: Không thể giải mã JSON từ {history_path}")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải lịch sử huấn luyện: {e}")
        return

    required_keys = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    if not all(key in history for key in required_keys):
        print(f"Lỗi: File lịch sử {history_path} thiếu key cần thiết: {required_keys}")
        return
    if not history['train_loss'] or not history['train_acc'] or not history['val_loss'] or not history['val_acc']:
        print(f"Cảnh báo: Dữ liệu lịch sử trong {history_path} bị rỗng hoặc thiếu.")
        return

    epochs = range(1, len(history['train_loss']) + 1)
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Biểu đồ Quá trình Huấn luyện', fontsize=16, fontweight='bold')

    # Vẽ Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Loss Huấn luyện', markersize=5) 
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Loss Kiểm định', markersize=5)   
    ax1.set_title('Biểu đồ Loss', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Vẽ Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Accuracy Huấn luyện', markersize=5) 
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Accuracy Kiểm định', markersize=5)  
    ax2.set_title('Biểu đồ Accuracy', fontsize=14) 
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y))) 

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Biểu đồ lịch sử huấn luyện đã được lưu vào {save_path}")
        except Exception as e:
            print(f"Lỗi khi lưu biểu đồ lịch sử huấn luyện: {e}")
    else:
        print("Cảnh báo: Không có đường dẫn lưu, biểu đồ huấn luyện sẽ không được lưu.")
    plt.close(fig)


if __name__ == '__main__':
    print("File visualization.py chứa các hàm vẽ biểu đồ.")
