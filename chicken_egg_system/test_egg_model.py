import os
import sys
import glob
import random
import cv2
from ultralytics import YOLO

def get_random_test_image(base_dir):
    """Tìm một ảnh ngẫu nhiên trong thư mục testing để chạy thử."""
    search_path = os.path.join(base_dir, "archive", "testing", "**", "*.jpg")
    images = glob.glob(search_path, recursive=True)
    if not images:
        search_path = os.path.join(base_dir, "archive", "testing", "**", "*.png")
        images = glob.glob(search_path, recursive=True)
    return random.choice(images) if images else None

def main():
    # Thư mục gốc của project chicken_egg_system
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Kiểm tra xem có cờ hiển thị giao diện không
    show_gui = "--show" in sys.argv
    if show_gui:
        sys.argv.remove("--show")
        
    # 1. Đường dẫn tới mô hình
    # Mặc định sử dụng best.pt, nếu có file ONNX có thể dùng thử best_int8.onnx
    model_path = os.path.join(base_dir, "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "best_int8.onnx")
        
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model tại: {model_path}")
        print("Vui lòng đảm bảo file 'best.pt' hoặc 'best_int8.onnx' nằm trong cùng thư mục với script này.")
        sys.exit(1)
        
    print(f"🔄 Đang tải mô hình YOLO từ: {model_path}...")
    model = YOLO(model_path)
    
    # 2. Xác định ảnh đầu vào
    img_path = None
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"❌ File ảnh không tồn tại: {img_path}")
            sys.exit(1)
    else:
        print("💡 Không truyền đường dẫn ảnh. Đang tìm một ảnh ngẫu nhiên trong thư mục dataset thử nghiệm...")
        img_path = get_random_test_image(base_dir)
        if not img_path:
            print("❌ Không tìm thấy ảnh nào trong thư mục 'archive/testing/'.")
            print("Vui lòng truyền đường dẫn ảnh cụ thể: python test_egg_model.py <path_to_image>")
            sys.exit(1)
            
    print(f"📸 Ảnh đầu vào: {img_path}")
    
    # Lấy tên lớp gốc (nhãn thực tế) từ tên file hoặc thư mục chứa ảnh
    filename = os.path.basename(img_path)
    parent_dir = os.path.basename(os.path.dirname(img_path))
    
    # Hỗ trợ cả định dạng tên file của Edge Impulse (ví dụ: fertile.fertile872.jpg)
    parts = filename.split(".")
    if len(parts) > 1 and parts[0] in ["fertile", "infertile", "dead"]:
        actual_label = parts[0]
    else:
        actual_label = parent_dir
        
    print(f"🏷️ Nhãn thực tế: {actual_label.upper()}")
    
    # 3. Chạy dự đoán
    results = model(img_path)
    
    # 4. Xử lý kết quả dự đoán
    for r in results:
        # Tải lại ảnh bằng OpenCV để vẽ kết quả lên đó
        img = cv2.imread(img_path)
        
        # Lấy xác suất của các lớp
        probs = r.probs
        top1_idx = probs.top1
        top1_label = r.names[top1_idx]
        top1_conf = probs.top1conf.item()
        
        print("\n Kết quả chi tiết từ Model:")
        for idx, label in r.names.items():
            conf = probs.data[idx].item()
            print(f"  - {label}: {conf:.2%}")
            
        print(f"\n Dự đoán kết quả cao nhất: {top1_label.upper()} với độ tin cậy {top1_conf:.2%}")
        
        # Vẽ nhãn lên ảnh
        label_text = f"Pred: {top1_label.upper()} ({top1_conf:.1%})"
        actual_text = f"Actual: {actual_label.upper()}"
        
        # Cấu hình font và màu sắc vẽ nhãn (màu xanh lá nếu đúng, màu đỏ nếu sai)
        is_correct = top1_label.lower() == actual_label.lower()
        color = (0, 255, 0) if is_correct else (0, 0, 255) # BGR
        
        # Vẽ chữ lên ảnh
        cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, actual_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 5. Lưu ảnh kết quả
        output_path = os.path.join(base_dir, "result_output.jpg")
        cv2.imwrite(output_path, img)
        print(f" Đã lưu kết quả trực quan hóa tại: {output_path}")
        
        # 6. Hiển thị ảnh nếu được yêu cầu và hệ thống hỗ trợ GUI
        if show_gui:
            try:
                cv2.imshow("Egg Quality Classification Test", img)
                print(" Đang mở cửa sổ hiển thị ảnh. Nhấn phím bất kỳ trên cửa sổ ảnh để đóng...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                print(" Môi trường không hỗ trợ hiển thị giao diện GUI trực tiếp. Hãy mở file ảnh đã lưu để xem.")
        else:
            print(" Thêm tham số '--show' nếu bạn muốn mở cửa sổ hiển thị ảnh trực tiếp (ví dụ: python test_egg_model.py --show).")

if __name__ == "__main__":
    main()
