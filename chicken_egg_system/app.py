import os
import sys
import glob
import random
import base64
import cv2
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__)

# Thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tải mô hình YOLO
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "best_int8.onnx")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Không tìm thấy file model tại: {MODEL_PATH}")
    sys.exit(1)

print(f"🔄 Đang tải mô hình YOLO từ: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("✅ Tải mô hình thành công!")

def get_actual_label(filepath):
    """Lấy nhãn thực tế từ tên file hoặc thư mục."""
    filename = os.path.basename(filepath)
    parent_dir = os.path.basename(os.path.dirname(filepath))
    
    parts = filename.split(".")
    if len(parts) > 1 and parts[0] in ["fertile", "infertile", "dead"]:
        return parts[0]
    return parent_dir

def image_to_base64(image_path_or_ndarray):
    """Chuyển ảnh sang dạng Base64 để hiển thị trực tiếp trên trình duyệt."""
    if isinstance(image_path_or_ndarray, str):
        with open(image_path_or_ndarray, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded
    else:
        # Nếu là mảng numpy (OpenCV image)
        _, buffer = cv2.imencode('.jpg', image_path_or_ndarray)
        encoded = base64.b64encode(buffer).decode("utf-8")
        return encoded

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/random-egg", methods=["GET"])
def random_egg():
    """Lấy một ảnh trứng ngẫu nhiên từ kho test và trả về thông tin + ảnh base64."""
    search_path = os.path.join(BASE_DIR, "archive", "testing", "**", "*.jpg")
    images = glob.glob(search_path, recursive=True)
    if not images:
        search_path = os.path.join(BASE_DIR, "archive", "testing", "**", "*.png")
        images = glob.glob(search_path, recursive=True)
        
    if not images:
        return jsonify({"error": "Không tìm thấy ảnh mẫu nào trong thư mục archive/testing/"}), 404
        
    img_path = random.choice(images)
    actual_label = get_actual_label(img_path)
    
    # Đọc ảnh để trả về ảnh gốc base64
    img_base64 = image_to_base64(img_path)
    
    return jsonify({
        "filepath": img_path,
        "filename": os.path.basename(img_path),
        "actual_label": actual_label,
        "image_base64": img_base64
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """Chạy dự đoán YOLO trên ảnh chỉ định và trả về biểu đồ xác suất + ảnh kết quả."""
    data = request.json
    img_path = data.get("filepath")
    
    if not img_path or not os.path.exists(img_path):
        return jsonify({"error": f"File ảnh không tồn tại: {img_path}"}), 400
        
    actual_label = get_actual_label(img_path)
    
    # Chạy YOLO
    results = model(img_path)
    r = results[0]
    
    probs = r.probs
    top1_idx = probs.top1
    top1_label = r.names[top1_idx]
    top1_conf = probs.top1conf.item()
    
    # Thống kê xác suất chi tiết
    detailed_probs = {}
    for idx, label in r.names.items():
        detailed_probs[label] = float(probs.data[idx].item())
        
    # Đọc ảnh và vẽ nhãn bằng OpenCV
    img = cv2.imread(img_path)
    label_text = f"Pred: {top1_label.upper()} ({top1_conf:.1%})"
    actual_text = f"Actual: {actual_label.upper()}"
    
    # Xanh lá nếu đúng, Đỏ nếu sai
    is_correct = top1_label.lower() == actual_label.lower()
    color = (46, 204, 113) if is_correct else (231, 76, 60) # BGR (OpenCV) -> tương ứng RGB xanh/đỏ
    # Lưu ý OpenCV sử dụng hệ màu BGR nên (xanh lá = (0, 255, 0), đỏ = (0, 0, 255))
    color_bgr = (0, 255, 0) if is_correct else (0, 0, 255)
    
    cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
    cv2.putText(img, actual_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    result_base64 = image_to_base64(img)
    
    return jsonify({
        "prediction": top1_label,
        "confidence": top1_conf,
        "actual_label": actual_label,
        "is_correct": is_correct,
        "probabilities": detailed_probs,
        "result_base64": result_base64
    })

@app.route("/api/upload-egg", methods=["POST"])
def upload_egg():
    """Nhận file ảnh upload từ người dùng và dự đoán."""
    if 'file' not in request.files:
        return jsonify({"error": "Không có file nào được tải lên"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Tên file rỗng"}), 400
        
    # Tạo thư mục tạm lưu file upload
    temp_dir = os.path.join(BASE_DIR, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    
    try:
        # Chạy YOLO
        results = model(temp_path)
        r = results[0]
        
        probs = r.probs
        top1_idx = probs.top1
        top1_label = r.names[top1_idx]
        top1_conf = probs.top1conf.item()
        
        detailed_probs = {}
        for idx, label in r.names.items():
            detailed_probs[label] = float(probs.data[idx].item())
            
        # Vẽ nhãn
        img = cv2.imread(temp_path)
        label_text = f"Pred: {top1_label.upper()} ({top1_conf:.1%})"
        
        # Mặc định xanh lá khi test ảnh tự upload
        cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        result_base64 = image_to_base64(img)
        
        # Xóa file tạm sau khi xử lý xong
        os.remove(temp_path)
        
        return jsonify({
            "prediction": top1_label,
            "confidence": top1_conf,
            "actual_label": "UPLOADED (UNKNOWN)",
            "is_correct": True,
            "probabilities": detailed_probs,
            "result_base64": result_base64
        })
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"Lỗi xử lý ảnh: {str(e)}"}), 500

if __name__ == "__main__":
    # Chạy trên port 5000 ở local
    app.run(host="0.0.0.0", port=5000, debug=True)
