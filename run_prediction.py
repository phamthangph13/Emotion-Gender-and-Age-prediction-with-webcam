import cv2
import numpy as np
import pickle
import time
import os
from sklearn.preprocessing import StandardScaler
import data_processing as dp

# Tạo thư mục kết quả
os.makedirs('Report/Predictions', exist_ok=True)

def load_models():
    """Tải các mô hình đã huấn luyện và các bộ chuẩn hóa"""
    models_dir = 'models'
    
    # Tải mô hình dự đoán tuổi
    with open(os.path.join(models_dir, 'age_model.pkl'), 'rb') as f:
        age_model = pickle.load(f)
    
    # Tải mô hình dự đoán giới tính
    with open(os.path.join(models_dir, 'gender_model.pkl'), 'rb') as f:
        gender_model = pickle.load(f)
    
    # Tải mô hình dự đoán cảm xúc
    with open(os.path.join(models_dir, 'emotion_model.pkl'), 'rb') as f:
        emotion_model = pickle.load(f)
    
    # Tải các bộ chuẩn hóa
    with open(os.path.join(models_dir, 'scaler_age.pkl'), 'rb') as f:
        scaler_age = pickle.load(f)
    
    with open(os.path.join(models_dir, 'scaler_gender.pkl'), 'rb') as f:
        scaler_gender = pickle.load(f)
    
    with open(os.path.join(models_dir, 'scaler_emotion.pkl'), 'rb') as f:
        scaler_emotion = pickle.load(f)
    
    return {
        'age_model': age_model,
        'gender_model': gender_model,
        'emotion_model': emotion_model,
        'scaler_age': scaler_age,
        'scaler_gender': scaler_gender,
        'scaler_emotion': scaler_emotion
    }

def process_face(face, models):
    """Xử lý ảnh khuôn mặt và đưa ra dự đoán"""
    # Thay đổi kích thước thành 64x64
    face_resized = cv2.resize(face, (64, 64))
    
    # Chuyển sang ảnh xám
    gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    # Trích xuất đặc trưng HOG
    from skimage.feature import hog
    hog_features = hog(gray_face, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, 
                      block_norm='L2-Hys', transform_sqrt=True)
    
    hog_features = hog_features.reshape(1, -1)
    
    # Dự đoán tuổi
    hog_features_age = models['scaler_age'].transform(hog_features)
    age_pred = models['age_model'].predict(hog_features_age)[0]
    
    # Dự đoán giới tính
    hog_features_gender = models['scaler_gender'].transform(hog_features)
    gender_pred = models['gender_model'].predict(hog_features_gender)[0]
    gender_label = 'Nữ' if gender_pred == 1 else 'Nam'
    
    # Dự đoán cảm xúc
    hog_features_emotion = models['scaler_emotion'].transform(hog_features)
    emotion_pred = models['emotion_model'].predict(hog_features_emotion)[0]
    emotion_labels = {
        0: 'Giận dữ', 1: 'Khinh miệt', 2: 'Ghê tởm', 3: 'Sợ hãi',
        4: 'Vui vẻ', 5: 'Buồn bã', 6: 'Ngạc nhiên'
    }
    emotion_label = emotion_labels[emotion_pred]
    
    return {
        'age': int(age_pred),
        'gender': gender_label,
        'emotion': emotion_label
    }

def draw_prediction(frame, face_coords, predictions):
    """Vẽ kết quả dự đoán lên khung hình"""
    (x, y, w, h) = face_coords
    
    # Vẽ hình chữ nhật quanh khuôn mặt
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Vẽ nền cho chữ
    cv2.rectangle(frame, (x, y+h), (x+w, y+h+80), (0, 255, 0), cv2.FILLED)
    
    # Thêm chữ
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Tuổi: {predictions['age']}", (x+5, y+h+20), font, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, f"Giới tính: {predictions['gender']}", (x+5, y+h+40), font, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, f"Cảm xúc: {predictions['emotion']}", (x+5, y+h+60), font, 0.5, (0, 0, 0), 2)
    
    return frame

def run_webcam_prediction():
    """Chạy dự đoán thời gian thực bằng webcam"""
    print("Đang tải các mô hình...")
    models = load_models()
    
    print("Đang tải bộ phát hiện khuôn mặt...")
    # Sử dụng Haar cascade để phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Khởi tạo webcam
    print("Đang khởi động webcam...")
    cap = cv2.VideoCapture(0)
    
    # Kiểm tra xem webcam đã được mở thành công chưa
    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam")
        return
    
    # Thiết lập kích thước khung hình
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    fps_count = 0
    fps_start = time.time()
    
    print("Nhấn 'q' để thoát")
    
    # Tạo cửa sổ
    cv2.namedWindow('Phân tích khuôn mặt', cv2.WINDOW_NORMAL)
    
    # Xử lý các khung hình
    while True:
        # Chụp khung hình
        ret, frame = cap.read()
        
        if not ret:
            print("Lỗi: Không thể chụp khung hình")
            break
        
        # Tính FPS
        fps_count += 1
        if fps_count >= 10:
            fps = fps_count / (time.time() - fps_start)
            fps_count = 0
            fps_start = time.time()
        else:
            fps = 0
        
        # Tạo bản sao để hiển thị
        display_frame = frame.copy()
        
        # Phát hiện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Xử lý từng khuôn mặt
        for (x, y, w, h) in faces:
            # Trích xuất khuôn mặt từ khung hình
            face = frame[y:y+h, x:x+w]
            
            # Bỏ qua khuôn mặt nhỏ
            if face.shape[0] < 30 or face.shape[1] < 30:
                continue
            
            # Xử lý khuôn mặt
            try:
                predictions = process_face(face, models)
                
                # Vẽ dự đoán lên khung hình hiển thị
                display_frame = draw_prediction(display_frame, (x, y, w, h), predictions)
            except Exception as e:
                print(f"Lỗi khi xử lý khuôn mặt: {e}")
        
        # Hiển thị FPS
        if fps > 0:
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị khung hình
        cv2.imshow('Phân tích khuôn mặt', display_frame)
        
        # Thoát vòng lặp khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng webcam")

def capture_and_save(output_path='Report/Predictions/captured_frame.jpg'):
    """Chụp một khung hình, xử lý và lưu kết quả"""
    print("Đang tải các mô hình...")
    models = load_models()
    
    print("Đang tải bộ phát hiện khuôn mặt...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Khởi tạo webcam
    print("Đang khởi động webcam để chụp...")
    cap = cv2.VideoCapture(0)
    
    # Kiểm tra xem webcam đã được mở thành công chưa
    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam")
        return
    
    # Thiết lập kích thước khung hình
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Tạo cửa sổ
    cv2.namedWindow('Chụp khung hình (Nhấn SPACE để chụp)', cv2.WINDOW_NORMAL)
    
    captured = False
    
    # Xử lý các khung hình
    while not captured:
        # Chụp khung hình
        ret, frame = cap.read()
        
        if not ret:
            print("Lỗi: Không thể chụp khung hình")
            break
        
        # Tạo bản sao để hiển thị
        display_frame = frame.copy()
        
        # Phát hiện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Vẽ hình chữ nhật quanh khuôn mặt
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Hiển thị hướng dẫn
        cv2.putText(display_frame, "Nhấn SPACE để chụp", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hiển thị khung hình
        cv2.imshow('Chụp khung hình (Nhấn SPACE để chụp)', display_frame)
        
        # Kiểm tra phím nhấn
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Phím Space
            captured = True
            
            # Xử lý tất cả khuôn mặt trong khung hình
            result_frame = frame.copy()
            for (x, y, w, h) in faces:
                # Trích xuất khuôn mặt từ khung hình
                face = frame[y:y+h, x:x+w]
                
                # Bỏ qua khuôn mặt nhỏ
                if face.shape[0] < 30 or face.shape[1] < 30:
                    continue
                
                # Xử lý khuôn mặt
                try:
                    predictions = process_face(face, models)
                    
                    # Vẽ dự đoán lên khung hình kết quả
                    result_frame = draw_prediction(result_frame, (x, y, w, h), predictions)
                except Exception as e:
                    print(f"Lỗi khi xử lý khuôn mặt: {e}")
            
            # Lưu khung hình kết quả
            cv2.imwrite(output_path, result_frame)
            print(f"Đã lưu khung hình chụp vào {output_path}")
        
        elif key == ord('q'):
            break
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng webcam")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chạy dự đoán phân tích khuôn mặt')
    parser.add_argument('--capture', action='store_true', help='Chụp một khung hình duy nhất thay vì dự đoán thời gian thực')
    parser.add_argument('--output', type=str, default='Report/Predictions/captured_frame.jpg', 
                        help='Đường dẫn đầu ra cho khung hình được chụp')
    
    args = parser.parse_args()
    
    if args.capture:
        capture_and_save(args.output)
    else:
        run_webcam_prediction() 