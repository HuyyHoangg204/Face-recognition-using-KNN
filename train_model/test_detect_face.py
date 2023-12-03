import cv2
import cv2.data
import joblib
import numpy as np

# Load mô hình từ tệp đã lưu
# knn_model = joblib.load('face_recognition_model.joblib')
# knn_model = joblib.load('build_model.joblib')
knn_model = joblib.load('updated_build_model.joblib')

# Tải bộ phân loại khuôn mặt đã được huấn luyện từ OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def extract_face_features(image, faces):
    face_features = []
    for (x, y, w, h) in faces:
        face_roi = cv2.resize(image[y:y + h, x:x + w], (100, 100))
        face_features.append(face_roi.flatten())
    return np.array(face_features)


# Hàm nhận diện và vẽ bounding box cho khuôn mặt trên khung hình
def recognize_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

    # Rút trích đặc trưng từ khuôn mặt
    if len(faces) > 0:
        face_features = extract_face_features(frame, faces)

        # Dự đoán nhãn cho khuôn mặt sử dụng mô hình KNN
        predictions = knn_model.predict(face_features)

        # Vẽ bounding box và hiển thị nhãn dự đoán
        for i, (x, y, w, h) in enumerate(faces):
            label = predictions[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# Mở kết nối với webcam
cap = cv2.VideoCapture(0)

# Đảm bảo kết nối với webcam được mở thành công
if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    # Nhận diện khuôn mặt và vẽ bounding box
    recognize_faces(frame)

    # Hiển thị khung hình
    cv2.imshow("Face Recognition", frame)

    # Thoát nếu nhấn phím 'Esc'
    if cv2.waitKey(1) == 27:
        break

# Đóng kết nối với webcam và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
