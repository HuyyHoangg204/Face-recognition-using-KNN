import cv2
import cv2.data
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import re
import time


def extract_face_features(image, faces):
    face_features = []
    for (x, y, w, h) in faces:
        face_roi = cv2.resize(image[y:y + h, x:x + w], (50, 50))
        face_features.append(face_roi.flatten()) # chuyển ảnh thành dạng 1D
    return np.array(face_features)


# Hàm để loại bỏ số từ chuỗi
def remove_numbers(input_string):
    return re.sub(r'\d+', '', input_string)


# Đọc dữ liệu ảnh và gắn nhãn
data = []
labels = []

# Thư mục chứa các ảnh đã chụp
images_folder = "captured_images"

if not os.path.exists(images_folder):
    os.makedirs(images_folder)

# Thư mục để lưu trữ các khuôn mặt đã phát hiện
detected_faces_folder = "detected_faces"
if not os.path.exists(detected_faces_folder):
    os.makedirs(detected_faces_folder)

# Load bộ phân loại khuôn mặt từ OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Đọc tất cả các tệp trong thư mục images_folder
# Lấy khuôn mặt
for filename in os.listdir(images_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path)

        # Phát hiện khuôn mặt trong ảnh
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        # Rút trích đặc trưng từ khuôn mặt và gắn nhãn
        if len(faces) > 0:

            # Xử lý tên ảnh để lấy tên người và gắn nhãn
            person_name = remove_numbers(os.path.splitext(filename)[0])
            person_folder = os.path.join(detected_faces_folder, person_name)
            print(f"person_name: {person_name}, num_faces: {len(faces)}")
            labels.extend([person_name] * len(faces))

            # Kiểm tra và tạo thư mục nếu chưa tồn tại
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)

            face_features = extract_face_features(image, faces)
            data.extend(face_features)

            # Lưu khuôn mặt phát hiện đầu tiên vào thư mục
            for j, (x, y, w, h) in enumerate(faces):
                face_roi = image[y:y + h, x:x + w]

                cur_time_millis = int(round(time.time() * 1000))

                cv2.imwrite(f"{person_folder}/{person_name}_{cur_time_millis}.png", face_roi)
                break

# Chuyển đổi danh sách thành mảng NumPy
data = np.array(data)
labels = np.array(labels)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Tạo và huấn luyện mô hình KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
predictions = knn_model.predict(X_test)

# Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Lưu mô hình vào tệp
joblib.dump(knn_model, 'model/fast_build.joblib')
