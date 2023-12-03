import sys

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


def extract_face_features(image):
    # Phát hiện khuôn mặt trong ảnh
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100))
    if len(faces) <= 0:
        return None

    # Rút trích đặc trưng từ toàn bộ ảnh
    face_features = []
    for (x, y, w, h) in faces:
        face_roi = cv2.resize(image[y:y + h, x:x + w], (100, 100))
        face_features.append(face_roi.flatten())  # chuyển ảnh thành dạng 1D

        cur_time_millis = int(round(time.time() * 1000))
        cv2.imwrite(f"detected_faces_new/build/{cur_time_millis}.png", face_roi)
    return np.array(face_features)


# Hàm để loại bỏ số từ chuỗi
def remove_numbers(input_string):
    return re.sub(r'\d+', '', input_string)


# Đọc dữ liệu ảnh và gắn nhãn
data = []
labels = []

# Đường dẫn lưu mô hình vào tệp
name_model = "model/build_model.joblib"

# Thư mục để lưu trữ các khuôn mặt đã phát hiện
detected_faces_folder = "detected_faces_new"
if not os.path.exists(detected_faces_folder):
    os.makedirs(detected_faces_folder)

# Thư mục để lưu trữ các khuôn mặt đã được huấn luyện cho model
build_folder = "build"
if not os.path.exists(build_folder):
    os.makedirs(build_folder)

# Load bộ phân loại khuôn mặt từ OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Đọc tất cả các tệp trong thư mục detected_faces_folder
for folder in os.listdir(detected_faces_folder):
    # không đọc các ảnh có trong folder build
    if folder == build_folder:
        continue
    person_path = os.path.join(detected_faces_folder, folder)

    # Kiểm tra xem item là folder hay không
    if os.path.isdir(person_path):
        print("path: " + person_path)
        for filename in os.listdir(person_path):
            if filename.endswith(".png"):
                image_path = os.path.join(person_path, filename)
                image = cv2.imread(image_path)

                # Xử lý tên ảnh để lấy tên người và gắn nhãn
                person_name = remove_numbers(os.path.splitext(filename)[0])
                person_folder = os.path.join(detected_faces_folder, person_name)
                print(f"person_name: {person_name}, file name: {filename}")

                # Rút trích đặc trưng từ toàn bộ ảnh
                face_features = extract_face_features(image)
                # nếu không rút trích được đặc trưng thì bỏ qua
                if face_features is None:
                    continue
                # Gắn nhãn cho mỗi khuôn mặt với tên người tương ứng
                labels.extend([person_name] * len(face_features))
                data.extend(face_features)

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

joblib.dump(knn_model, name_model)

print("[+] Build Model XONG ...")
