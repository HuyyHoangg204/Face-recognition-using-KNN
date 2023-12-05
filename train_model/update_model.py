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


def extract_face_features(image, face_cascade):
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


def update_model(detected_faces_folder, model_file_path, path_update):
    # Đọc dữ liệu ảnh và gắn nhãn
    data = []
    labels = []

    if not os.path.exists(detected_faces_folder):
        os.makedirs(detected_faces_folder)

    # Load bộ phân loại khuôn mặt từ OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Đọc tất cả các tệp trong thư mục detected_faces_folder
    for folder in os.listdir(detected_faces_folder):
        if folder == "build":
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
                    face_features = extract_face_features(image, face_cascade)
                    # nếu không rút trích được đặc trưng thì bỏ qua
                    if face_features is None:
                        continue
                    # Gắn nhãn cho mỗi khuôn mặt với tên người tương ứng
                    labels.extend([person_name] * len(face_features))
                    data.extend(face_features)

    # Chuyển đổi danh sách thành mảng NumPy
    data = np.array(data)
    labels = np.array(labels)

    # Khai báo biến model
    knn_model = None

    # Kiểm tra xem tệp mô hình có tồn tại hay không
    if os.path.exists(model_file_path):
        # Load mô hình đã huấn luyện nếu tệp tồn tại
        knn_model = joblib.load(model_file_path)
        print("Model loaded successfully.")
    else:
        # Nếu tệp không tồn tại, tạo một mô hình mới
        knn_model = KNeighborsClassifier(n_neighbors=3)
        print("Model loaded failed, Create new model.")

    # Thêm dữ liệu mới vào tập huấn luyện
    knn_model.fit(data, labels)

    # Lưu lại mô hình sau khi đã thêm dữ liệu mới
    joblib.dump(knn_model, path_update)

    print("[+] Update Model XONG ...")


if __name__ == "__main__":
    import sys

    detected_faces_folder = "detected_faces_update"  # Thư mục để lưu trữ các khuôn mặt cần update cho model
    model_file_path = 'model/build_model.joblib'  # Đường dẫn đến tệp lưu trữ mô hình
    path_update = 'model/updated_build_model.joblib'  # Đường dẫn lưu model mới
    update_model(detected_faces_folder, model_file_path, path_update)
