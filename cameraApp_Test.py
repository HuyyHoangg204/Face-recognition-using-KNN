import cv2
import cv2.data
import numpy as np
import tkinter as tk
from PyQt6.QtWidgets import QMessageBox
from sklearn.neighbors import KNeighborsClassifier

import SQL


class CameraApp:
    button_x, button_y, button_width, button_height = 10, 10, 100, 40
    cur_frame = None
    img_counter = 0

    def __init__(self, tk, mk):
        self.tk_dang_ky = tk
        self.mk_dang_ky = mk

        # Tải bộ phân loại khuôn mặt đã được huấn luyện từ OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Tạo một mô hình KNN trống
        self.knn_model = KNeighborsClassifier(n_neighbors=3)

    def train_knn_model(self, features, labels):
        self.knn_model.fit(features, labels)

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        return faces

    def extract_face_features(self, image, faces):
        face_features = []
        for (x, y, w, h) in faces:
            face_roi = cv2.resize(image[y:y + h, x:x + w], (50, 50))
            face_features.append(face_roi.flatten())

        # Chuyển đổi danh sách thành mảng NumPy
        return np.array(face_features)

    def show_notification(self, message):
        notification_window = tk.Tk()
        notification_window.title("Thông báo")

        label = tk.Label(notification_window, text=message)
        label.pack(padx=10, pady=10)

        ok_button = tk.Button(notification_window, text="OK", command=notification_window.destroy)
        ok_button.pack(pady=10)
        notification_window.mainloop()

    # Thêm các biến để lưu dữ liệu huấn luyện
    training_data = []
    training_labels = []

    def on_button_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Save data
            if (self.button_x <= x <= self.button_x + self.button_width and
                    self.button_y <= y <= self.button_y + self.button_height):
                face_features = self.extract_face_features(self.cur_frame, self.detect_face(self.cur_frame))

                # Sử dụng thông tin đăng nhập từ file main.py (hoặc thay đổi cách bạn lấy thông tin)
                # SQL.save_data_to_mysql(self.tk_dang_ky, self.mk_dang_ky, face_features)

                # Lưu ảnh ra folder
                # img_name = "data/{}_{}.png".format(self.tk_dang_ky, self.img_counter)
                # cv2.imwrite(img_name, self.cur_frame)

                # thông báo
                self.show_notification("Ảnh đã được chụp:")
                self.img_counter += 1

    def show_cam(self):
        # Gắn sự kiện nhấn chuột vào cửa sổ
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Webcam Screenshot")
        cv2.setMouseCallback("Webcam Screenshot", self.on_button_click)
        while True:
            ret, self.cur_frame = cam.read()

            if not ret:
                print("Lỗi: Không thể lấy được khung hình")
                break

            # Hiển thị nút "Chụp ảnh"
            cv2.rectangle(self.cur_frame, (self.button_x, self.button_y), (self.button_x + self.button_width,
                                                                           self.button_y + self.button_height),
                          (255, 255, 255), -1)
            cv2.putText(self.cur_frame, "Take Picture", (self.button_x + 10, self.button_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0),
                        2)

            # Hiển thị khung hình
            cv2.imshow("Webcam Screenshot", self.cur_frame)

            # Nhấn 'Esc' hoặc kiểm tra trạng thái của cửa sổ để thoát
            k = cv2.waitKey(1)
            if k % 256 == 27 or cv2.getWindowProperty("Webcam Screenshot", cv2.WND_PROP_VISIBLE) < 1:
                print("Thoát khỏi cửa sổ webcam")
                break

        # Giải phóng camera và đóng cửa sổ
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    cam = CameraApp("admin", "123")
    cam.show_cam()
