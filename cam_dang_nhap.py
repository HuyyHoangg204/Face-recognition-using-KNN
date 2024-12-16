import cv2
import cv2.data
import tkinter as tk
import joblib
import numpy as np
import time
import train_model.test_detect_face


class CamDangNhap:
    cur_frame = None
    doneShow = False

    def __init__(self, tk, mk):
        self.start_time = None
        self.tk_dang_ky = tk
        self.mk_dang_ky = mk
        # Load mô hình từ tệp đã lưu
        self.knn_model = joblib.load('train_model/model/updated_build_model.joblib')

        # Tải bộ phân loại khuôn mặt đã được huấn luyện từ OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recognize_faces(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        # Rút trích đặc trưng từ khuôn mặt
        if len(faces) > 0:
            face_features = train_model.test_detect_face.extract_face_features(frame, faces)

            # Dự đoán nhãn cho khuôn mặt sử dụng mô hình KNN
            predictions = self.knn_model.predict(face_features)


            # Vẽ bounding box và hiển thị nhãn dự đoán
            for i, (x, y, w, h) in enumerate(faces):
                label = predictions[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Target', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if (time.time() - self.start_time) > 3:
                    if label == self.tk_dang_ky:
                        self.show_notification(f"Nhận diện thành công tài khoản {self.tk_dang_ky}")
                        self.doneShow = True
                    else:
                        if (time.time() - self.start_time) > 5:
                            self.show_notification("Nhận diện thất bại")
                            self.doneShow = True

    def show_notification(self, message):
        notification_window = tk.Tk()
        notification_window.title("Thông báo")

        label = tk.Label(notification_window, text=message)
        label.pack(padx=10, pady=10)

        ok_button = tk.Button(notification_window, text="OK", command=notification_window.destroy)
        ok_button.pack(pady=10)
        notification_window.mainloop()

    def show_cam(self):
        # Gắn sự kiện nhấn chuột vào cửa sổ
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Webcam Screenshot")
        self.start_time = time.time()  # Thời điểm bắt đầu
        while True:
            ret, self.cur_frame = cam.read()

            if not ret:
                print("Lỗi: Không thể lấy được khung hình")
                break

            self.recognize_faces(self.cur_frame)

            # Hiển thị khung hình
            cv2.imshow("Webcam Screenshot", self.cur_frame)

            # Nhấn 'Esc' hoặc kiểm tra trạng thái của cửa sổ để thoát
            k = cv2.waitKey(1)
            if k % 256 == 27 or cv2.getWindowProperty("Webcam Screenshot", cv2.WND_PROP_VISIBLE) < 1:
                print("Thoát khỏi cửa sổ webcam")
                break

            if self.doneShow:
                break

        # Giải phóng camera và đóng cửa sổ
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = CamDangNhap("khanh", "123")
    cam.show_cam()
