import cv2
import cv2.data
import tkinter as tk
import joblib
import numpy as np
import time


class CamDangKy:
    button_x, button_y, button_width, button_height = 10, 10, 100, 40
    text_x, text_y, text_width, text_height = 10, 60, 150, 40
    img_counter = 0
    cur_frame = None
    ret = None
    doneShow = False

    def __init__(self, tk_dang_ky, mk):
        self.start_time = None
        self.tk_dang_ky = tk_dang_ky
        self.mk_dang_ky = mk
        self.image_collection = []
        # Load mô hình từ tệp đã lưu
        self.knn_model = joblib.load('train_model/face_recognition_model.joblib')

        # Tải bộ phân loại khuôn mặt đã được huấn luyện từ OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        return faces

    def extract_face_features(self, image, faces):
        face_features = []
        for (x, y, w, h) in faces:
            face_roi = cv2.resize(image[y:y + h, x:x + w], (50, 50))
            face_features.append(face_roi.flatten())
        return np.array(face_features)

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
        cv2.setMouseCallback("Webcam Screenshot", self.on_button_click)
        self.start_time = time.time()  # Thời điểm bắt đầu
        while True:
            self.ret, self.cur_frame = cam.read()

            if not self.ret:
                print("Lỗi: Không thể lấy được khung hình")
                break

            # Hiển thị nút "Chụp ảnh"
            cv2.rectangle(self.cur_frame, (self.button_x, self.button_y), (self.button_x + self.button_width,
                                                                           self.button_y + self.button_height),
                          (255, 255, 255), -1)
            cv2.putText(self.cur_frame, "Chup anh", (self.button_x + 10, self.button_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0),
                        2)

            if self.img_counter > 3:
                # Hiển thị nút "Lưu dữ liệu"
                cv2.rectangle(self.cur_frame, (self.text_x, self.text_y), (self.text_x + self.text_width,
                                                                           self.text_y + self.text_height),
                              (255, 255, 255), -1)
                cv2.putText(self.cur_frame, "Luu du lieu", (self.text_x + 10, self.text_y + 30),
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

            if self.doneShow:
                break

        # Giải phóng camera và đóng cửa sổ
        cam.release()
        cv2.destroyAllWindows()

    def on_button_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Save data
            if (self.button_x <= x <= self.button_x + self.button_width and
                    self.button_y <= y <= self.button_y + self.button_height):
                if self.ret:
                    gray_frame = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3,
                                                               minSize=(30, 30))
                    if len(faces) > 0:
                        # Lưu ảnh đầy đủ vào danh sách
                        self.image_collection.append(self.cur_frame)
                        # thông báo
                        self.show_notification("Ảnh đã được chụp:")
                        self.img_counter += 1

            elif (self.text_x <= x <= self.text_x + self.text_width and
                  self.text_y <= y <= self.text_y + self.text_height):

                if self.image_collection:
                    # for image in self.image_collection:
                    #     # Chuyển đổi ảnh sang ảnh xám
                    #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #
                    #     # Phát hiện khuôn mặt trong ảnh mới
                    #     faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3,
                    #                                           minSize=(30, 30))
                    #     i = 0
                    #     if len(faces) > 0:
                    #         # Rút trích đặc trưng từ khuôn mặt mới
                    #         new_face_features = self.extract_face_features(image, faces)
                    #
                    #         # Chắc chắn rằng new_face_features có kích thước 900
                    #         new_face_features = new_face_features.flatten()
                    #
                    #         # Cập nhật dữ liệu huấn luyện với đặc trưng và nhãn mới
                    #         new_predictions = self.knn_model.predict([new_face_features])
                    #         self.knn_model._fit_X = np.concatenate((self.knn_model._fit_X, [new_face_features]), axis=0)
                    #         self.knn_model._y = np.concatenate((self.knn_model._y, new_predictions), axis=0)
                    #
                    #         img_name = "dangky/{}_{}.png".format(self.tk_dang_ky, i)
                    #         i += 1
                    #         cv2.imwrite(img_name, self.cur_frame)
                    #
                    # # Huấn luyện lại mô hình KNN với dữ liệu huấn luyện đã được cập nhật
                    # self.knn_model.fit(self.knn_model._fit_X, self.knn_model._y)
                    #
                    # # Lưu mô hình đã cập nhật
                    # joblib.dump(self.knn_model, 'face_recognition_model_updated.joblib')
                    # # Hiển thị thông báo
                    # self.show_notification("Lưu dữ liệu thành công!!")
                    self.show_notification("Lưu dữ liệu đang trong quá trình phát triển")
                else:
                    self.show_notification("Không có dữ liệu để lưu")


if __name__ == "__main__":
    cam = CamDangKy("khanh", "123")
    cam.show_cam()
