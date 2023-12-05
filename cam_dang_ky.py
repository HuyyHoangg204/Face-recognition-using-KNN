import cv2
import cv2.data
import tkinter as tk
import joblib
import numpy as np
import time
import train_model.prepare_image
import train_model.update_model


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
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

        # train model test hàm này sẽ được gọi bên file main.py khi đăng ký,
        # tháo comment này ra để test thôi
        # self.update_model()

        # Giải phóng camera và đóng cửa sổ
        cam.release()
        cv2.destroyAllWindows()

    def on_button_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # event nút chụp ảnh
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
                    else:
                        self.show_notification("Ảnh phải có khuôn mặt")

            # event nút lưu dữ liệu
            elif (self.text_x <= x <= self.text_x + self.text_width and
                  self.text_y <= y <= self.text_y + self.text_height):

                if self.image_collection:
                    self.doneShow = True
                else:
                    self.show_notification("Không có dữ liệu để lưu")

    def update_model(self):
        try:
            # lưu ảnh đã chụp vào thư mục
            target_directory = "train_model\captured_images_update"
            for index, frame in enumerate(self.image_collection):
                image_name = f"{self.tk_dang_ky}{index + 1}.png"
                import os
                target_path = os.path.join(target_directory, image_name)
                # Lưu ảnh vào thư mục đích
                cv2.imwrite(target_path, frame)

                print(f"Image {index + 1} saved to {target_path}")

            # Tiền sử lý dữ liệu:
            images_folder = "train_model/captured_images_update"  # Thư mục chứa các ảnh đã chụp
            detected_faces_folder = "train_model/detected_faces_update"  # Thư mục để lưu trữ các khuôn mặt đã phát hiện
            train_model.prepare_image.prepare_image(images_folder, detected_faces_folder)

            # update model
            model_file_path = 'train_model/model/build_model.joblib'  # Đường dẫn đến tệp lưu trữ mô hình
            path_update = 'train_model/model/updated_build_model.joblib'  # Đường dẫn lưu model mới
            train_model.update_model.update_model(detected_faces_folder, model_file_path, path_update)
            self.show_notification("Đã cập nhật dữ liệu vào model mới")
            return True
        except Exception as e:
            print("Lỗi update_model: ", str(e))
            return False


if __name__ == "__main__":
    cam = CamDangKy("khanh", "123")
    cam.show_cam()
