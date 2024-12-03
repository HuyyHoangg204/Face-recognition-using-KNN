import os
import sys
import re
import cv2
import cv2.data
import numpy as np
import time


# Hàm để loại bỏ số từ chuỗi
def remove_numbers(input_string):
    return re.sub(r'\d+', '', input_string)


def prepare_image(images_folder, detected_faces_folder):
    if not os.path.exists(images_folder):
        print("Thư mục chứa ảnh không tồn tại. Kết thúc tiền sử lý dữ liệu")
        sys.exit()

    if not os.path.exists(detected_faces_folder):
        print("Thư mục lưu trữ các khuôn mặt đã phát hiện không tồn tại, <[TẠO MỚI THƯ MỤC]>")
        os.makedirs(detected_faces_folder)

    # Load bộ phân loại khuôn mặt từ OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Đọc tất cả các tệp trong thư mục images_folder
    for filename in os.listdir(images_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            print("image_path: " + image_path)
            image = cv2.imread(image_path)

            # Phát hiện khuôn mặt trong ảnh
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(150, 150))

            # Rút trích đặc trưng từ khuôn mặt và gắn nhãn
            if len(faces) > 0:

                # Xử lý tên ảnh để lấy tên người và gắn nhãn
                person_name = remove_numbers(os.path.splitext(filename)[0])
                person_folder = os.path.join(detected_faces_folder, person_name)
                print(f"person_name: {person_name}, số lượng mặt trong ảnh: {len(faces)}, file name: {filename}")

                # Kiểm tra và tạo thư mục nếu chưa tồn tại
                if not os.path.exists(person_folder):
                    os.makedirs(person_folder)

                # Lưu khuôn mặt đã phát hiện vào thư mục
                for j, (x, y, w, h) in enumerate(faces):
                    face_roi = image[y:y + h, x:x + w]

                    cur_time_millis = int(round(time.time() * 1000))

                    cv2.imwrite(f"{person_folder}/{person_name}{cur_time_millis}.png", face_roi)

    print("[+] XONG ...")


if __name__ == "__main__":
    images_folder = "captured_images"  # Thư mục chứa các ảnh đã chụp
    detected_faces_folder = "detected_faces"  # Thư mục để lưu trữ các khuôn mặt đã phát hiện
    prepare_image(images_folder, detected_faces_folder)
