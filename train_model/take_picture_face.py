import cv2


def capture_images(output_folder, num_images=10):
    # Mở kết nối với webcam
    cap = cv2.VideoCapture(0)

    # Đảm bảo kết nối với webcam được mở thành công
    if not cap.isOpened():
        print("Không thể mở webcam.")
        return

    # Lặp để chụp số lượng ảnh cần thiết
    for i in range(num_images):
        # Đọc khung hình từ webcam
        ret, frame = cap.read()

        # Hiển thị khung hình
        cv2.imshow("Press 'Esc' to capture", frame)

        # Chờ phím bấm
        key = cv2.waitKey(0)

        # Nếu phím bấm là 'Esc' (27), thoát khỏi vòng lặp
        if key == 27:
            break
        # Nếu phím bấm là 'Space' (32), lưu ảnh vào thư mục đích
        elif key == 32:
            image_path = f"{output_folder}/{image_name}{i + 1}.png"
            cv2.imwrite(image_path, frame)
            print(f"Captured image {i + 1} saved as {image_path}")

    # Đóng kết nối với webcam và đóng cửa sổ hiển thị
    cap.release()
    cv2.destroyAllWindows()


image_name = "bon"

# Gọi hàm để chụp 10 ảnh và lưu vào thư mục "captured_images"
capture_images(output_folder="captured_images", num_images=10)
