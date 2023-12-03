from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox

import SQL
from cam_dang_nhap import CamDangNhap
from gui.demo import Ui_MainWindow


class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Kết nối sự kiện clicked của nút "Đăng nhập" với hàm xử lý
        self.ui.pushButton.clicked.connect(self.dang_nhap_clicked)
        self.ui.pushButton_2.clicked.connect(self.dang_ky_clicked)

    def dang_nhap_clicked(self):
        try:
            username = self.ui.text_DangNhap.text()
            password = self.ui.text_DangKy.text()
            if not username or not password:
                error_message = QMessageBox()
                error_message.setIcon(QMessageBox.Icon.Critical)
                error_message.setText("Tài khoản hoặc mật khẩu không được để trống.")
                error_message.setWindowTitle("Lỗi")
                error_message.exec()
            else:
                # Gọi hàm kiểm tra tài khoản và mật khẩu từ file SQL.py
                if SQL.check_credentials(username, password):
                    # Nếu kiểm tra thành công, bắt đầu nhận diện khuôn mặt
                    self.start_webcam_dang_nhap(username, password)
                else:
                    # Nếu kiểm tra không thành công, hiển thị thông báo lỗi
                    error_message = QMessageBox()
                    error_message.setIcon(QMessageBox.Icon.Critical)
                    error_message.setText("Tài khoản hoặc mật khẩu không chính xác.")
                    error_message.setWindowTitle("Lỗi")
                    error_message.exec()

        except Exception as e:
            print("Lỗi dang_nhap_clicked:", str(e))

    def start_webcam_dang_nhap(self, username, password):
        print("start_webcam")
        cam = CamDangNhap(username, password)
        cam.show_cam()
        pass

    def dang_ky_clicked(self):
        try:
            tk_dang_ky = self.ui.text_DangNhap.text()
            mk_dang_ky = self.ui.text_DangKy.text()

            if not tk_dang_ky or not mk_dang_ky:
                error_message = QMessageBox()
                error_message.setIcon(QMessageBox.Icon.Critical)
                error_message.setText("Tài khoản hoặc mật khẩu không được để trống.")
                error_message.setWindowTitle("Lỗi")
                error_message.exec()
            else:
                # Kiểm tra xem tên người dùng đã tồn tại hay chưa
                if SQL.check_username_exist(tk_dang_ky):
                    error_message = QMessageBox()
                    error_message.setIcon(QMessageBox.Icon.Critical)
                    error_message.setText("Tài khoản đã tồn tại.")
                    error_message.setWindowTitle("Lỗi")
                    error_message.exec()
                else:
                    # xác thực thêm khuôn mặt ở đây

                    # Nếu tên người dùng chưa tồn tại, thực hiện đăng ký
                    if SQL.register_user(tk_dang_ky, mk_dang_ky):
                        success_message = QMessageBox()
                        success_message.setIcon(QMessageBox.Icon.Information)
                        success_message.setText("Đăng ký thành công.")
                        success_message.setWindowTitle("Thông báo")
                        success_message.exec()
                    else:
                        error_message = QMessageBox()
                        error_message.setIcon(QMessageBox.Icon.Critical)
                        error_message.setText("Đăng ký không thành công. Vui lòng thử lại.")
                        error_message.setWindowTitle("Lỗi")
                        error_message.exec()

        except Exception as e:
            print("Lỗi dang_ky_clicked:", str(e))
            # Xử lý ngoại lệ (nếu cần)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', 'Bạn có muốn thoát không?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    MainApp = MainApplication()
    MainApp.show()
    sys.exit(app.exec())
