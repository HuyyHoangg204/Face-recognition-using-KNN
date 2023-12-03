import mysql.connector
import re

mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'khanhdz123',
    'database': 'tri_tue_nhan_tao'
}


def clean_input(input_string):
    # Hàm này sử dụng regex để loại bỏ các ký tự không an toàn từ chuỗi
    return re.sub(r'[^a-zA-Z0-9]', '', input_string)


def check_credentials(username, password):
    try:
        # Làm sạch tên người dùng và mật khẩu trước khi truy vấn
        clean_username = clean_input(username)
        clean_password = clean_input(password)

        # Kết nối đến cơ sở dữ liệu
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor()

        # Truy vấn SQL để kiểm tra tài khoản và mật khẩu
        query = "SELECT * FROM user_data WHERE username = %s AND password = %s"
        cursor.execute(query, (clean_username, clean_password))
        result = cursor.fetchone()

        # Đóng kết nối
        cursor.close()
        conn.close()

        # Trả về kết quả kiểm tra
        return result is not None

    except Exception as e:
        print("Lỗi check_credentials:", str(e))
        return False


def check_username_exist(username):
    try:
        # Làm sạch tên người dùng trước khi truy vấn
        clean_username = clean_input(username)

        # Kết nối đến cơ sở dữ liệu
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor()

        # Truy vấn SQL để kiểm tra xem tên người dùng đã tồn tại hay chưa
        query = "SELECT * FROM user_data WHERE username = %s"
        cursor.execute(query, (clean_username,))
        result = cursor.fetchone()

        # Đóng kết nối
        cursor.close()
        conn.close()

        # Trả về kết quả kiểm tra
        return result is not None

    except Exception as e:
        print("Lỗi check_username_exist:", str(e))
        return False


def register_user(username, password):
    try:
        # Làm sạch tên người dùng và mật khẩu trước khi thêm vào cơ sở dữ liệu
        clean_username = clean_input(username)
        clean_password = clean_input(password)

        # Kiểm tra xem tên người dùng đã tồn tại hay chưa
        if check_username_exist(clean_username):
            return False  # Tên người dùng đã tồn tại

        # Kết nối đến cơ sở dữ liệu
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor()

        # Truy vấn SQL để thêm người dùng mới
        query = "INSERT INTO user_data (username, password) VALUES (%s, %s)"
        cursor.execute(query, (clean_username, clean_password))
        conn.commit()

        # Đóng kết nối
        cursor.close()
        conn.close()

        return True  # Đăng ký thành công

    except Exception as e:
        print("Lỗi register_user:", str(e))
        return False


def save_data_to_mysql(username, password):
    connection = None
    try:
        # Kết nối đến MySQL
        connection = mysql.connector.connect(
            host=mysql_config['host'],
            user=mysql_config['user'],
            password=mysql_config['password'],
            database=mysql_config['database']
        )

        if connection.is_connected():
            cursor = connection.cursor(prepared=True)

            # Kiểm tra nếu bảng không tồn tại, tạo bảng
            cursor.execute("CREATE TABLE IF NOT EXISTS user_data (id INT AUTO_INCREMENT PRIMARY KEY, "
                           "username VARCHAR(255), password VARCHAR(255))")

            # Thực hiện INSERT vào bảng người dùng
            query = "INSERT INTO user_data (username, password) VALUES (%s, %s)"
            cursor.execute(query, (username, password))

            # Lưu thay đổi và đóng kết nối
            connection.commit()
            cursor.close()

            print("Đã lưu dữ liệu vào MySQL.")

    except mysql.connector.Error as e:
        print("Lỗi MySQL:", str(e))

    finally:
        if connection and connection.is_connected():
            connection.close()
