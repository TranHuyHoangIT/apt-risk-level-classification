# APT Risk Level Classification
## Hướng dẫn sử dụng
### 0. Prerequisites
Đảm các công cụ sau đã được cài đặt:
- Node.js (phiên bản 18.x hoặc cao hơn) và npm (phiên bản 9.x hoặc cao hơn)
- Python (phiên bản 3.10 hoặc cao hơn)
- pip (tương thích với phiên bản Python)
### 1. Clone repository
```bash
https://github.com/TranHuyHoangIT/apt-risk-level-classification.git
```
### 2. Cài đặt dependencies
**Backend**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend**
```bash
cd frontend
npm install
```
### 3. Setup database
- Cấu hình cài đặt cơ sở dữ liệu MySQL trong file backend/app.py (username, password, database).

### 4. Run application
**Backend**
```bash
cd backend
python app.py
```

**Frontend**
```bash
cd frontend
npm start
```
Ứng dụng sẽ tự động khởi chạy và có thể truy cập tại địa chỉ: http://localhost:3000.

## 5. Cấu trúc thư mục của hệ thống 📂 
```plaintext
apt-risk-level-classification/
├── frontend/                # Giao diện người dùng (phần giao diện phía client)
│   ├── node_modules/        # Thư mục chứa các gói phụ thuộc của Node.js
│   ├── public/              # Thư mục chứa các tệp tĩnh (static files)
│   ├── src/                 # Thư mục chứa mã nguồn giao diện
│   │   ├── components/      # Các thành phần tái sử dụng của giao diện
│   │   ├── pages/           # Các trang chính của ứng dụng
│   │   ├── services/        # Các dịch vụ và function hỗ trợ API
│   │   ├── styles/          # Các tệp CSS và style của ứng dụng
│   │   ├── utils/           # Các hàm tiện ích chung
│   │   ├── App.jsx          # Thành phần chính của ứng dụng React
│   │   ├── App.css          # Tệp CSS dành riêng cho App.jsx
│   │   ├── index.css        # Tệp CSS toàn cục
│   │   └── index.js         # Điểm khởi chạy ứng dụng React
│   ├── package-lock.json    # File khóa phiên bản các gói phụ thuộc
│   ├── package.json         # File quản lý phụ thuộc và script của frontend
│   ├── README.md            # Tài liệu hướng dẫn cho phần frontend
│   └── tailwind.config.js   # Cấu hình Tailwind CSS
├── backend/                 # Xử lý logic phía server
│   ├── app.py               # Điểm khởi động chính của ứng dụng backend
│   ├── routes.py            # Định nghĩa các endpoint API
│   ├── models.py            # Các mô hình dữ liệu cơ sở dữ liệu
│   ├── cicflowmeter/        # Module cicflowmeter
│   ├── data/                # Thư mục chứa file CSV dữ liệu train model 
│   ├── model_trained/       # Thư mục chứa mô hình đã huấn luyện
│   ├── uploads/             # Thư mục lưu trữ các tệp được tải lên
│   └── requirements.txt     # File liệt kê các dependencies Python của backend
├── README.md                # Tài liệu hướng dẫn tổng quan cho dự án
└── APT_SQL.sql              # File script SQL để cấu hình cơ sở dữ liệu