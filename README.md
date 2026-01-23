# AI_Nông_Nghiệp 🌾🤖

Dự án nghiên cứu khoa học ứng dụng Trí tuệ nhân tạo (AI) trong việc xử lý, truy xuất và phân tích tài liệu tiếng Việt, phục vụ lĩnh vực nông nghiệp và các bài toán liên quan đến biến đổi khí hậu.

---

## 📌 Giới thiệu
Dự án xây dựng một hệ thống AI hỗ trợ nghiên cứu nông nghiệp dựa trên:
- Mô hình embedding tài liệu tiếng Việt
- Cơ chế truy xuất tăng cường (Retrieval-Augmented Generation – RAG)
- Vector Database phục vụ tìm kiếm ngữ nghĩa

Hệ thống hướng tới việc hỗ trợ sinh viên và nhà nghiên cứu trong việc khai thác tri thức từ tài liệu chuyên ngành nông nghiệp.

---

## ⚠️ Lưu ý quan trọng về mô hình AI
Do **giới hạn dung lượng file của GitHub**, các **mô hình AI KHÔNG được lưu trữ trong repository này**.

👉 Người dùng **bắt buộc phải tải mô hình từ Google Drive** trước khi chạy project.

---

## 📦 Tải mô hình (BẮT BUỘC)

### 🔹 Vietnamese Document Embedding Model
- Nơi lưu trữ: **Google Drive**
- Link tải:  
👉 **https://drive.google.com/file/d/1IGYuMceO5dA_HncZmXtoEalpYd8Ti2Pv/view?usp=sharing**

### 📂 Cấu trúc sau khi tải và giải nén
Sau khi tải xong, giải nén và đặt mô hình đúng đường dẫn sau:

```
AI_NongNghiep/
│── models/
│   └── s.vietnamese-document-embedding/
│       └── model.safetensors
```

⚠️ **Không đổi tên thư mục hoặc file**, nếu sai đường dẫn chương trình sẽ không chạy.

---

## 🚀 Cách chạy project (rất đơn giản)

### Yêu cầu
- Đã cài **Python 3.9 trở lên**
- Đã tải và đặt model đúng như hướng dẫn trên

### ▶️ Chạy chương trình
Chỉ cần **double-click** vào file:

```
run.bat
```

File này sẽ tự động:
- Tạo virtual environment
- Kích hoạt môi trường
- Cài đặt thư viện cần thiết
- Chuẩn bị vector database
- Chạy ứng dụng AI

👉 **Không cần gõ lệnh thủ công**

---

## 📂 Cấu trúc thư mục chính

```
AI_NongNghiep/
│── models/              # Chứa model (không theo dõi bởi Git)
│── data/                # Dữ liệu đầu vào
│── src/                 # Mã nguồn
│── vectorstore/         # Vector database
│── run.bat              # File chạy tự động
│── requirements.txt
│── README.md
```

---

## 🧠 Bối cảnh nghiên cứu khoa học
Dự án được thực hiện trong khuôn khổ **nghiên cứu khoa học sinh viên**, tập trung vào:
- AI trong nông nghiệp
- Xử lý ngôn ngữ tự nhiên (NLP) tiếng Việt
- Embedding & Vector Search
- Retrieval-Augmented Generation (RAG)
- Hướng tới khả năng mở rộng và giải thích mô hình (XAI)

---

## 📄 Ghi chú cho giảng viên & người đánh giá
- Repository chỉ chứa **mã nguồn**, không chứa file dung lượng lớn
- Mô hình AI được cung cấp qua Google Drive để đảm bảo tái lập nghiên cứu
- Người dùng chỉ cần tải model và chạy `run.bat`

---

## 📬 Liên hệ
Dự án phục vụ mục đích học tập và nghiên cứu khoa học.
