# AI_Nông_Nghiệp

Dự án nghiên cứu khoa học ứng dụng Trí tuệ nhân tạo (AI) trong việc xử lý, truy xuất và phân tích tài liệu tiếng Việt, phục vụ lĩnh vực nông nghiệp và các bài toán liên quan đến biến đổi khí hậu.

---

## Giới thiệu
Dự án xây dựng một hệ thống AI hỗ trợ nghiên cứu nông nghiệp dựa trên:
- Mô hình embedding tài liệu tiếng Việt
- Cơ chế truy xuất tăng cường (Retrieval-Augmented Generation – RAG)
- Vector Database phục vụ tìm kiếm ngữ nghĩa

---

## Lưu ý quan trọng về mô hình AI
Do **giới hạn dung lượng file của GitHub**, các **mô hình AI KHÔNG được lưu trữ trong repository này**.

Người dùng **bắt buộc phải tải mô hình từ Google Drive** trước khi chạy project.

---

## Tải mô hình (BẮT BUỘC)

### Vietnamese Document Embedding Model
- Nơi lưu trữ: **Google Drive**
- Link tải:  
 **https://drive.google.com/file/d/1IGYuMceO5dA_HncZmXtoEalpYd8Ti2Pv/view?usp=sharing**

### Cấu trúc sau khi tải và giải nén
Sau khi tải xong, giải nén và đặt mô hình đúng đường dẫn sau:

```
AI_NongNghiep/
│── models/
│   └── s.vietnamese-document-embedding/
│       └── model.safetensors
```

## Lấy key API Gemini:
B1: Vào google AI studio

B2: Lấy API key về

B3: Dán vào file ".env"


## Cách chạy project:
B1: Đứng ở folder AI_NongNghiep

B2: Cài đặt và vào môi trường .venv

B3: Chạy lệnh sau để tải những thư viện cần thiết:
```
pip install -r requirements.txt
```

B4: Chạy lệnh sau và đợi chương trình thực thi xong (Sẽ tạo ra file vectorstore):
```
python src\prepare_vector_db.py
```

B5: Chạy lệnh sau để khởi chạy hệ thống:
```
streamlit run app.py
```

## Bối cảnh nghiên cứu khoa học
Dự án được thực hiện trong khuôn khổ **nghiên cứu khoa học sinh viên**, tập trung vào:
- AI trong nông nghiệp
- Xử lý ngôn ngữ tự nhiên (NLP) tiếng Việt
- Embedding & Vector Search
- Retrieval-Augmented Generation (RAG)
- Hướng tới khả năng mở rộng và giải thích mô hình (XAI)

---

## Ghi chú cho giảng viên & người đánh giá
- Repository chỉ chứa **mã nguồn**, không chứa file dung lượng lớn
- Mô hình AI được cung cấp qua Google Drive để đảm bảo tái lập nghiên cứu

