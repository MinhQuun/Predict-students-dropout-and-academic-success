# 🎓 Dự đoán Tỷ Lệ Bỏ Học Của Sinh Viên Bằng Machine Learning

## 📌 Mục Tiêu Dự Án

Dự án nhằm **giảm tỷ lệ bỏ học và tăng khả năng tốt nghiệp** bằng cách sử dụng các kỹ thuật **học máy** để phát hiện sớm sinh viên có nguy cơ bỏ học. Qua đó, giúp nhà trường có thể đưa ra các **biện pháp hỗ trợ kịp thời**.

---

## 📊 Nội Dung Phân Tích

Chương trình thực hiện đầy đủ các bước phân tích dữ liệu và huấn luyện mô hình dự đoán:

- ✅ **Xử lý dữ liệu thiếu và bất thường** trong tập dữ liệu sinh viên.
- ✅ **Tìm mối quan hệ giữa các yếu tố** học tập, tài chính và xã hội.
- ✅ **Phân tích các yếu tố ảnh hưởng đến quyết định bỏ học** của sinh viên.
- ✅ **Phân tích mối quan hệ giữa điểm số và tỷ lệ bỏ học** (ANOVA).
- ✅ **Dự đoán kết quả học tập của sinh viên** dựa trên các yếu tố đầu vào.
- ✅ **Sử dụng các mô hình phân lớp** để phân loại sinh viên vào 3 nhóm:
  - `0`: Bỏ học (Dropout)
  - `1`: Đang học (Enrolled)
  - `2`: Tốt nghiệp (Graduate)
- ✅ **So sánh hiệu quả giữa các mô hình học máy**:
  - Naive Bayes
  - Decision Tree
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
- ✅ **Phân tích mối quan hệ giữa hỗ trợ tài chính và tỷ lệ bỏ học**:
  - Tình trạng học bổng
  - Việc đóng học phí đúng hạn
  - Tình trạng nợ
- ✅ **Tìm hiểu ảnh hưởng của yếu tố xã hội**:
  - Nghề nghiệp và học vấn của phụ huynh
  - Quốc tịch, tình trạng di cư
- ✅ **Xác định các yếu tố chính làm tăng nguy cơ bỏ học** bằng mô hình CatBoost.
- ✅ **Đề xuất các biện pháp giảm thiểu rủi ro bỏ học** dựa trên phân tích dữ liệu.

---

## 🌐 Giao Diện Web

Dự án đã được triển khai với **Streamlit**, bạn có thể dễ dàng:

- Trực quan hóa dữ liệu bằng biểu đồ tương tác.
- Khám phá phân tích chuyên sâu tại trang `insight_page`.
- Trực tiếp dự đoán và đánh giá hiệu quả mô hình học máy ngay trên trình duyệt.

---

## 🛠️ Hướng Dẫn Cài Đặt

```bash
git clone https://github.com/tenban/project-dropout-prediction.git
cd project-dropout-prediction
pip install -r requirements.txt
streamlit run app.py
