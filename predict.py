import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model 1 lần khi import
model = joblib.load('model.joblib')

def predict_page():
    st.header("Dự đoán Sinh viên Bỏ học")

    st.sidebar.markdown("### Chọn thông số sinh viên")

    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0

    DaytimeEveningAttendance = st.sidebar.number_input("Đi học buổi tối hoặc ban ngày (1 - có, 0 - không)", 0, 1)
    Displaced = st.sidebar.number_input("Sinh viên có di cư không? (1 - có, 0 - không)", 0, 1)
    Tuitionfeesuptodate = st.sidebar.number_input("Học phí đã đóng đầy đủ? (1 - có, 0 - không)", 0, 1)
    Gender = st.sidebar.number_input("Giới tính (1 - Nam, 0 - Nữ)", 0, 1)
    Scholarshipholder = st.sidebar.number_input("Có học bổng? (1 - có, 0 - không)", 0, 1)
    Ageatenrollment = st.sidebar.slider("Tuổi lúc nhập học", 0, 70, 20)
    Curricularunits1stsemgrade = st.sidebar.slider("Điểm học kỳ 1", 0, 20, 5)
    Curricularunits2ndsemgrade = st.sidebar.slider("Điểm học kỳ 2", 0, 20, 5)

    row = np.array([
        DaytimeEveningAttendance, Displaced, Tuitionfeesuptodate,
        Gender, Scholarshipholder, Ageatenrollment,
        Curricularunits1stsemgrade, Curricularunits2ndsemgrade
    ])

    columns = ['Attendance', 'Displaced', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
            'Age at enrollment', 'Grade semester 1', 'Grade semester 2']

    X = pd.DataFrame([row], columns=columns)
    prediction = model.predict(X)[0]

    if st.sidebar.button("Dự đoán"):
        st.subheader("Kết quả dự đoán")
        if prediction == 1:
            st.error("🚨 Có nguy cơ bỏ học 🥲")
            st.snow()
        else:
            st.success("🎓 Dự kiến tốt nghiệp 😘")
            st.balloons()

        st.subheader("Khuyến nghị hỗ trợ")
        recommend(prediction, Tuitionfeesuptodate, Curricularunits1stsemgrade, Curricularunits2ndsemgrade)

    # Nút Reset
    if st.sidebar.button("Reset"):
        st.session_state.reset_counter += 1
        st.experimental_rerun()

def recommend(prediction, tuition_up_to_date, grade_sem1, grade_sem2):
    if prediction != 1:
        st.markdown("Không có cảnh báo. Hãy giữ vững tinh thần học tập nhé!")
    else:
        if tuition_up_to_date == 0 and grade_sem2 < 4:
            st.markdown("""
            **Hỗ trợ tài chính:**
            [Chính sách hỗ trợ học phí - HUIT](https://huit.edu.vn/thong-bao/tai-chinh.html)

            **Tư vấn học tập:**
            [Phòng Công tác Sinh viên - HUIT](https://www.facebook.com/ctsv.huit/?locale=vi_VN)

            **Hỗ trợ học tập:**
            [Hỗ trợ học tập - HUIT](https://thuvien.huit.edu.vn/)
            """)
            with st.expander("Ghi chú"):
                st.markdown("""
                - Chính sách hỗ trợ học phí dành cho sinh viên có hoàn cảnh khó khăn.
                - Phòng CTSV hỗ trợ tư vấn học tập và xây dựng kế hoạch cá nhân
                - Hệ thống thư viện HUIT cung cấp tài liệu học tập và ôn thi.
                """)
        elif tuition_up_to_date == 0:
            st.markdown("**Hỗ trợ tài chính:** Xem chính sách tại [HUIT - Học bổng & Hỗ trợ](https://huit.edu.vn/thong-bao/tai-chinh.html)")
        elif grade_sem1 > 5 and grade_sem2 < 4:
            st.markdown("""
            **Tư vấn học tập:**
            [Phòng Công tác Sinh viên - HUIT](https://www.facebook.com/ctsv.huit/?locale=vi_VN)

            **Hỗ trợ học tập:**
            [Hỗ trợ học tập - HUIT](https://thuvien.huit.edu.vn/)
            """)
        elif grade_sem2 < 4:
            st.markdown("**Tư vấn học tập:** Đăng ký tư vấn tại [Phòng Công tác Sinh viên - HUIT](https://www.facebook.com/ctsv.huit/?locale=vi_VN)")
        else:
            st.markdown("Không xác định nguyên nhân rõ ràng. Liên hệ [Phòng Công tác Sinh viên - HUIT](https://www.facebook.com/ctsv.huit/?locale=vi_VN) để được hỗ trợ.")
