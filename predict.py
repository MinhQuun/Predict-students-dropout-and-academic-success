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
            st.error("Có nguy cơ bỏ học :(")
        else:
            st.success("Dự kiến tốt nghiệp :D")
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
            [Chính sách hỗ trợ học phí - Bộ Giáo dục & Đào tạo](https://moet.gov.vn/)

            **Tư vấn học tập:**
            [Phòng tư vấn sinh viên ĐHCN](https://www.vku.udn.vn/)

            **Hỗ trợ học tập:**
            [Học trực tuyến tại Kyna.vn](https://kyna.vn/)
            """)
            with st.expander("Ghi chú"):
                st.markdown("""
                Chính sách hỗ trợ học phí dành cho sinh viên có hoàn cảnh khó khăn.
                Phòng tư vấn giúp sinh viên xây dựng kế hoạch học tập phù hợp.
                Các nền tảng học trực tuyến hỗ trợ bổ sung kiến thức.
                """)
        elif tuition_up_to_date == 0:
            st.markdown("**Hỗ trợ tài chính:** Xem chính sách hỗ trợ học phí tại trang chính thức của trường và Bộ Giáo dục.")
        elif grade_sem1 > 5 and grade_sem2 < 4:
            st.markdown("""
            **Tư vấn học tập:**
            [Phòng tư vấn sinh viên ĐHCN](https://www.vku.udn.vn/)

            **Hỗ trợ học tập:**
            [Học trực tuyến tại Kyna.vn](https://kyna.vn/)
            """)
        elif grade_sem2 < 4:
            st.markdown("**Tư vấn học tập:** Hãy liên hệ phòng tư vấn để được hỗ trợ cải thiện kết quả học tập.")
        else:
            st.markdown("Không xác định nguyên nhân rõ ràng. Hãy theo dõi và nỗ lực hơn nữa.")
