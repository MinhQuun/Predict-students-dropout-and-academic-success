import streamlit as st
from predict import predict_page
from insight import insight_page
from member import member_page

st.set_page_config(page_title="Student Dropout System", page_icon=":student:")

st.title("🎓 Dự đoán Sinh viên Bỏ học và Ra trường - Đại học Công thương :student:")

page = st.sidebar.selectbox("Chọn trang", ["Member", "Insight", "Predict"])

if page == "Member":
    member_page()
elif page == "Insight":
    insight_page()
else:
    predict_page()
