import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from scipy.stats import chi2_contingency, f_oneway
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_data():
    data = pd.read_csv("data.csv", sep=';')
    data.columns = [x.replace(' ', '_') for x in data.columns]
    return data

def section_1(data):
    st.subheader("1. Xử lý dữ liệu thiếu và bất thường trong tập dữ liệu sinh viên")

    st.markdown("### Làm sạch dữ liệu")
    st.write(f"Tập dữ liệu có {data.shape[0]} dòng và {data.shape[1]} cột.")

    st.write("**Kiểm tra giá trị thiếu:**")
    st.dataframe(data.isnull().sum())
    st.markdown("*Không có giá trị NULL nào trong tập dữ liệu.*")

    st.write("**Kiểm tra các giá trị trùng lặp:**")
    duplicate = data[data.duplicated()]
    st.write(f"Số dòng trùng lặp: {len(duplicate)}")
    if not duplicate.empty:
        st.dataframe(duplicate)
    st.markdown("*Không có hàng trùng lặp nào trong tập dữ liệu.*")

    st.markdown("### Thăm dò dữ liệu")
    st.write("**Thông tin cột dữ liệu:**")
    st.dataframe(data.dtypes)
    st.markdown("*Có thể thấy được có 7 kiểu dữ liệu số thực, 29 số nguyên và 1 kiểu dữ liệu object.*")

    st.write("**Phân phối biến Target:**")
    labels = ['Graduate', 'Dropout', 'Enrolled']
    sizes = [data.Target[data['Target'] == 'Graduate'].count(),
            data.Target[data['Target'] == 'Dropout'].count(),
            data.Target[data['Target'] == 'Enrolled'].count()]
    explode = (0, 0.1, 0.1)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title("Tỷ lệ của biến Target", size=12)
    st.pyplot(fig1)
    st.markdown("*Khoảng 49,9% sinh viên đã tốt nghiệp, 32,1% bỏ học và 17,1% đang theo học một khóa học khác.*")

    st.markdown("### Xác định ngoại lai")
    data_num = data.select_dtypes(include=['float64'])
    for col in data_num.columns:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(data[col], kde=True, ax=axs[0], color='red')
        axs[0].set_title(f'Phân phối {col}')
        sns.boxplot(x=data[col], ax=axs[1], color='green')
        axs[1].set_title(f'Boxplot {col}')
        st.pyplot(fig)
    st.markdown("*Hầu hết các đặc trưng đều chứa các giá trị ngoại lai, ngoại trừ tỷ lệ thất nghiệp, lạm phát và GDP.*")

    st.markdown("### Xử lý ngoại lai bằng IQR")
    cols_iqr = ['Previous_qualification_(grade)', 'Admission_grade',
                'Curricular_units_1st_sem_(grade)', 'Curricular_units_2nd_sem_(grade)']
    for col in cols_iqr:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        old_len = len(data)
        data = data[(data[col] >= Q1 - 3*IQR) & (data[col] <= Q3 + 3*IQR)]
        st.write(f"Loại bỏ ngoại lai trong {col}: {old_len} -> {len(data)} dòng")

    st.write("**Thống kê sau khi xử lý ngoại lệ:**")
    st.dataframe(data[cols_iqr].describe())

    fig, axs = plt.subplots(4, 1, figsize=(18, 20))
    sns.boxplot(x=data['Previous_qualification_(grade)'], ax=axs[0], palette='BuGn')
    axs[0].set_title('Previous Qualification Grade', fontsize=14, pad=10)
    sns.boxplot(x=data['Admission_grade'], ax=axs[1], palette='BuGn')
    axs[1].set_title('Admission Grade', fontsize=14, pad=10)
    sns.boxplot(x=data['Curricular_units_1st_sem_(grade)'], ax=axs[2], palette='BuGn')
    axs[2].set_title('1st Semester Grade', fontsize=14, pad=10)
    sns.boxplot(x=data['Curricular_units_2nd_sem_(grade)'], ax=axs[3], palette='BuGn')
    axs[3].set_title('2nd Semester Grade', fontsize=14, pad=10)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("*Đã xử lý ngoại lệ bằng phương pháp IQR và trực quan hóa lại bằng biểu đồ hộp.*")
    return data

def section_2(data):
    st.subheader("2. Tìm mối quan hệ giữa các yếu tố")

    st.markdown("### Mối tương quan giữa các biến số liên tục")
    data_num = data.select_dtypes(include=['float64'])
    cor = data_num.corr()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r, ax=ax)
    ax.set_title('Mối quan hệ giữa các biến ngẫu nhiên liên tục')
    st.pyplot(fig)

    st.markdown("""
    Từ biểu đồ trên, có thể quan sát thấy:
    - `Curricular_units_1st_sem_(grade)` và `Curricular_units_2nd_sem_(grade)` có mối tương quan chặt chẽ.
    - `Previous_qualification_(grade)` và `Admission_grade` có mối tương quan trung bình.
    - GDP, tỷ lệ lạm phát và tỷ lệ thất nghiệp có mối quan hệ tiêu cực với các yếu tố khác.
    """)

def section_3(data):
    st.subheader("3. Phân tích tác động của từng biến phân loại lên biến Target")

    st.markdown("""
    Phân tích sự phân phối của biến mục tiêu `Target` theo các yếu tố phân loại như:
    - Hình thức học (ban ngày/tối)
    - Tình trạng đóng học phí đúng hạn
    - Giới tính
    - Tình trạng di cư
    - Nợ học phí
    - Tình trạng nhận học bổng
    """)

    fig, ax = plt.subplots(2, 3, figsize=(40, 30))
    sns.countplot(x='Daytime/evening_attendance\t', hue='Target', data=data, palette='Set2', ax=ax[0][0])
    sns.countplot(x='Tuition_fees_up_to_date', hue='Target', data=data, palette='Set2', ax=ax[0][1])
    sns.countplot(x='Gender', hue='Target', data=data, palette='Set2', ax=ax[0][2])
    sns.countplot(x='Displaced', hue='Target', data=data, palette='Set2', ax=ax[1][0])
    sns.countplot(x='Debtor', hue='Target', data=data, palette='Set2', ax=ax[1][1])
    sns.countplot(x='Scholarship_holder', hue='Target', data=data, palette='Set2', ax=ax[1][2])
    st.pyplot(fig)

    st.markdown("""
    **Nhận xét:**

    1. Những sinh viên **không nộp học phí đúng hạn** có tỷ lệ bỏ học cao rõ rệt.
    2. **Nam sinh** có xu hướng bỏ học nhiều hơn nữ sinh.
    3. Sinh viên **mắc nợ với trường** có tỷ lệ bỏ học cao hơn sinh viên không nợ.
    4. Sinh viên **không nhận học bổng** có tỷ lệ bỏ học cao hơn so với sinh viên nhận học bổng.
    5. **Sinh viên học ban ngày** có tỷ lệ tốt nghiệp cao hơn đáng kể so với học buổi tối.
    """)


def section_4(data):
    st.subheader("4. Phân tích mối quan hệ giữa điểm số và tỷ lệ bỏ học")

    st.markdown("### Xác định các biến điểm số")
    score_cols = ['Previous_qualification_(grade)', 'Admission_grade',
                'Curricular_units_1st_sem_(grade)', 'Curricular_units_2nd_sem_(grade)']
    st.markdown("""
    Trong dataset, các biến liên quan đến điểm số gồm:

    - `Previous_qualification_(grade)` (0-200)
    - `Admission_grade` (0-200)
    - `Curricular_units_1st_sem_(grade)` (0-20)
    - `Curricular_units_2nd_sem_(grade)` (0-20)
    """)

    le = LabelEncoder()
    data['Target'] = le.fit_transform(data['Target'])
    data_num = data.select_dtypes(include=['float64', 'int64'])
    data_num['Target'] = data['Target']

    st.markdown("### Kiểm định ANOVA")
    st.markdown("""
    - H₀: Không có sự khác biệt trung bình điểm số giữa các nhóm Target (không ảnh hưởng).
    - H₁: Có sự khác biệt trung bình điểm số giữa các nhóm Target (có ảnh hưởng).

    *Nếu p-value < 0.05 → Bác bỏ H₀ → Điểm số ảnh hưởng đến khả năng bỏ học.*
    """)

    p_values = {}
    for col in score_cols:
        group0 = data_num[data_num['Target'] == 0][col]
        group1 = data_num[data_num['Target'] == 1][col]
        group2 = data_num[data_num['Target'] == 2][col]
        _, p_val = f_oneway(group0, group1, group2)
        p_values[col] = p_val

    p_series = pd.Series(p_values).sort_values()
    st.write("**📋 Bảng p-value cho từng biến điểm số:**")
    for feature, p in p_series.items():
        st.write(f"{feature:<45}: {p:.10f}")

    st.markdown("""
    *Tất cả các biến đều có p-value < 0.05 → Điểm số có ảnh hưởng rõ rệt đến khả năng bỏ học.*
    """)

    st.markdown("### Phân tích phân phối điểm số theo Target")
    for col in score_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=data_num, x=col, hue='Target', kde=True, element="step", ax=ax)
        ax.set_title(f'Phân phối {col} theo Target')
        st.pyplot(fig)

        if col == 'Previous_qualification_(grade)':
            st.markdown("""
            **Nhận xét:**
            1.   Sinh viên tốt nghiệp có điểm nền tảng cao hơn sinh viên bỏ học rõ rệt, chủ yếu tập trung trên 130 điểm.
            2.   Sinh viên bỏ học có điểm nền tảng thấp hơn, đa phần tập trung dưới 130 điểm.
            3.   Điểm số nền tảng càng cao thì khả năng tốt nghiệp càng cao.
            """)
        elif col == 'Admission_grade':
            st.markdown("""
            **Nhận xét:**
            1. Sinh viên tốt nghiệp thường có điểm đầu vào cao hơn sinh viên bỏ học.
            2. Tuy nhiên, vẫn có sự chồng lấn — điểm cao vẫn có thể bỏ học.
            3. Điểm đầu vào là yếu tố quan trọng nhưng không quyết định duy nhất.
            """)
        elif col == 'Curricular_units_1st_sem_(grade)':
            st.markdown("""
            **Nhận xét:**
            1. Trên 80% sinh viên tốt nghiệp có điểm học kỳ 1 > 12.
            2. Khoảng 65% sinh viên bỏ học có điểm từ 10–12.
            3. Điểm dưới 11 có nguy cơ bỏ học cao.
            """)
        elif col == 'Curricular_units_2nd_sem_(grade)':
            st.markdown("""
            **Nhận xét:**
            1. Hơn 90% sinh viên tốt nghiệp có điểm học kỳ 2 trên 12.
            2. Sinh viên bỏ học tập trung ở mức 10–12, rất ít sinh viên bỏ học đạt trên 13 điểm.
            """)

    st.markdown("### Phân tích chi tiết trung bình theo nhóm")
    means = data_num.groupby('Target')[score_cols].mean()
    st.dataframe(means)
    st.markdown("""
    **Nhận xét:**
    - Trung bình điểm của nhóm sinh viên tốt nghiệp cao hơn rõ rệt so với nhóm sinh viên bỏ học.
    """)

def section_5(data):
    st.subheader("5. Phân tích mối quan hệ giữa hỗ trợ tài chính và tỷ lệ bỏ học")

    st.markdown("""
    ### Xác định các biến tài chính
    - `Tuition_fees_up_to_date`: Đóng học phí đúng hạn (0: Không, 1: Có)
    - `Scholarship_holder`: Nhận học bổng (0: Không, 1: Có)
    - `Debtor`: Có nợ với trường (0: Không, 1: Có)

    ### Kiểm định Chi-Square
    - H₀: Không có mối quan hệ giữa biến tài chính và khả năng bỏ học.
    - H₁: Có mối quan hệ giữa biến tài chính và khả năng bỏ học.
                
    *Nếu p-value < 0.05 → bác bỏ H₀ → biến tài chính có ảnh hưởng đến bỏ học.*
    """)

    financial_vars = ['Tuition_fees_up_to_date', 'Scholarship_holder', 'Debtor']
    target_binary = (data['Target'] == 0).astype(int)

    p_values = {}
    for feature in financial_vars:
        table = pd.crosstab(data[feature], target_binary)
        _, p, _, _ = chi2_contingency(table)
        p_values[feature] = p

    p_series = pd.Series(p_values).sort_values()
    st.write("**📋 Bảng p-value cho từng biến tài chính:**")
    for feature, p in p_series.items():
        st.write(f"{feature:<30}: {p:.10f}")

    st.markdown("""
    *Tất cả các biến đều có p-value < 0.05 → Tài chính có ảnh hưởng rõ rệt đến khả năng bỏ học.*
    """)

    st.markdown("### Phân tích đơn biến")
    for col in financial_vars:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=col, hue='Target', data=data, ax=ax)
        ax.set_title(f'Tỷ lệ bỏ học theo {col}')
        st.pyplot(fig)

        if col == 'Tuition_fees_up_to_date':
            st.markdown("""
            **Nhận xét:**
            1. Sinh viên không đóng học phí đúng hạn có tỷ lệ bỏ học >60%.
            2. Trong khi sinh viên đóng đúng hạn chỉ <20% bỏ học.
            """)
        elif col == 'Scholarship_holder':
            st.markdown("""
            **Nhận xét:**
            1. Gần 35% sinh viên không có học bổng bỏ học.
            2. Trong khi chỉ khoảng 10% sinh viên có học bổng bỏ học.
            """)
        elif col == 'Debtor':
            st.markdown("""
            **Nhận xét:**
            1. Sinh viên mắc nợ có tỷ lệ bỏ học >50%.
            2. Không nợ thì tỷ lệ chỉ khoảng 30%.
            """)

    st.markdown("### Phân tích đa biến (kết hợp các yếu tố)")
    st.write("Tỷ lệ bỏ học khi kết hợp 3 yếu tố tài chính:")
    crosstab = pd.crosstab(
        index=[data['Tuition_fees_up_to_date'], data['Scholarship_holder'], data['Debtor']],
        columns=data['Target'],
        normalize='index'
    )
    dropout_rates = crosstab.iloc[:, 0].sort_values(ascending=False) * 100
    dropout_df = pd.DataFrame(dropout_rates).reset_index()
    dropout_df.columns = ['Đóng học phí', 'Học bổng', 'Nợ', 'Tỷ lệ bỏ học']
    st.dataframe(dropout_df)

    st.markdown("""
    **Nhận xét:**
    1. Không đóng học phí + không học bổng + có nợ → nguy cơ bỏ học cao nhất.
    """)

    st.markdown("### Trực quan hóa với Heatmap")
    heatmap_data = data.groupby(['Tuition_fees_up_to_date', 'Scholarship_holder'])['Target'].apply(lambda x: (x == 0).mean()).reset_index()
    heatmap_pivot = heatmap_data.pivot(index="Tuition_fees_up_to_date", columns="Scholarship_holder", values="Target")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=0.5, ax=ax)
    ax.set_title('Tỷ lệ bỏ học theo Học phí và Học bổng')
    ax.set_xlabel('Học bổng (1: Có, 0: Không)')
    ax.set_ylabel('Đóng học phí đúng hạn (1: Có, 0: Không)')
    st.pyplot(fig)

    st.markdown("""
    **Nhận xét:**
    1. Không đóng học phí + không học bổng → Tỷ lệ bỏ học cao nhất.
    2. Đóng học phí + có học bổng → Tỷ lệ bỏ học thấp nhất.
    """)

def section_6(data):
    st.subheader("6. Tìm hiểu sự ảnh hưởng của môi trường xã hội đến kết quả học tập của sinh viên")

    st.markdown("""
    ### Xác định các biến môi trường xã hội
    Các yếu tố liên quan gồm:
    - Nghề nghiệp cha mẹ
    - Trình độ học vấn cha mẹ
    - Quốc tịch
    - Tình trạng di cư 
    
    ### Giả thuyết kiểm định:
    - H₀: Không có mối liên hệ giữa biến xã hội và khả năng bỏ học.
    - H₁: Có mối liên hệ giữa biến xã hội và khả năng bỏ học.
                
    *Nếu p-value < 0.05 → bác bỏ H₀ → Biến xã hội có ảnh hưởng đến khả năng bỏ học.*
    """)

    social_vars = [
        "Mother's_occupation", "Father's_occupation",
        "Mother's_qualification", "Father's_qualification",
        'Nacionality', 'Displaced'
    ]
    target_binary = (data['Target'] == 0).astype(int)

    p_values_social = {}
    for feature in social_vars:
        table = pd.crosstab(data[feature], target_binary)
        _, p, _, _ = chi2_contingency(table)
        p_values_social[feature] = p

    p_series = pd.Series(p_values_social).sort_values()
    st.write("**📋 Bảng p-value cho từng biến xã hội:**")
    for feature, p in p_series.items():
        st.write(f"{feature:<30}: {p:.10f}")

    st.markdown("### Đánh giá ý nghĩa thống kê")
    significance_level = 0.05
    irrelevant_cols = []
    for feature in p_series.index:
        if p_series[feature] <= significance_level:
            st.write(f"{feature:<30} ==> ❌ Bác bỏ H₀ (có liên quan đến khả năng bỏ học)")
        else:
            st.write(f"{feature:<30} ==> ✅ Không bác bỏ H₀ (không liên quan)")
            irrelevant_cols.append(feature)

    st.markdown("**Các biến xã hội không có ý nghĩa thống kê và có thể loại bỏ nếu cần:**")
    if irrelevant_cols:
        st.write(irrelevant_cols)
    else:
        st.write("Tất cả các biến xã hội đều có ảnh hưởng đến khả năng bỏ học.")
    
    st.markdown("### Biểu đồ trực quan từng biến xã hội")
    for feature in social_vars:
        if feature != 'Nacionality':
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.countplot(x=feature, hue='Target', data=data, ax=ax)
            plt.xticks(rotation=45)
            ax.set_title(f'Tỷ lệ bỏ học theo {feature}')
            st.pyplot(fig)  

    st.markdown("""
    **Nhận xét tổng quan:**
    - Trình độ học vấn và nghề nghiệp của cha mẹ có liên quan đến tỷ lệ bỏ học.
    - Sinh viên không di cư có khả năng tốt nghiệp cao hơn.
    """)

def section_7():
    st.subheader("7. Đề xuất các biện pháp giảm thiểu tỷ lệ bỏ học")

    st.markdown("""
    ### Dựa trên kết quả phân tích ở các phần trước, một số biện pháp cụ thể có thể được đề xuất như sau:

    #### 1. Hỗ trợ tài chính
    - **Tăng cường cấp học bổng** cho sinh viên có nguy cơ cao (dựa vào điểm đầu vào và hoàn cảnh).
    - **Nhắc nhở và hỗ trợ đóng học phí đúng hạn**, đồng thời thiết lập các chính sách giãn nợ cho sinh viên có khó khăn tài chính.

    #### 2. Hỗ trợ học tập
    - **Tư vấn học thuật** và **hỗ trợ thêm** cho sinh viên có điểm thấp trong năm đầu tiên.
    - Tổ chức các buổi **kèm cặp học tập** (mentoring) giữa sinh viên năm trên và năm dưới.

    #### 3. Quan tâm yếu tố xã hội và cá nhân
    - Theo dõi sinh viên thuộc nhóm có **gia đình ít học**, **cha mẹ thất nghiệp** hoặc **di cư** để có chính sách hỗ trợ riêng.
    - Mở rộng dịch vụ **tham vấn tâm lý và hướng nghiệp** tại trường.

    #### 4. Theo dõi sớm và cảnh báo sớm
    - Xây dựng hệ thống **phát hiện sớm sinh viên có nguy cơ bỏ học** dựa trên mô hình học máy.
    - Kết hợp các yếu tố: điểm học kỳ, học phí, học bổng, nợ, và dữ liệu xã hội để **xây dựng cảnh báo rủi ro bỏ học**.

    #### 5. Chính sách giáo dục và quản lý
    - Cải thiện **chính sách linh hoạt học tập** (cho phép bảo lưu, chuyển đổi ngành).
    - Tăng cường **quản lý tương tác sinh viên - nhà trường**, phản hồi sớm từ sinh viên về khó khăn.

    ### Kết luận:
    Việc giảm tỷ lệ bỏ học cần kết hợp nhiều yếu tố: tài chính, học tập, tâm lý và chính sách. Phân tích dữ liệu đóng vai trò then chốt giúp xác định nhóm sinh viên có nguy cơ cao và đưa ra biện pháp hỗ trợ kịp thời.
    """)

def insight_page():
    st.title("Khám Phá Dữ Liệu Sinh Viên")
    st.markdown("**Trang này giúp bạn hiểu rõ hơn về dữ liệu sinh viên và các yếu tố ảnh hưởng đến việc bỏ học.**")

    data = load_data()
    data = section_1(data)
    section_2(data)
    section_3(data)
    section_4(data)
    section_5(data)
    section_6(data)
    section_7()

    