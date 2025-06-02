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

# Để tắt warning Streamlit liên quan đến seaborn/matplotlib
import warnings
warnings.filterwarnings('ignore')

def insight_page():
    st.title("Phân tích dữ liệu dự đoán bỏ học của sinh viên")

    @st.cache_data
    def load_data():
        data = pd.read_csv("data.csv", sep=';')
        data.columns = [x.replace(' ', '_') for x in data.columns]
        return data

    data = load_data()

    st.header("1. Thông tin bộ dữ liệu và kiểm tra dữ liệu thiếu, trùng lặp")
    st.write(f"Tập dữ liệu có {data.shape[0]} dòng và {data.shape[1]} cột.")
    st.write("Kiểm tra giá trị null trong dữ liệu:")
    st.write(data.isnull().sum())
    st.write("Kiểm tra dữ liệu trùng lặp:")
    duplicate = data[data.duplicated()]
    st.write(f"Số hàng trùng: {len(duplicate)}")
    if len(duplicate) > 0:
        st.dataframe(duplicate)
    else:
        st.write("Không có hàng trùng nào.")

    st.header("2. Phân phối biến mục tiêu (Target)")
    target_counts = data['Target'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    labels = target_counts.index.tolist()
    sizes = target_counts.values
    explode = [0 if l=='Graduate' else 0.1 for l in labels]
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)
    st.write("Nhận xét: Khoảng 49.9% tốt nghiệp, 32.1% bỏ học, 17.1% đang học.")

    st.header("3. Xác định và xử lý ngoại lai (Outliers)")
    # Chọn biến liên tục
    data_num = data.select_dtypes(include=['float64'])
    st.write("Biến liên tục trong dữ liệu:")
    st.write(data_num.columns.tolist())

    # Biểu đồ phân phối và boxplot
    for col in data_num.columns:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(data[col], kde=True, ax=axs[0], color='red')
        axs[0].set_title(f'Phân phối {col}')
        sns.boxplot(x=data[col], ax=axs[1], color='green')
        axs[1].set_title(f'Boxplot {col}')
        st.pyplot(fig)

    # Xử lý ngoại lai với IQR (chỉ với 4 biến điểm số theo notebook)
    cols_iqr = ['Previous_qualification_(grade)', 'Admission_grade',
                'Curricular_units_1st_sem_(grade)', 'Curricular_units_2nd_sem_(grade)']

    for col in cols_iqr:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        old_len = len(data)
        data = data[(data[col] >= Q1 - 3*IQR) & (data[col] <= Q3 + 3*IQR)]
        st.write(f"Loại bỏ ngoại lai cột {col}: giảm từ {old_len} xuống {len(data)} dòng")

    st.header("4. Mối tương quan giữa các biến liên tục")
    cor = data[cols_iqr].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cor, annot=True, cmap='CMRmap_r', ax=ax2)
    st.pyplot(fig2)
    st.write("Nhận xét: Có mối tương quan cao giữa điểm học kỳ 1 và 2, mối quan hệ trung bình giữa điểm nền tảng và điểm tuyển sinh.")

    st.header("5. Phân tích dữ liệu phân loại")
    # Danh sách biến phân loại đã lấy từ notebook
    cols_cat = ["Mother's_occupation", "Father's_occupation", 'Marital_status',
                'Application_order', 'Age_at_enrollment', "Father's_qualification",
                "Mother's_qualification", 'Tuition_fees_up_to_date', 'Gender',
                "Nacionality", 'Displaced', 'Debtor', "Scholarship_holder",
                'Application_mode', 'Course']

    st.write("Biểu đồ đếm các biến phân loại:")
    for col in cols_cat:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(x=col, data=data, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.header("6. Phân tích ảnh hưởng các biến phân loại lên Target")
    fig3, ax3 = plt.subplots(2, 3, figsize=(20, 10))
    sns.countplot(x='Application_order', hue='Target', data=data, palette='Set2', ax=ax3[0][0])
    sns.countplot(x='Tuition_fees_up_to_date', hue='Target', data=data, palette='Set2', ax=ax3[0][1])
    sns.countplot(x='Gender', hue='Target', data=data, palette='Set2', ax=ax3[0][2])
    sns.countplot(x='Displaced', hue='Target', data=data, palette='Set2', ax=ax3[1][0])
    sns.countplot(x='Debtor', hue='Target', data=data, palette='Set2', ax=ax3[1][1])
    sns.countplot(x='Scholarship_holder', hue='Target', data=data, palette='Set2', ax=ax3[1][2])
    st.pyplot(fig3)
    st.write("""
    Nhận xét:
    - Sinh viên không đóng học phí đúng hạn có nguy cơ bỏ học cao hơn.
    - Nam sinh có tỷ lệ bỏ học cao hơn nữ.
    - Sinh viên mắc nợ có nguy cơ bỏ học cao hơn.
    """)

    st.header("7. Kiểm định Chi-Square các biến đầu vào và Target")
    # Chuẩn bị biến đầu vào và biến mục tiêu
    X = data[[
        'Curricular_units_2nd_sem_(enrolled)', 'Scholarship_holder', 'Application_order',
        'Curricular_units_1st_sem_(evaluations)', 'Application_mode', 'Course', 'Nacionality',
        'Curricular_units_1st_sem_(without_evaluations)', 'International',
        'Curricular_units_2nd_sem_(without_evaluations)', 'Age_at_enrollment',
        'Curricular_units_1st_sem_(credited)', 'Curricular_units_2nd_sem_(credited)', 'Debtor',
        'Daytime/evening_attendance\t', 'Marital_status', 'Previous_qualification',
        "Mother's_qualification", 'Curricular_units_1st_sem_(approved)', "Mother's_occupation",
        'Gender', 'Displaced', 'Curricular_units_2nd_sem_(evaluations)', 'Tuition_fees_up_to_date',
        'Educational_special_needs', "Father's_qualification", 'Curricular_units_2nd_sem_(approved)',
        'Curricular_units_1st_sem_(enrolled)', "Father's_occupation"
    ]]
    y = data['Target']

    f_score = chi2(X, y)
    p_value = pd.Series(f_score[1], index=X.columns).sort_values()

    st.write("Bảng p-value kiểm định Chi-Square:")
    for feature, p in p_value.items():
        st.write(f"{feature:<45}: {p:.10f}")

    st.write("Các biến không có ý nghĩa thống kê (p ≥ 0.05):")
    irrelevant_cols = [i for i in p_value.index if p_value[i] >= 0.05]
    st.write(irrelevant_cols)

    st.header("8. Kiểm định ANOVA điểm số theo Target")
    le = LabelEncoder()
    data['Target'] = le.fit_transform(data['Target'])
    data_num = data.select_dtypes(include=['float64', 'int64'])
    data_num['Target'] = data['Target']

    score_cols = ['Previous_qualification_(grade)', 'Admission_grade',
                  'Curricular_units_1st_sem_(grade)', 'Curricular_units_2nd_sem_(grade)']

    p_values_anova = {}
    for col in score_cols:
        group0 = data_num[data_num['Target'] == 0][col]
        group1 = data_num[data_num['Target'] == 1][col]
        group2 = data_num[data_num['Target'] == 2][col]
        _, p_val = f_oneway(group0, group1, group2)
        p_values_anova[col] = p_val

    p_values_anova_s = pd.Series(p_values_anova).sort_values()
    st.write("Bảng p-value kiểm định ANOVA:")
    for feature, p in p_values_anova_s.items():
        st.write(f"{feature:<45}: {p:.10f}")

    st.write("Nhận xét: Tất cả biến điểm số đều có ảnh hưởng đáng kể đến Target (p < 0.05).")

    st.header("9. Biểu đồ phân phối điểm số theo Target")
    for col in score_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=data_num, x=col, hue='Target', kde=True, element="step", ax=ax)
        ax.set_title(f'Phân phối {col} theo Target')
        st.pyplot(fig)

    st.header("10. Phân tích ảnh hưởng hỗ trợ tài chính đến bỏ học")
    financial_vars = ['Tuition_fees_up_to_date', 'Scholarship_holder', 'Debtor']
    target_binary = (data['Target'] == 0).astype(int)

    p_values_fin = {}
    for feature in financial_vars:
        table = pd.crosstab(data[feature], target_binary)
        _, p, _, _ = chi2_contingency(table)
        p_values_fin[feature] = p

    p_fin_s = pd.Series(p_values_fin).sort_values()
    st.write("Bảng p-value kiểm định Chi-Square cho biến tài chính:")
    for feature, p in p_fin_s.items():
        st.write(f"{feature:<30}: {p:.10f}")

    st.write("Biểu đồ tỷ lệ bỏ học theo các yếu tố tài chính:")
    for col in financial_vars:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=col, hue='Target', data=data, ax=ax)
        ax.set_title(f'Tỷ lệ bỏ học theo {col}')
        st.pyplot(fig)

    st.header("11. Phân tích tương tác biến tài chính (Heatmap tỷ lệ bỏ học)")
    heatmap_data = data.groupby(['Tuition_fees_up_to_date', 'Scholarship_holder'])['Target'].apply(lambda x: (x == 0).mean()).reset_index()
    heatmap_pivot = heatmap_data.pivot(index="Tuition_fees_up_to_date", columns="Scholarship_holder", values="Target")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=0.5, ax=ax)
    ax.set_title('Tỷ lệ bỏ học theo Học phí và Học bổng')
    ax.set_xlabel('Học bổng (1: Có, 0: Không)')
    ax.set_ylabel('Đóng học phí đúng hạn (1: Có, 0: Không)')
    st.pyplot(fig)

    st.header("12. Dự đoán với các mô hình phân loại và so sánh")
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    X = data.drop('Target', axis=1)
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    classifiers = {
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(),
        'CatBoost': CatBoostClassifier(random_seed=350, iterations=500, verbose=False)
    }

    results = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=None)
        results[name] = acc * 100
        st.write(f"### {name}")
        st.write(f"- Accuracy: {acc:.4f}")
        st.write(f"- F1-score per class: {f1}")
        st.text(classification_report(y_test, y_pred))

    st.header("13. So sánh độ chính xác các mô hình")
    model_names = list(results.keys())
    accuracies = list(results.values())
    df_models = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Model', y='Accuracy', data=df_models, palette='gray', ax=ax)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.5, f'{height:.1f}%', ha='center')
    ax.set_ylim(0, 110)
    st.pyplot(fig)

    st.header("14. Ma trận nhầm lẫn của mô hình CatBoost")
    model_cb = classifiers['CatBoost']
    y_pred_cb = model_cb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_cb)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)
    st.write("""
    Nhận xét:
    - Mô hình phân biệt tốt lớp 'Tốt nghiệp'.
    - Gặp khó khi phân biệt 'Bỏ học' và 'Đang học'.
    """)

    st.header("15. Ảnh hưởng yếu tố xã hội đến tỷ lệ bỏ học")
    social_vars = ["Mother's_occupation", "Father's_occupation",
                "Mother's_qualification", "Father's_qualification",
                'Nacionality', 'Displaced']

    p_values_social = {}
    for feature in social_vars:
        table = pd.crosstab(data[feature], target_binary)
        _, p, _, _ = chi2_contingency(table)
        p_values_social[feature] = p

    p_social_s = pd.Series(p_values_social).sort_values()
    st.write("Bảng p-value cho biến xã hội:")
    for feature, p in p_social_s.items():
        st.write(f"{feature:<30}: {p:.10f}")

    irrelevant_social_cols = [f for f in p_social_s.index if p_social_s[f] >= 0.05]
    if irrelevant_social_cols:
        st.write("Biến xã hội không liên quan (p ≥ 0.05):", irrelevant_social_cols)
    else:
        st.write("Tất cả biến xã hội có ảnh hưởng đến khả năng bỏ học.")

    st.header("16. Yếu tố quan trọng theo mô hình CatBoost")
    importances = model_cb.get_feature_importance()
    feat_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    st.write("Top 15 đặc trưng ảnh hưởng nhiều nhất:")
    st.dataframe(feat_imp_df.head(15))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title("Top 15 yếu tố ảnh hưởng đến kết quả học tập (CatBoost)")
    st.pyplot(fig)

    st.write("""
    **Nhận xét chính:**
    1. Số môn được phê duyệt học kỳ 2, học kỳ 1 rất quan trọng.
    2. Ngành học ảnh hưởng rõ ràng đến nguy cơ bỏ học.
    3. Tuổi nhập học và trình độ cha mẹ cũng liên quan.
    4. Tình trạng tài chính và kinh tế xã hội là yếu tố góp phần.
    """)

