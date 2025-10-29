# Tên file: streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io # Dùng để xử lý file cho việc download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report, ConfusionMatrixDisplay
)

# ==========================================================
# 🔰 PHẦN LOGIC XỬ LÝ (Lấy từ code gốc của bạn)
# ==========================================================
# Đưa toàn bộ code phân tích vào một hàm
def run_analysis(file_mat, file_por, config):
    try:
        # --- Bước 1: Đọc và CHUẨN BỊ dữ liệu ---
        df_mat = pd.read_csv(file_mat, sep=';')
        df_por = pd.read_csv(file_por, sep=';')

        df = pd.concat([df_mat, df_por], ignore_index=True)
        
        # Chuyển điểm về thang 10
        df['G1'] = df['G1'] / 2.0
        df['G2'] = df['G2'] / 2.0
        df['G3'] = df['G3'] / 2.0

        # --- Bước 2: Mã hoá dữ liệu ---
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])
        
        # --- Bước 3: Chuẩn bị dữ liệu ---
        X = df.drop('G3', axis=1)
        y_reg = df['G3']

        # Lấy mốc đậu từ config
        passing_threshold = config['moc_diem_dau']
        y_clf = (y_reg >= passing_threshold).astype(int)

        # Chia dữ liệu (Lấy config)
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            X, y_reg, y_clf,
            test_size=config['ty_le_test'],
            random_state=config['seed'],
            stratify=y_clf
        )
        
        # --- PHẦN 1: MÔ HÌNH HỒI QUY ---
        model_reg = RandomForestRegressor(n_estimators=config['so_cay'], random_state=config['seed'])
        model_reg.fit(X_train, y_reg_train)
        y_reg_pred = model_reg.predict(X_test)
        mae = mean_absolute_error(y_reg_test, y_reg_pred)
        r2 = r2_score(y_reg_test, y_reg_pred)

        # --- PHẦN 2: MÔ HÌNH PHÂN LOẠI ---
        model_clf = RandomForestClassifier(n_estimators=config['so_cay'], random_state=config['seed'])
        model_clf.fit(X_train, y_clf_train)
        y_clf_pred = model_clf.predict(X_test)
        accuracy = accuracy_score(y_clf_test, y_clf_pred)
        report = classification_report(
            y_clf_test, y_clf_pred,
            target_names=[f'Trượt (<{passing_threshold})', f'Đậu (>= {passing_threshold})']
        )
        
        # --- Tạo ma trận nhầm lẫn ---
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_estimator(
            model_clf, X_test, y_clf_test,
            display_labels=['Trượt','Đậu'], cmap=plt.cm.Blues, ax=ax
        )
        plt.title(f"Ma trận nhầm lẫn (Mốc {passing_threshold})")
        
        # --- PHẦN 3: Tạo DataFrame kết quả ---
        compare_full = pd.DataFrame({
            'Điểm G3 Thực tế': y_reg_test.values,
            'Điểm Dự đoán ': np.round(y_reg_pred, 2),
            'Thực tế (0=Trượt, 1=Đậu)': y_clf_test.values,
            'Dự đoán (0=Trượt, 1=Đậu)': y_clf_pred
        })

        # --- Chuẩn bị kết quả trả về ---
        results = {
            'status': 'success',
            'total_rows': len(df),
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,
            'report': report,
            'figure': fig,
            'dataframe': compare_full
        }
        return results

    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# ==========================================================
# 🎨 PHẦN GIAO DIỆN (Streamlit)
# ==========================================================

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Dự đoán Kết quả Học tập (2 File)",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Ứng dụng Dự đoán Kết quả Học tập (Gộp 2 file)")

# --- Thanh bên (Sidebar) để chứa các nút điều khiển ---
with st.sidebar:
    st.header("⚙️ Thiết lập Phân tích")
    
    # 1. Tải file (Đã cập nhật thành 2 file)
    uploaded_file_mat = st.file_uploader("1. Tải file 'student-mat.csv'", type=["csv"])
    uploaded_file_por = st.file_uploader("2. Tải file 'student-por.csv'", type=["csv"])
    
    st.divider()

    # 2. Các thanh trượt
    moc_diem_dau = st.slider("3. Mốc điểm đậu (0-10):", 0.0, 10.0, 4.0, 0.5)
    
    # Cập nhật default của ty_le_test thành 0.2 (20%)
    ty_le_test = st.slider("4. Tỷ lệ test (0.1-0.9):", 0.1, 0.9, 0.2, 0.05) 
    
    so_cay = st.slider("5. Số cây (50-500):", 50, 500, 150, 50)
    seed = st.number_input("6. Seed ngẫu nhiên:", value=42)
    
    # 3. Nút chạy
    run_button = st.button("🚀 CHẠY PHÂN TÍCH", type="primary")

# --- Khu vực hiển thị chính ---
if run_button:
    # Kiểm tra xem cả 2 file đã được tải lên chưa
    if uploaded_file_mat is not None and uploaded_file_por is not None:
        
        with st.spinner('Đang gộp 2 file và phân tích dữ liệu...'):
            
            # Lấy các giá trị config
            config = {
                'moc_diem_dau': moc_diem_dau,
                'ty_le_test': ty_le_test,
                'so_cay': so_cay,
                'seed': seed,
            }
            
            # Chạy phân tích
            results = run_analysis(uploaded_file_mat, uploaded_file_por, config)
        
        if results['status'] == 'success':
            st.success(f"🎉 Phân tích hoàn tất! (Đã xử lý {results['total_rows']} dòng dữ liệu)")
            
            # Hiển thị kết quả theo 2 cột
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Kết quả Hồi quy (Dự đoán điểm)")
                met1, met2 = st.columns(2)
                met1.metric("MAE (Sai số)", f"{results['mae']*100:.2f}%")
                met2.metric("R² (Độ chính xác)", f"{results['r2']*100:.2f}%")
                
                st.subheader("🎯 Kết quả Phân loại (Đậu/Trượt)")
                st.metric("Độ chính xác", f"{results['accuracy']*100:.2f}%")
                
                # Hiển thị Báo cáo
                st.text("Báo cáo chi tiết:")
                st.text(results['report'])

            with col2:
                st.subheader("📈 Ma trận nhầm lẫn")
                st.pyplot(results['figure'])

            st.divider()
            
            # Hiển thị bảng kết quả
            st.subheader("📋 So sánh kết quả chi tiết")
            st.dataframe(results['dataframe'])
            
            # --- Tạo nút Tải xuống ---
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                results['dataframe'].to_excel(writer, index=False, sheet_name='KetQuaDuDoan')
            
            st.download_button(
                label="📥 Tải file Excel kết quả",
                data=output.getvalue(),
                file_name="KetQuaDuDoanHocTap_Gop.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        else:
            st.error(f"❌ Lỗi: {results['message']}")
            st.warning("Vui lòng kiểm tra lại file CSV (phải có dấu phân cách là ';')")
            
    else:
        # Nếu chưa tải đủ file
        st.warning("Vui lòng tải lên CẢ HAI file 'student-mat.csv' và 'student-por.csv' ở thanh bên trái.")
else:
    # Hướng dẫn khi chưa nhấn nút
    st.info("Chào mừng! Vui lòng tải lên cả 2 file, thiết lập các tham số, sau đó nhấn 'CHẠY PHÂN TÍCH'.")
