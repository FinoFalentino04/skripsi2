import streamlit as st
import numpy as np
import joblib

# 1. Mengatur konfigurasi halaman web
st.set_page_config(page_title="Prediksi Risiko Kehamilan K-NN", layout="centered")

# 2. Judul Aplikasi 
st.title("Prediksi Kehamilan Risiko Tinggi Berdasarkan Usia dan Status Gizi")
st.subheader("Menggunakan Algoritma K-Nearest Neighbor (K-NN) - Kecamatan Cisaat")

st.write("---")

# 3. Memuat Model K-NN dan Z-Score Scaler
# Menggunakan st.cache_resource agar model tidak di-load berulang kali dari memori
@st.cache_resource
def load_model_and_scaler():
    try:
        # Pastikan Anda telah mengunduh knn_model.pkl dan scaler.pkl terbaru 
        # dari Colab setelah melakukan perbaikan aturan pelabelan medis tadi
        model_knn = joblib.load('knn_model_cisaat.pkl')
        model_scaler = joblib.load('scaler_cisaat.pkl')
        return model_knn, model_scaler
    except FileNotFoundError:
        return None, None

knn, scaler = load_model_and_scaler()

if knn is None or scaler is None:
    st.error("⚠️ File 'knn_model.pkl' atau 'scaler.pkl' tidak ditemukan. Pastikan kedua file tersebut sudah diunggah dan berada di direktori yang sama dengan app.py.")
else:
    # 4. Membuat Form Input Data Pasien
    st.write("### Masukkan Data Warga / Pasien Baru")
    
    col1, col2 = st.columns(2)
    
    # Urutan input harus sama persis dengan saat training di Colab:
    # ['Usia', 'Berat Badan Sebelum Hamil (kg)', 'IMT', 'LiLA']
    with col1:
        usia = st.number_input("Usia (Tahun)", min_value=10, max_value=60, value=25, step=1)
        bb = st.number_input("Berat Badan Sebelum Hamil (kg)", min_value=30.0, max_value=150.0, value=50.0, step=0.1)
        
    with col2:
        imt = st.number_input("Indeks Massa Tubuh (IMT)", min_value=10.0, max_value=50.0, value=21.5, step=0.1)
        lila = st.number_input("Lingkar Lengan Atas / LiLA (cm)", min_value=15.0, max_value=40.0, value=24.0, step=0.1)
        
    st.write("---")
    
    # 5. Tombol Prediksi dan Eksekusi Logika K-NN
    if st.button("Lakukan Prediksi Risiko", type="primary", use_container_width=True):
        
        # Membentuk array dari input pengguna
        data_input = np.array([[usia, bb, imt, lila]])
        
        # WAJIB: Normalisasi data input menggunakan Z-Score Scaler dari data latih
        data_input_scaled = scaler.transform(data_input)
        
        # Prediksi menggunakan model K-NN
        prediksi = knn.predict(data_input_scaled)
        
        # 6. Menampilkan Hasil Klasifikasi beserta penjelasan medisnya
        st.write("### Hasil Prediksi K-NN:")
        
        if prediksi[0] == 1:
            st.error("⚠️ **Kategori: RISIKO TINGGI**")
            st.write("Sistem memprediksi pasien masuk dalam kategori Risiko Tinggi berdasarkan perhitungan jarak K-NN. Hal ini dapat dipicu oleh deteksi kesamaan pola pada faktor usia ekstrem (< 20 atau > 35 tahun), indikasi Kurang Energi Kronis (LiLA < 23.5 cm), IMT tidak normal, atau berat badan kurang dari standar (< 45 kg).")
        else:
            st.success("✅ **Kategori: TIDAK BERISIKO**")
            st.write("Sistem memprediksi status kehamilan pasien aman. Parameter usia, berat badan, IMT, dan LiLA memiliki kedekatan jarak (Euclidean) dengan data warga yang tidak memiliki risiko klinis.")
