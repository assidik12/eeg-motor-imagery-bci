# **Klasifikasi Sinyal EEG Motor Imagery Lintas-Subjek dengan EEGNet**

## **📖 Ringkasan Proyek**

Proyek ini adalah implementasi _end-to-end_ dari sebuah Brain-Computer Interface (BCI) yang mampu mengklasifikasikan niat gerakan tangan (kiri vs. kanan) dari data mentah EEG. Model ini menggunakan arsitektur _Deep Learning_ **EEGNet** dan dilatih pada dataset besar yang terdiri dari **20 subjek** untuk memastikan kemampuannya dalam melakukan generalisasi pada data dari pengguna baru.

Tujuan utama proyek ini adalah untuk membangun sebuah model BCI lintas-subjek (_cross-subject_) yang andal, yang merupakan fondasi penting untuk aplikasi BCI yang tidak memerlukan kalibrasi panjang untuk setiap pengguna baru.

## **✨ Fitur Utama**

- **Pipeline Lengkap:** Dari _preprocessing_ data EEG mentah menggunakan **MNE-Python** hingga pelatihan model _Deep Learning_ dengan **TensorFlow/Keras**.
- **Arsitektur Canggih:** Mengimplementasikan **EEGNet**, sebuah arsitektur CNN yang dirancang khusus untuk data EEG, dari nol.
- **Validasi Lintas-Subjek:** Model dilatih pada data dari 20 subjek dan diuji pada data dari subjek yang sepenuhnya baru untuk membuktikan kemampuan generalisasinya.
- **Simulasi Prediksi:** Termasuk skrip (simulate_prediction.py) untuk mendemonstrasikan bagaimana model yang telah dilatih dapat digunakan untuk membuat prediksi pada satu sampel data EEG baru.

## **📂 Struktur Repositori**

```
eeg-motor-imagery-bci/
│
├── README.md                 \# Dokumentasi ini
├── BCI\_Motor\_Imagery\_EEGNet.ipynb \# Notebook utama berisi seluruh alur kerja
├── eegnet\_model\_final.h5     \# Model Keras yang sudah terlatih
├── training\_history.pkl      \# Riwayat akurasi & loss selama pelatihan
├── simulate\_prediction.py    \# Skrip untuk menjalankan simulasi prediksi
├── results/
│   └── training\_plot.png       \# Gambar hasil plot pelatihan
└── requirements.txt          \# Daftar library yang dibutuhkan
```

## **🛠️ Setup & Instalasi**

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

1. **Clone repositori ini:**  
   git clone [https://github.com/assidik12/eeg-motor-imagery-bci](https://github.com/assidik12/eeg-motor-imagery-bci.git)  
   cd NAMA_REPO_ANDA

2. **Buat sebuah _virtual environment_ (sangat direkomendasikan):**  
   python \-m venv venv  
   source venv/bin/activate \# Di Windows, gunakan \`venv\\Scripts\\activate\`

3. **Instal semua _library_ yang dibutuhkan:**  
   pip install \-r requirements.txt

## **🚀 Cara Menjalankan**

1. **Menjalankan Alur Kerja Lengkap (Pelatihan):**
   - Buka dan jalankan sel-sel di dalam notebook BCI_Motor_Imagery_EEGNet.ipynb secara berurutan.
   - Notebook ini akan menangani pengunduhan data, _preprocessing_, pelatihan model, dan penyimpanan hasil secara otomatis. _Catatan: Proses pelatihan mungkin memakan waktu cukup lama._
2. **Menjalankan Simulasi Prediksi (Setelah Model Dilatih):**

   - Setelah file eegnet_model_final.h5 berhasil dibuat, jalankan skrip simulasi dari terminal:

   python simulate_prediction.py

   - Skrip ini akan memuat model, mengambil data dari subjek baru (yang tidak ada dalam set pelatihan), dan membuat prediksi tunggal.

## **📊 Hasil**

Model ini berhasil dilatih pada **20 subjek** dan mencapai performa yang stabil pada data validasi, membuktikan kemampuannya untuk mempelajari pola umum dari sinyal EEG _motor imagery_.

- **Akurasi Validasi:** Mencapai akurasi yang stabil di kisaran **55-58%**, yang merupakan _baseline_ yang solid untuk masalah BCI lintas-subjek yang sangat menantang ini.
- **Generalisasi:** Plot pelatihan menunjukkan bahwa model berhasil belajar tanpa mengalami _overfitting_ yang parah, di mana kurva validasi mengikuti tren umum dari kurva pelatihan.

## **💡 Rencana Peningkatan di Masa Depan**

Proyek ini adalah fondasi yang kuat. Beberapa area untuk peningkatan di masa depan meliputi:

- **Validasi yang Lebih Kuat:** Mengimplementasikan skema **Leave-One-Subject-Out Cross-Validation (LOSO-CV)** untuk mendapatkan metrik performa yang lebih andal.
- **Interpretasi Model (XAI):** Memvisualisasikan filter spasial dan temporal dari EEGNet untuk memahami dasar neurofisiologis dari keputusan model.
- **Eksplorasi Arsitektur:** Mencoba arsitektur yang lebih canggih seperti model hybrid **CNN-LSTM** atau **EEG-Transformer**.
- **Augmentasi Data:** Menerapkan teknik augmentasi sinyal untuk meningkatkan ketahanan dan performa model.
