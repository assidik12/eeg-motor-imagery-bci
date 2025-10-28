# **Klasifikasi Sinyal EEG Motor Imagery Lintas-Subjek dengan EEGNet**

## **📖 Ringkasan Proyek**

Proyek ini adalah implementasi _end-to-end_ dari sebuah Brain-Computer Interface (BCI) yang mampu mengklasifikasikan niat gerakan tangan (kiri vs. kanan) dari data mentah EEG. Model ini menggunakan arsitektur _Deep Learning_ **EEGNet** dan dilatih pada dataset besar yang terdiri dari **20 subjek** untuk memastikan kemampuannya dalam melakukan generalisasi pada data dari pengguna baru.

Tujuan utama proyek ini adalah untuk membangun sebuah model BCI lintas-subjek (_cross-subject_) yang andal, yang merupakan fondasi penting untuk aplikasi BCI yang tidak memerlukan kalibrasi panjang untuk setiap pengguna baru.

## **✨ Fitur Utama**

- **Pipeline Lengkap:** Dari _preprocessing_ data EEG mentah menggunakan **MNE-Python** hingga pelatihan model _Deep Learning_ dengan **TensorFlow/Keras**.
- **Arsitektur Canggih:** Mengimplementasikan **EEGNet**, sebuah arsitektur CNN yang dirancang khusus untuk data EEG, dari nol.
- **Validasi Lintas-Subjek:** Model dilatih pada data dari 20 subjek dan diuji pada data dari subjek yang sepenuhnya baru untuk membuktikan kemampuan generalisasinya.
- **REST API Server:** API FastAPI untuk melakukan prediksi real-time melalui HTTP endpoint.
- **Docker Support:** Containerized deployment menggunakan Docker untuk portabilitas dan kemudahan deployment.
- **Simulasi Prediksi:** Termasuk skrip client untuk mendemonstrasikan bagaimana model yang telah dilatih dapat digunakan untuk membuat prediksi pada data EEG baru.

## **📂 Struktur Repositori**

```
eeg-motor-imagery-bci/
│
├── README.md                     # Dokumentasi ini
├── requirements.txt              # Daftar library yang dibutuhkan
├── Dockerfile                    # Docker configuration untuk containerization
├── .dockerignore                 # File yang diabaikan saat build Docker image
│
├── src/                          # Source code utama
│   ├── config.py                 # Konfigurasi global (parameter EEG, model, dll)
│   ├── download_data.py          # Script untuk mengunduh dataset BCI Competition IV 2a
│   ├── data_processing.py        # Preprocessing data EEG dengan MNE
│   ├── model.py                  # Implementasi arsitektur EEGNet
│   ├── train.py                  # Script pelatihan model
│   ├── api.py                    # FastAPI REST API server untuk prediksi
│   └── client.py                 # Client script untuk testing API
│
├── data/                         # Dataset EEG (diunduh otomatis)
│   └── A01T.gdf, A01E.gdf, ...   # File GDF dari BCI Competition
│
├── models/                       # Model yang sudah terlatih
│   └── eegnet_model_final.h5     # Model Keras final
│
└── results/                      # Hasil pelatihan
    ├── training_history.pkl      # Riwayat akurasi & loss
    └── training_plot.png         # Visualisasi hasil pelatihan
```

## **🛠️ Setup & Instalasi**

### **Opsi 1: Instalasi Lokal**

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

1. **Clone repositori ini:**

   ```bash
   git clone https://github.com/assidik12/eeg-motor-imagery-bci.git
   cd eeg-motor-imagery-bci
   ```

2. **Buat virtual environment:**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### **Opsi 2: Docker (Recommended untuk Production)**

Untuk deployment yang lebih mudah dan konsisten, gunakan Docker:

1. **Build Docker image:**

   ```bash
   docker build -t eeg-bci-api .
   ```

2. **Jalankan container:**

   ```bash
   docker run -d -p 8000:8000 --name eeg-bci eeg-bci-api
   ```

3. **Akses API:**

   - API akan berjalan di: `http://localhost:8000`
   - Dokumentasi interaktif: `http://localhost:8000/docs`

4. **Stop container:**
   ```bash
   docker stop eeg-bci
   docker rm eeg-bci
   ```

## **🚀 Cara Menjalankan**

### **A. Training Model**

1. **Download dataset:**

   ```bash
   python src/download_data.py
   ```

2. **Preprocessing data:**

   ```bash
   python src/data_processing.py
   ```

3. **Train model:**
   ```bash
   python src/train.py
   ```
   _Catatan: Proses pelatihan mungkin memakan waktu cukup lama (beberapa jam tergantung hardware)._

### **B. Menjalankan API Server**

#### **Opsi 1: Local**

```bash
# Jalankan dengan uvicorn
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Atau jalankan langsung
python src/api.py
```

#### **Opsi 2: Docker**

```bash
# Pastikan Docker sudah terinstall
docker build -t eeg-bci-api .
docker run -d -p 8000:8000 --name eeg-bci eeg-bci-api
```

#### **Akses API:**

- **Base URL:** `http://localhost:8000`
- **Docs (Swagger UI):** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`

### **C. Testing Prediksi**

Setelah API server berjalan, test dengan client script:

```bash
python src/client.py
```

Atau test manual dengan curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_data.json
```

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
