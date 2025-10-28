import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist, validator
from typing import List
import uvicorn

# Impor config kita dari Sesi 1
# Kita perlu tahu di mana model disimpan dan bentuk datanya
try:
    import config
except ImportError:
    print("PERINGATAN: Tidak dapat mengimpor config.py. Menggunakan nilai default.")
    # Sediakan fallback jika config.py tidak ditemukan (berguna untuk testing cepat)
    class ConfigFallback:
        MODEL_OUTPUT_DIR = "../../models"
        MODEL_FILENAME = "eegnet_model.h5"
        CHANS = 22
        SAMPLES = 1000
        EVENT_ID = {'769': 0, '770': 1, '771': 2, '772': 3} # Default
    config = ConfigFallback()


# --- 1. Definisi "Data Contract" (Pydantic) ---

# Ini adalah data yang kita harapkan dari klien
class RawEpochData(BaseModel):
    # Kita harapkan data berbentuk List[List[float]]
    # yang bisa dikonversi ke numpy array (CHANS, SAMPLES) -> (22, 1000)
    data: List[List[float]]
    
    # Validasi tambahan Pydantic (opsional tapi bagus)
    # @validator('data')
    # def check_channels_count(cls, v):
    #     if len(v) != config.CHANS:
    #         raise ValueError(f"Data harus memiliki {config.CHANS} channels (List), diterima {len(v)}")
    #     return v

# Ini adalah data yang akan kita kirim kembali
class PredictionResponse(BaseModel):
    predicted_label: str
    predicted_index: int
    confidence: float
    raw_probabilities: List[float]

# --- 2. Inisialisasi Aplikasi FastAPI ---

app = FastAPI(
    title="EEGNet Motor Imagery API",
    description="API untuk memprediksi motor imagery dari data epoch EEG mentah.",
    version="1.0.0"
)

# Tempat untuk menyimpan model yang sudah dimuat
model = None
# Mapping dari index ke nama kelas (cth: 0 -> '769')
CLASS_LABELS = {}

# --- 3. Logika Startup (Memuat Model) ---

@app.on_event("startup")
def load_model_on_startup():
    """
    Memuat model Keras .h5 saat server pertama kali dinyalakan.
    Ini memastikan model ada di memori dan siap untuk prediksi cepat.
    """
    global model, CLASS_LABELS
    
    model_path = os.path.join(config.MODEL_OUTPUT_DIR, config.MODEL_FILENAME)
    
    if not os.path.exists(model_path):
        print(f"--- FATAL ERROR: File model tidak ditemukan di {model_path} ---")
        # Di aplikasi produksi, ini harus menghentikan server
    else:
        print(f"Memuat model dari: {model_path}...")
        try:
            model = tf.keras.models.load_model(model_path)
            model.summary() # Tampilkan summary di log server
            print("--- Model berhasil dimuat. ---")
            
            # Buat mapping terbalik untuk label
            # dari {'769': 0, ...} menjadi {0: '769', ...}
            CLASS_LABELS = {v: k for k, v in config.EVENT_ID.items()}
            print(f"Class labels dimuat: {CLASS_LABELS}")
            
        except Exception as e:
            print(f"--- FATAL ERROR: Gagal memuat model: {e} ---")

# --- 4. Endpoint Health Check (Best Practice) ---

@app.get("/")
def health_check():
    """
    Endpoint sederhana untuk mengecek apakah server berjalan.
    """
    return {"status": "API is running!"}

# --- 5. Endpoint Prediksi Utama ---

@app.post("/predict", response_model=PredictionResponse)
def predict_eeg(request: RawEpochData):
    """
    Menerima satu epoch data EEG (22, 1000) dan mengembalikan prediksi.
    """
    global model, CLASS_LABELS
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load on startup.")
        
    try:
        # 1. Konversi data Pydantic ke Numpy Array
        # Bentuk input: (22, 1000)
        X = np.array(request.data)
        
        # 2. Preprocessing (HARUS SAMA PERSIS dengan saat training)
        # a. Konversi V ke uV
        X = X * 1e6
        
        # 3. Reshape data untuk model
        # Model Keras mengharapkan 'batch'
        # Bentuk harus: (batch_size, n_channels, n_samples, 1)
        # -> (1, 22, 1000, 1)
        X_batch = X.reshape(1, config.CHANS, config.SAMPLES, 1)

        # 4. Jalankan prediksi
        # 'probs' akan berbentuk [[0.1, 0.7, 0.1, 0.1]]
        probs = model.predict(X_batch)[0] # Ambil hasil pertama dari batch
        
        # 5. Post-processing (Interpretasi hasil)
        predicted_index = int(np.argmax(probs))
        confidence = float(probs[predicted_index])
        predicted_label = CLASS_LABELS.get(predicted_index, "Unknown") # 'Unknown' jika index tidak ada

        return PredictionResponse(
            predicted_label=predicted_label,
            predicted_index=predicted_index,
            confidence=confidence,
            raw_probabilities=probs.tolist() # Konversi numpy array ke list JSON
        )
        
    except ValueError as ve:
        # Ini terjadi jika Pydantic check gagal atau numpy reshape gagal
        raise HTTPException(status_code=400, detail=f"Invalid data format: {ve}")
    except Exception as e:
        # Tangkap error lainnya
        print(f"ERROR: Terjadi kesalahan saat prediksi: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Bagian ini memungkinkan kita menjalankan file ini dengan `python src/api.py`
if __name__ == "__main__":
    print("Menjalankan server API (untuk debugging)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)