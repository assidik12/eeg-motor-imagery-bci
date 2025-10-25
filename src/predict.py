import os
import numpy as np
import tensorflow as tf
import config
from data_processing import load_and_preprocess_data

def predict_single_sample():
    """
    Memuat model yang sudah dilatih dan menjalankan prediksi 
    pada satu sampel data.
    
    Untuk simulasi ini, kita akan:
    1. Memuat model .h5 dari disk.
    2. Memuat dataset (menggunakan fungsi preprocessing kita).
    3. Mengambil SATU sampel dari dataset itu.
    4. Menjalankan prediksi pada sampel tersebut.
    """
    
    print("Memulai skrip inferensi...")
    
    # 0. Buat mapping terbalik dari ID ke Label untuk output
    # config.EVENT_ID = {'769': 0, '770': 1, '771': 2, '772': 3}
    # Kita ingin: {0: '769', 1: '770', 2: '771', 3: '772'}
    CLASS_LABELS = {v: k for k, v in config.EVENT_ID.items()}
    
    # 1. Tentukan path dan muat model
    model_path = os.path.join(config.MODEL_OUTPUT_DIR, config.MODEL_FILENAME)
    
    if not os.path.exists(model_path):
        print(f"Error: File model tidak ditemukan di {model_path}")
        print("Harap jalankan train.py terlebih dahulu.")
        return
        
    print(f"Memuat model dari: {model_path}")
    # Memuat model Keras (.h5)
    # Ini memuat arsitektur, weights, dan konfigurasi optimizer
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model berhasil dimuat.")
        model.summary()
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        return

    # 2. Muat data untuk mendapatkan sampel
    # Kita ambil data dari subjek yang sama kita latih
    print(f"Memuat data sampel dari subjek {config.SUBJECTS_TO_PROCESS[0]}...")
    X, y = load_and_preprocess_data(config.DATA_DIR, config.SUBJECTS_TO_PROCESS[0])
    
    if X is None:
        print("Gagal memuat data sampel. Skrip berhenti.")
        return
        
    # 3. Ambil satu sampel data acak
    sample_index = np.random.randint(0, X.shape[0])
    sample_X = X[sample_index] 
    sample_y_true = y[sample_index] # Ini masih one-hot (cth: [0, 1, 0, 0])
    
    print(f"Mengambil sampel acak index ke-{sample_index}")
    print(f"Bentuk data sampel (sample_X): {sample_X.shape}") # (22, 1000, 1)
    
    # 4. Format sampel untuk prediksi
    # Model.predict() mengharapkan 'batch' data, bukan sampel tunggal.
    # Kita perlu mengubah (22, 1000, 1) -> (1, 22, 1000, 1)
    sample_X_batch = np.expand_dims(sample_X, axis=0)
    print(f"Bentuk data batch (sample_X_batch): {sample_X_batch.shape}")

    # 5. Jalankan prediksi
    print("Menjalankan prediksi (model.predict)...")
    prediction_probs = model.predict(sample_X_batch)
    
    # 6. Interpretasi hasil
    # 'prediction_probs' akan berbentuk [[0.1, 0.2, 0.6, 0.1]] (batch size 1)
    predicted_index = np.argmax(prediction_probs[0])
    predicted_label = CLASS_LABELS[predicted_index]
    
    true_index = np.argmax(sample_y_true)
    true_label = CLASS_LABELS[true_index]
    
    print("\n--- HASIL PREDIKSI ---")
    print(f"Prediksi (probabilitas mentah): {prediction_probs[0]}")
    print(f"Label Sebenarnya: {true_label} (Index: {true_index})")
    print(f"Hasil Prediksi:   {predicted_label} (Index: {predicted_index})")
    
    if predicted_index == true_index:
        print("\nPrediksi BENAR!")
    else:
        print("\nPrediksi SALAH.")

if __name__ == '__main__':
    predict_single_sample()
