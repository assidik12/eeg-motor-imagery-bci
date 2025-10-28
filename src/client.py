import requests  # Library untuk membuat permintaan HTTP
import json
import os
import sys
import mne
import numpy as np

# --- (BAGIAN 0: Menyiapkan Path) ---
# Karena skrip ini ada di 'src/', kita bisa impor 'config' secara langsung.
try:
    import config
except ImportError:
    print(f"ERROR: Gagal mengimpor 'config.py'.")
    print("Pastikan skrip ini ada di dalam folder 'src/' dan dijalankan dari root proyek.")
    print("Contoh: python src/client.py")
    sys.exit(1)

# --- (BAGIAN 1: Mempersiapkan Data Uji Coba) ---

print("Mempersiapkan data uji coba (mengambil 1 epoch)...")

# Path ke data
# config.DATA_DIR adalah '../data' (relatif dari 'src/')
SRC_DIR = os.path.dirname(os.path.abspath(__file__)) # Ini adalah 'src/'
DATA_DIR_ABSOLUTE = os.path.abspath(os.path.join(SRC_DIR, config.DATA_DIR))

subject = config.SUBJECTS_TO_PROCESS[0] # Ambil subjek pertama dari config
file_path = os.path.join(DATA_DIR_ABSOLUTE, f'A0{subject}T.gdf')

if not os.path.exists(file_path):
    print(f"ERROR: File data tidak ditemukan di {file_path}")
    print(">>> Pastikan Anda sudah menjalankan 'python src/download_data.py' terlebih dahulu! <<<")
    sys.exit(1)
else:
    print(f"Memuat data dari: {file_path}")
    
    # 1. Muat data mentah
    raw = mne.io.read_raw_gdf(file_path, preload=True, eog=['EOG-left', 'EOG-central', 'EOG-right'])
    raw.set_eeg_reference('average', projection=False)
    
    # 2. Filter (SAMA DENGAN TRAINING)
    raw.filter(config.L_FREQ, config.H_FREQ, fir_design='firwin', skip_by_annotation='edge')
    
    # 3. Epoch (SAMA DENGAN TRAINING)
    events, _ = mne.events_from_annotations(raw, event_id=config.EVENT_ID)
    
    # --- ⬇️ ⬇️ ⬇️ INI PERBAIKANNYA ⬇️ ⬇️ ⬇️ ---
    # Kita gunakan string 'eeg' (lowercase) untuk memilih TIPE channel
    epochs = mne.Epochs(raw, events, event_id=config.EVENT_ID, 
                        tmin=config.TMIN, tmax=config.TMAX, 
                        baseline=None, preload=True,
                        picks='eeg') # <-- PERUBAHAN DARI ['EEG']
    # --- ⬆️ ⬆️ ⬆️ INI PERBAIKANNYA ⬆️ ⬆️ ⬆️ ---
                        
    # 4. Ambil data dalam VOLT (V)
    # Bentuknya (n_epochs, 22, 1251) <-- 1251 adalah (TMAX * sfreq) + 1
    X_all_epochs_volt = epochs.get_data() 
    
    # 5. Ambil SATU sampel (epoch pertama)
    sample_epoch_v = X_all_epochs_volt[0] # Bentuk (22, 1251)
    
    # 6. TRUNCATE data agar sama dengan data training (1000 sampel)
    # Model dilatih dengan data yang dipotong sesuai config.SAMPLES
    # Ini meniru baris `X = epochs_data[:, :, :config.SAMPLES]` di data_processing.py
    sample_epoch_v_truncated = sample_epoch_v[:, :config.SAMPLES] # Bentuk (22, 1000)
    
    print(f"Bentuk epoch asli: {sample_epoch_v.shape}")
    print(f"Bentuk epoch terpotong (truncated): {sample_epoch_v_truncated.shape}")
    
    # 7. Konversi data yang SUDAH DIPOTONG ke format list JSON
    sample_epoch_list = sample_epoch_v_truncated.tolist()
    
    # # 8. Buat payload JSON
    payload = {
        "data": sample_epoch_list
    }
    
    import sys

    print(f"Ukuran payload JSON: {sys.getsizeof(json.dumps(payload)) / 1024:.2f} KB")
    print(f"Ukuran array numpy: {sample_epoch_v_truncated.nbytes / 1024:.2f} KB")

    print(f"Data uji coba (epoch 0 dari subjek {subject}) siap dikirim.")
    
    # --- (BAGIAN 2: Memanggil API Anda) ---
    
    API_URL = "http://127.0.0.1:8000" 
    predict_endpoint = f"{API_URL}/predict"
    
    print(f"Mengirim permintaan POST ke: {predict_endpoint}...")
    
    try:
        # Kirim data sebagai JSON
        response = requests.post(predict_endpoint, json=payload, timeout=30) # 30 detik timeout
        
        # Cek status
        response.raise_for_status() # Akan error jika status 4xx or 5xx
        
        # Tampilkan hasil
        print("\n--- ✅ SUKSES! Respons dari API: ---")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.HTTPError as errh:
        print(f"\n--- ❌ HTTP Error: {errh} ---")
        print(f"Detail: {response.text}")
    except requests.exceptions.ConnectionError as errc:
        print(f"\n--- ❌ Error Koneksi: {errc} ---")
        print(">>> PASTIKAN SERVER API ANDA SUDAH BERJALAN! <<<")
        print(">>> (Jalankan 'uvicorn src.api:app --reload' di terminal lain) <<<")
    except requests.exceptions.Timeout as errt:
        print(f"\n--- ❌ Timeout Error: {errt} ---")
    except requests.exceptions.RequestException as err:
        print(f"\n--- ❌ Error Lainnya: {err} ---")

