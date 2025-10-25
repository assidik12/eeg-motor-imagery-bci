# src/config.py

# --- Konfigurasi Data ---
# Path ke direktori data mentah .gdf
DATA_DIR = './data/' 
# Subjek yang akan diproses (nantinya bisa di-loop)
SUBJECTS_TO_PROCESS = [1] 
# Frekuensi sampling data (di notebook Anda adalah 250 Hz)
SAMPLING_RATE = 250

# --- Parameter Preprocessing ---
# Filter frekuensi
L_FREQ = 8.0
H_FREQ = 30.0
# Parameter Epoching (windowing)
TMIN = -1.0  # Waktu mulai epoch (detik) relatif terhadap event
TMAX = 4.0   # Waktu akhir epoch (detik)
# Definisi event (berdasarkan notebook Anda untuk 4 kelas)
EVENT_ID = {'769': 0, '770': 1, '771': 2, '772': 3}
# Jumlah kelas
NB_CLASSES = len(EVENT_ID)

# --- Parameter Arsitektur Model ---
# Ini harus sesuai dengan data Anda setelah diproses
# CHANS = 22 (dari notebook)
CHANS = 22
# SAMPLES = 1000 (dari notebook, Anda mengambil 1000 sampel pertama dari epoch)
# Perhitungan: (TMAX - TMIN) * SAMPLING_RATE = 5.0 * 250 = 1251 sampel
# Notebook Anda memotongnya menjadi 1000, jadi kita ikuti:
SAMPLES = 1000

# Parameter EEGNet (sesuai notebook Anda)
MODEL_PARAMS = {
    'F1': 8,
    'D': 2,
    'F2': 16,
    'kernLength': 125,  # Sesuai notebook (SAMPLING_RATE / 2)
    'dropoutRate': 0.5,
    'dropoutType': 'Dropout'
}

# --- Parameter Training ---
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 300
RANDOM_SEED = 42

# --- Path Output ---
# Tempat menyimpan model yang sudah dilatih
MODEL_OUTPUT_DIR = './models/'
MODEL_FILENAME = 'eegnet_model.h5'
