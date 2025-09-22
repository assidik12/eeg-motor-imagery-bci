import numpy as np
from tensorflow.keras.models import load_model
import mne
from mne.datasets import eegbci
import warnings

# Mengabaikan pesan peringatan yang tidak relevan dari MNE
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# FUNGSI HELPER: Untuk mengambil data dari satu subjek baru
# =============================================================================
def get_data_from_subject(subject_id):
    """Fungsi ini memuat dan memproses data EEG untuk satu subjek."""
    runs = [6, 10, 14] # Run motor imagery
    raw_files = eegbci.load_data(subject_id, runs, update_path=True, verbose=False)
    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose='WARNING') for f in raw_files])

    raw.pick_types(meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    raw.filter(l_freq=4.0, h_freq=38.0, fir_design='firwin', skip_by_annotation='edge')

    event_id = dict(T1=1, T2=2)
    events, _ = mne.events_from_annotations(raw, event_id=event_id)

    tmin, tmax = -1., 4.
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        proj=True, baseline=None, preload=True, verbose=False)

    X = epochs.get_data()
    y = epochs.events[:, -1] - 1
    return X, y

# =============================================================================
# SKRIP UTAMA UNTUK SIMULASI PREDIKSI
# =============================================================================

# 1. SETUP
MODEL_PATH = 'eegnet_model_final.h5'
CLASS_MAP = {0: 'Tangan Kiri', 1: 'Tangan Kanan'}
TEST_SUBJECT_ID = 25 # Gunakan subjek yang TIDAK termasuk dalam data pelatihan (1-20)

# 2. MUAT MODEL "OTAK" BCI KITA
print(f"🧠 Memuat model terlatih dari '{MODEL_PATH}'...")
model = load_model(MODEL_PATH)
print("✅ Model berhasil dimuat.")

# 3. AMBIL DATA UJI BARU
print(f"\n📡 Mengambil data EEG dari subjek baru (ID: {TEST_SUBJECT_ID})...")
X_new, y_new = get_data_from_subject(TEST_SUBJECT_ID)

# Pilih satu sampel acak dari subjek baru ini
random_index = np.random.randint(0, len(X_new))
sample_X = X_new[random_index]
true_label_index = y_new[random_index]
true_label_name = CLASS_MAP[true_label_index]
print(f"✅ Satu sampel '{true_label_name}' berhasil diambil.")

# 4. PERSIAPAN DATA UNTUK MODEL
# Model membutuhkan input 4D: (batch_size, channels, samples, 1)
# Kita ubah shape dari (64, 801) -> (1, 64, 801, 1)
sample_for_prediction = np.expand_dims(np.expand_dims(sample_X, axis=0), axis=-1)

# 5. LAKUKAN PREDIKSI
print("\n🤖 Melakukan prediksi dengan EEGNet...")
prediction_probabilities = model.predict(sample_for_prediction)

# 6. TERJEMAHKAN HASIL PREDIKSI
predicted_class_index = np.argmax(prediction_probabilities)
predicted_class_name = CLASS_MAP[predicted_class_index]
confidence = np.max(prediction_probabilities) * 100

# 7. TAMPILKAN HASILNYA!
print("\n" + "="*40)
print("     HASIL SIMULASI BRAIN-COMPUTER INTERFACE")
print("="*40)
print(f"Label Sebenarnya\t: {true_label_name}")
print(f"Prediksi Model\t\t: {predicted_class_name}")
print(f"Tingkat Kepercayaan\t: {confidence:.2f}%")
print("="*40)

if predicted_class_name == true_label_name:
    print("\n🎉 KESIMPULAN: Prediksi BENAR!")
else:
    print("\n📉 KESIMPULAN: Prediksi SALAH.")