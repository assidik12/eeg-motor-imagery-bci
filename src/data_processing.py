import os
import glob
import numpy as np
import mne
from tensorflow.keras.utils import to_categorical

# Impor konfigurasi dari file config.py
import config

def load_and_preprocess_data(data_dir, subject_id):
    """
    Memuat data GDF untuk satu subjek, menerapkan filter, 
    membuat epoch, dan memformatnya untuk training.
    
    Menggunakan parameter dari file config.py
    """
    
    # 1. Cari file data untuk subjek
    # Asumsi nama file adalah 'A0<subject_id>T.gdf' atau 'A0<subject_id>E.gdf'
    search_path = os.path.join(data_dir, f'A0{subject_id}*.gdf')
    gdf_files = glob.glob(search_path)
    
    if not gdf_files:
        print(f"Error: Tidak ada file .gdf ditemukan untuk subjek {subject_id} di {data_dir}")
        return None, None
    
    print(f"Memuat file: {gdf_files}")
    
    # 2. Load data mentah menggunakan MNE
    # Kita gabungkan semua file gdf untuk subjek ini (jika ada T dan E)
    raw_files = [mne.io.read_raw_gdf(f, preload=True) for f in gdf_files]
    raw = mne.concatenate_raws(raw_files)
    
    # 3. Membersihkan channel names (sesuai notebook)
    raw.rename_channels(lambda s: s.strip('.'))
    
    # 4. Set EOG channels (sesuai notebook)
    # Ini penting agar MNE tahu channel mana yang bukan EEG
    raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})
    
    # 5. Terapkan Band-pass filter
    raw.filter(l_freq=config.L_FREQ, h_freq=config.H_FREQ, 
               fir_design='firwin', skip_by_annotation='edge')
    
    # 6. Ekstraksi Events
    events, _ = mne.events_from_annotations(raw, event_id=config.EVENT_ID)
    
    # 7. Buat Epochs
    # Pilih hanya channel EEG
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, 
                           exclude='bads')
                           
    epochs = mne.Epochs(raw, events, event_id=config.EVENT_ID, 
                        tmin=config.TMIN, tmax=config.TMAX, 
                        proj=True, picks=picks, 
                        baseline=None, preload=True)
                        
    # 8. Ambil data dan label
    # Format MNE: (n_epochs, n_channels, n_samples)
    X = epochs.get_data() 
    y = epochs.events[:, -1] # Ambil label dari events
    
    # 9. Penyesuaian Label (sesuai notebook)
    # Label asli adalah 769, 770, ...
    # Kita map ke 0, 1, 2, ...
    # Kita bisa asumsikan `epochs.events[:, -1]` sudah berisi nilai dari EVENT_ID
    # Jika tidak, kita perlu mapping. Mari kita periksa:
    # Event_id MNE `{'769': 0, '770': 1, '771': 2, '772': 3}` 
    # MNE akan me-map '769' (di anotasi) ke event ID 0 (di `epochs.events`)
    # Jadi 'y' sudah berisi 0, 1, 2, 3.
    
    # 10. Memformat data untuk Keras/TensorFlow
    
    # a. Konversi data (dari V ke uV, sesuai notebook)
    X *= 1e6
    
    # b. Potong sampel (sesuai notebook, dari 1251 menjadi 1000)
    X = X[:, :, :config.SAMPLES]
    
    # c. Reshape data: (n_epochs, n_channels, n_samples) 
    #    -> (n_epochs, n_channels, n_samples, 1)
    # Ini adalah format 'channels_first' yang diharapkan EEGNet
    X = X.reshape(X.shape[0], config.CHANS, config.SAMPLES, 1)
    
    # d. Konversi label ke one-hot encoding
    y_one_hot = to_categorical(y, num_classes=config.NB_CLASSES)
    
    print(f"Data preprocessing selesai untuk subjek {subject_id}.")
    print(f"Bentuk X: {X.shape}")
    print(f"Bentuk y: {y_one_hot.shape}")
    
    return X, y_one_hot

if __name__ == '__main__':
    # Bagian ini untuk testing cepat
    # Jalankan file ini (python src/data_processing.py) 
    # Pastikan Anda punya folder 'data' dengan file 'A01T.gdf' di dalamnya
    
    print("Menjalankan data_processing.py sebagai skrip mandiri...")
    
    # Asumsi Anda punya folder 'data' di root
    # Sesuaikan path ini jika perlu
    TEST_DATA_DIR = './data/'
    TEST_SUBJECT = 1
    
    X_test, y_test = load_and_preprocess_data(TEST_DATA_DIR, TEST_SUBJECT)
    
    if X_test is not None:
        print("\nTesting berhasil.")
        print(f"Shape X: {X_test.shape}")
        print(f"Shape y: {y_test.shape}")
        print(f"Contoh label (one-hot): {y_test[0]}")
        print(f"Nilai min/max X: {np.min(X_test):.2f} / {np.max(X_test):.2f}")
    else:
        print("Testing gagal.")