import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# --- Impor Modul Kustom Kita ---
import config
from model import EEGNet
from data_processing import load_and_preprocess_data

def set_seeds(seed=config.RANDOM_SEED):
    """
    Mengatur random seeds untuk reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(f"Random seeds diatur ke: {seed}")

def train_model():
    """
    Fungsi utama untuk melatih model.
    """
    print("Memulai proses training...")
    
    # 1. Atur Seeds
    set_seeds()
    
    # 2. Muat dan Proses Data
    # Saat ini kita hanya melatih pada subjek pertama (sesuai config)
    print(f"Memuat data untuk subjek: {config.SUBJECTS_TO_PROCESS[0]}...")
    X, y = load_and_preprocess_data(config.DATA_DIR, config.SUBJECTS_TO_PROCESS[0])
    
    if X is None or y is None:
        print("Gagal memuat data. Proses training dibatalkan.")
        return
        
    # 3. Split Data (Train/Validation)
    # Kita split data dari subjek ini menjadi train dan validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=config.RANDOM_SEED,
        stratify=y  # Penting untuk data yang tidak seimbang
    )
    
    print(f"Bentuk X_train: {X_train.shape} | Bentuk y_train: {y_train.shape}")
    print(f"Bentuk X_val: {X_val.shape}   | Bentuk y_val: {y_val.shape}")

    # 4. Buat Model
    print("Membuat arsitektur model EEGNet...")
    model = EEGNet(
        nb_classes=config.NB_CLASSES,
        Chans=config.CHANS,
        Samples=config.SAMPLES,
        **config.MODEL_PARAMS # Memasukkan F1, D, F2, dll. dari config
    )
    
    model.summary()
    
    # 5. Kompilasi Model
    print("Mengk kompilasi model...")
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    # 6. Tentukan Path untuk Menyimpan Model
    if not os.path.exists(config.MODEL_OUTPUT_DIR):
        os.makedirs(config.MODEL_OUTPUT_DIR)
        print(f"Membuat direktori: {config.MODEL_OUTPUT_DIR}")
        
    model_save_path = os.path.join(config.MODEL_OUTPUT_DIR, config.MODEL_FILENAME)
    print(f"Model akan disimpan di: {model_save_path}")

    # 7. Siapkan Callbacks
    # ModelCheckpoint menyimpan model terbaik (berdasarkan val_accuracy)
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # EarlyStopping menghentikan training jika tidak ada peningkatan
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=50, # Jumlah epoch tanpa peningkatan sebelum berhenti
        verbose=1,
        mode='min',
        restore_best_weights=True # Kembalikan weights terbaik saat berhenti
    )

    callbacks_list = [checkpoint, early_stop]
    
    # 8. Latih Model
    print("=== MEMULAI TRAINING ===")
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    print("=== TRAINING SELESAI ===")
    
    # 9. (Opsional) Evaluasi model terbaik pada data validasi
    # Karena restore_best_weights=True, model sudah memiliki weights terbaik
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nHasil akhir pada data validasi (dari weights terbaik):")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    
    # Catatan: File .h5 di `model_save_path` berisi model terbaik
    # berkat `save_best_only=True` pada ModelCheckpoint.
    print(f"Model terbaik disimpan di {model_save_path}")

if __name__ == '__main__':
    train_model()
