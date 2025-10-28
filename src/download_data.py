import os
import sys
import urllib.request
from pathlib import Path

try:
    import config
except ImportError:
    print("Gagal mengimpor config, mencoba menambah path...")
    sys.path.insert(0, os.path.abspath('.'))
    try:
        import config
    except ImportError:
        print("Fatal: Tidak dapat menemukan config.py.")
        print("Pastikan Anda menjalankan skrip ini dari dalam folder 'src/'")
        sys.exit(1)


def download_file(url, destination):
    """
    Mengunduh file dari URL ke destination menggunakan urllib
    """
    try:
        print(f"    Mengunduh dari: {url}")
        urllib.request.urlretrieve(url, destination)
        print(f"    Ukuran file: {os.path.getsize(destination) / (1024*1024):.2f} MB")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def download_eeg_data():
    """
    Mengunduh file data BCI Competition IV 2a (format GDF)
    Dataset: https://www.bbci.de/competition/iv/desc_2a.pdf
    """
    
    # URL dasar untuk BCI Competition IV dataset 2a
    BASE_URL = "http://bnci-horizon-2020.eu/database/data-sets/001-2014/"
    
    src_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(src_dir, config.DATA_DIR))
    
    print(f"Memastikan direktori data ada di: {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    subjects = config.SUBJECTS_TO_PROCESS
    print(f"Akan mengunduh data BCI Competition IV 2a untuk subjek: {subjects}")
    print("Dataset: Motor Imagery (4 class)")
    print("Format: GDF (General Data Format)\n")

    for subject in subjects:
        # File training dan evaluation
        files_to_download = [
            (f'A0{subject}T.gdf', 'training'),  # Training data
            (f'A0{subject}E.gdf', 'evaluation') # Evaluation data
        ]
        
        for file_name, file_type in files_to_download:
            file_path = os.path.join(data_dir, file_name)
            
            if os.path.exists(file_path):
                print(f"  ✓ File {file_name} ({file_type}) sudah ada, unduhan dilewati.")
                continue
            
            print(f"  Mengunduh {file_name} ({file_type})...")
            
            # URL lengkap ke file GDF
            url = f"{BASE_URL}{file_name}"
            
            success = download_file(url, file_path)
            
            if success:
                # Verifikasi file adalah GDF yang valid
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(8)
                        if header[:3] != b'GDF':
                            print(f"  ⚠️ Peringatan: File mungkin bukan format GDF yang valid")
                            os.remove(file_path)
                            print(f"  ✗ File {file_name} dihapus karena tidak valid")
                            continue
                    print(f"  ✓ Selesai mengunduh {file_name}")
                except Exception as e:
                    print(f"  ⚠️ Error verifikasi file: {e}")
            else:
                print(f"  ✗ Gagal mengunduh {file_name}")
                print(f"\n  CATATAN: Jika unduhan gagal, Anda bisa:")
                print(f"  1. Unduh manual dari: {BASE_URL}")
                print(f"  2. Letakkan file di folder: {data_dir}")

    print("\n=== Informasi Dataset ===")
    print("Dataset: BCI Competition IV - Dataset 2a")
    print("Subjek: 9 orang")
    print("Kelas: 4 (left hand, right hand, both feet, tongue)")
    print("Channel: 22 EEG + 3 EOG")
    print("Sampling rate: 250 Hz")
    print("\nJika unduhan gagal, kunjungi:")
    print("http://bnci-horizon-2020.eu/database/data-sets")


if __name__ == "__main__":
    download_eeg_data()
