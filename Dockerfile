FROM python:3.11-slim

# --- Tahap 2: Menyiapkan Lingkungan di Dalam Container ---

WORKDIR /app

# --- Tahap 3: Menginstal Dependensi ---

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# --- Tahap 4: Menyalin File Proyek ---

COPY ./src ./src

COPY ./models ./models

# --- Tahap 5: Menjalankan Aplikasi ---

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]