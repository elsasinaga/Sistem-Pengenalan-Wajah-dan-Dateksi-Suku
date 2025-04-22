# Sistem Pengenalan Wajah dan Deteksi Suku 
Elsa Monika Sinaga - 231511074  
Isyana Putri Indriana - 231511078  
Syahla Salsabila - 231511095

# Sistem Pengenalan Wajah dan Deteksi Suku/Etnis

Elsa Monika Sinaga - 231511074  
Isyana Putri Indriana - 231511078  
Syahla Salsabila - 231511095  

Sistem ini mengimplementasikan dua fungsi utama:

1. **Face Similarity**: Menggunakan model Siamese (ResNet50) untuk menghitung kemiripan antar wajah berdasarkan *Euclidean distance*.
2. **Deteksi Suku/Etnis**: Menggunakan model MobileNetV2 untuk memprediksi suku/etnis dari wajah yang terdeteksi menggunakan MTCNN.

Proyek ini mencakup Jupyter Notebook untuk analisis dan evaluasi, serta aplikasi Streamlit untuk antarmuka pengguna.

## Struktur Direktori

```
project/
├── training.csv
├── validation.csv
├── testing.csv
├── feature_extractor_siamese.h5
├── ethnicity_model.h5
├── label_encoder.pkl
├── training/
│   ├── Jawa/
│   │   ├── John/
│   │   │   ├── john_varian1.jpg
│   │   │   ├── john_varian2.jpg
│   │   │   ├── ...
│   ├── Sunda/
│   │   ├── ...
├── validation/
│   ├── Jawa/
│   │   ├── ...
│   ├── Sunda/
│   │   ├── ...
├── testing/
│   ├── Jawa/
│   │   ├── ...
│   ├── Sunda/
│   │   ├── ...
├── app_face_similarity.py
├── app_gabungan.py
├── app_split_data.py
├── evaluate_facesimilarity.py
├── model_training.py
├── train_face_similarity.py
├── face_similarity_ethnicity_detection.ipynb
├── requirements.txt
```

- **CSV Files**: `training.csv`, `validation.csv`, `testing.csv` berisi metadata gambar dengan kolom `path`, `nama`, `suku`, `ekspresi`, `sudut`, `jarak`, `pencahayaan`.
- **Model Files**:
  - `feature_extractor_siamese.h5`: Model Siamese untuk *face similarity*.
  - `ethnicity_model.h5`: Model MobileNetV2 untuk deteksi suku/etnis.
  - `label_encoder.pkl`: LabelEncoder untuk suku/etnis.
- **Gambar**: Disimpan di `training/[suku]/[nama]/[nama]_varianX.jpg`, `validation/[suku]/[nama]/[nama]_varianX.jpg`, `testing/[suku]/[nama]/[nama]_varianX.jpg`.
- **Aplikasi**:
  - `app_face_similarity.py`: Aplikasi Streamlit untuk *face similarity*.
  - `app_gabungan.py`: Aplikasi Streamlit untuk *face similarity* dan deteksi suku/etnis.
  - `app_split_data.py`: Aplikasi Streamlit untuk memproses dan membagi dataset.
- **Notebook**: `face_similarity_ethnicity_detection.ipynb` untuk evaluasi dan inferensi.
- **Skrip**:
  - `evaluate_facesimilarity.py`: Evaluasi metrik *face similarity*.
  - `model_training.py`: Pelatihan model deteksi suku/etnis.
  - `train_face_similarity.py`: Pelatihan model Siamese.

## Prasyarat

- Python 3.11
- Sistem operasi: Windows, Linux, atau MacOS
- Koneksi internet untuk menginstal dependensi
- GPU (opsional, untuk pelatihan lebih cepat)

## Setup

### 1. Kloning Repositori

Jika proyek berada di repositori Git, klon terlebih dahulu:

```bash
git clone https://github.com/elsasinaga/Sistem-Pengenalan-Wajah-dan-Deteksi-Suku.git
cd project
```

Jika tidak, pastikan semua file berada di direktori `project/`.

### 2. Buat Lingkungan Virtual

Buat dan aktifkan lingkungan virtual untuk mengisolasi dependensi:

```bash
python -m venv env
.\env\Scripts\activate  # Windows
# atau
source env/bin/activate  # Linux/Mac
```

### 3. Instal Dependensi

Pastikan file `requirements.txt` ada di direktori utama dengan isi berikut:

```
streamlit==1.18.0
opencv-python==4.5.5.64
mtcnn==0.1.1
tensorflow==2.12.0
keras==2.12.0
numpy==1.23.2
pandas==1.5.3
scikit-learn==1.6.1
Pillow==9.4.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
```

Instal dependensi:

```bash
pip install -r requirements.txt
```

Verifikasi instalasi:

```bash
pip list
```

### 4. Siapkan Dataset

Pastikan dataset sudah tersedia:

- **Folder Gambar**:

  - `training/[suku]/[nama]/[nama]_varianX.jpg`
  - `validation/[suku]/[nama]/[nama]_varianX.jpg`
  - `testing/[suku]/[nama]/[nama]_varianX.jpg`
  - Contoh: `testing/Jawa/John/john_varian1.jpg`

- **File CSV**:

  - `training.csv`, `validation.csv`, `testing.csv` harus ada di direktori utama.

  - Format CSV (dipisahkan dengan `;`):

    ```
    path,nama,suku,ekspresi,sudut,jarak,pencahayaan
    testing/Jawa/John/john_varian1.jpg,John,Jawa,Senyum,Frontal,Jauh,Terang
    testing/Jawa/John/john_varian2.jpg,John,Jawa,Serius,Frontal,Dekat,Redup
    testing/Sunda/Mary/mary_varian1.jpg,Mary,Sunda,Terkejut,Atas,Sedang,Redup
    ...
    ```

- Jika dataset belum ada, gunakan `app_split_data.py` untuk membuatnya (lihat bagian *Menjalankan Aplikasi Streamlit*).

Verifikasi dataset:

```python
import pandas as pd
df = pd.read_csv('testing.csv', sep=';')
print(df.head())
for path in df['path']:
    if not os.path.exists(path):
        print(f"Path tidak valid: {path}")
```

### 5. Siapkan Model

Pastikan model berikut ada di direktori utama:

- `feature_extractor_siamese.h5`
- `ethnicity_model.h5`
- `label_encoder.pkl`

Jika model belum ada, latih model menggunakan skrip berikut:

- Untuk *face similarity*:

  ```bash
  python train_face_similarity.py
  ```

- Untuk deteksi suku/etnis:

  ```bash
  python model_training.py
  ```

Verifikasi model:

```bash
dir feature_extractor_siamese.h5 ethnicity_model.h5 label_encoder.pkl  # Windows
# atau
ls feature_extractor_siamese.h5 ethnicity_model.h5 label_encoder.pkl  # Linux/Mac
```

## Menjalankan Sistem

### 1. Menjalankan Jupyter Notebook

Notebook `face_similarity_ethnicity_detection.ipynb` digunakan untuk evaluasi, visualisasi, dan inferensi.

1. Jalankan Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Buka browser di `http://localhost:8888`.

2. Buka `face_similarity_ethnicity_detection.ipynb`.

3. Jalankan sel-sel secara berurutan untuk:

   - Memuat model dan dataset.
   - Memvalidasi dataset.
   - Mengevaluasi *face similarity* (ROC, EER, TAR, FAR, FRR, precision, recall, F1).
   - Mengevaluasi deteksi suku/etnis (classification report, confusion matrix).
   - Membuat visualisasi (ROC curve, distribusi jarak, t-SNE, confusion matrix).
   - Menjalankan inferensi pada dua gambar.

4. Output akan disimpan di direktori utama, misalnya:

   - `roc_curve.png`
   - `confusion_matrix_facesimilarity.png`
   - `confusion_matrix_ethnicity.png`
   - `inference_example.png`

### 2. Menjalankan Aplikasi Streamlit

Aplikasi Streamlit menyediakan antarmuka pengguna untuk berinteraksi dengan sistem.

#### a. Aplikasi Face Similarity

Untuk membandingkan kemiripan antar wajah:

```bash
streamlit run app_face_similarity.py
```

- Buka browser di `http://localhost:8501`.
- Unggah dua gambar atau gunakan kamera.
- Lihat skor kemiripan dan status *match*/*tidak match*.

#### b. Aplikasi Gabungan

Untuk *face similarity* dan deteksi suku/etnis:

```bash
streamlit run app_gabungan.py
```

- Buka browser di `http://localhost:8501`.
- Unggah dua gambar atau gunakan kamera.
- Lihat skor kemiripan, status *match*/*tidak match*, dan prediksi suku/etnis beserta probabilitas.

#### c. Aplikasi Split Data

Untuk memproses dan membagi dataset:

```bash
streamlit run app_split_data.py
```

- Buka browser di `http://localhost:8501`.
- Unggah gambar, lakukan deteksi wajah, dan augmentasi.
- Simpan dataset ke `training/`, `validation/`, `testing/` dan buat CSV.

### 3. Evaluasi Tambahan

Untuk evaluasi metrik *face similarity* secara terpisah:

```bash
python evaluate_facesimilarity.py
```

Output termasuk metrik (ROC, EER, TAR, FAR, FRR) dan visualisasi.

## Penanganan Masalah

1. **ModuleNotFoundError**:

   - Pastikan semua dependensi terinstal:

     ```bash
     pip install -r requirements.txt
     ```

   - Verifikasi:

     ```bash
     pip list
     ```

2. **FileNotFoundError**:

   - Periksa keberadaan model:

     ```bash
     dir feature_extractor_siamese.h5 ethnicity_model.h5 label_encoder.pkl
     ```

   - Periksa path di CSV:

     ```python
     import pandas as pd
     df = pd.read_csv('testing.csv', sep=';')
     for path in df['path']:
         if not os.path.exists(path):
             print(f"Path tidak valid: {path}")
     ```

3. **Wajah Tidak Terdeteksi**:

   - Uji deteksi wajah:

     ```python
     from mtcnn import MTCNN
     import cv2
     detector = MTCNN()
     img = cv2.imread('testing/Jawa/John/john_varian1.jpg')
     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     faces = detector.detect_faces(img_rgb)
     print(f"Wajah terdeteksi: {len(faces)}")
     ```

   - Pastikan gambar memiliki wajah yang jelas.

4. **Peringatan TensorFlow**:

   - Jika muncul peringatan seperti `WARNING:tensorflow:Compiled the loaded model...`, abaikan atau simpan ulang model:

     ```python
     from tensorflow.keras.models import load_model
     model = load_model('feature_extractor_siamese.h5')
     model.save('feature_extractor_siamese_updated.h5')
     ```

5. **Dataset Kecil**:

   - Gunakan `app_split_data.py` untuk augmentasi data.

   - Pastikan setiap individu memiliki minimal 2 gambar:

     ```python
     import pandas as pd
     df = pd.read_csv('testing.csv', sep=';')
     print(df['nama'].value_counts())
     ```

## Contoh Output

- **Jupyter Notebook**:

  - Statistik dataset:

    ```
    testing.csv: 30 gambar, 8 individu, 3 suku
    ```

  - Metrik *face similarity*:

    ```
    Threshold dengan FAR < 0.05: 1.5000, AUC: 0.9500
    TAR: 0.9000, FAR: 0.0400, FRR: 0.1000
    Precision: 0.9200, Recall: 0.9000, F1: 0.9100
    ```

  - Evaluasi deteksi suku/etnis:

    ```
    precision    recall  f1-score   support
    Jawa       0.85      0.90      0.87        10
    Sunda      0.80      0.75      0.77         8
    Batak      0.88      0.86      0.87         7
    ```

  - Inferensi:

    ```
    Skor Kemiripan: 0.9123 (Jarak Euclidean: 0.8542)
    Status: Match
    Suku Gambar 1: Jawa (Probabilitas: 85.23%)
    Suku Gambar 2: Jawa (Probabilitas: 87.65%)
    ```

- **Aplikasi Streamlit**:

  - Antarmuka untuk mengunggah gambar, menampilkan skor kemiripan, dan prediksi suku/etnis.
  - Visualisasi wajah dengan bounding box.

## Kontribusi

Untuk menambahkan fitur atau memperbaiki bug:

1. Fork repositori.

2. Buat branch baru:

   ```bash
   git checkout -b fitur-baru
   ```

3. Commit perubahan:

   ```bash
   git commit -m "Menambahkan fitur baru"
   ```

4. Push dan buat pull request:

   ```bash
   git push origin fitur-baru
   ```

## Lisensi

\[Masukkan lisensi, misalnya MIT atau lainnya\]

## Kontak

Untuk pertanyaan atau dukungan, hubungi \[email/penyedia proyek\].
