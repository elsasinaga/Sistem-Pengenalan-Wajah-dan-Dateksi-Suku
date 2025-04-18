import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from PIL import Image
import os

# Konfigurasi
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
detector = MTCNN()

# Muat model dengan try-except untuk debug
try:
    feature_model = load_model('feature_extractor.h5')
    etnis_model = load_model('etnis_detection_model.h5')
    # Kompilasi manual untuk atasi warning
    feature_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    etnis_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    st.write("Model berhasil dimuat dan dikompilasi!")
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

df = pd.read_csv('all_dataset_metadata(isyan).csv')
suku_classes = df['suku'].unique()

def detect_face(image):
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if not faces:
        return None, None
    x, y, w, h = faces[0]['box']
    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, (128, 128)).astype('float32') / 255.0
    return face, (x, y, w, h)

def extract_embedding(face, model):
    face = np.expand_dims(face, axis=0)
    embedding = model.predict(face)
    return embedding[0]

# Inisialisasi session state untuk menyimpan foto kamera
if 'photo1' not in st.session_state:
    st.session_state.photo1 = None
if 'photo2' not in st.session_state:
    st.session_state.photo2 = None
if 'camera_step' not in st.session_state:
    st.session_state.camera_step = 1

# Streamlit UI
st.title("Pengenalan Wajah dan Deteksi Suku/Etnis")

# Pilih mode
mode = st.selectbox("Pilih Mode", ["", "Face Similarity", "Deteksi Suku/Etnis"])

if mode:
    # Pilih input
    input_type = st.radio("Pilih Input", ["File Upload", "Kamera"])

    if input_type == "Kamera":
        if mode == "Face Similarity":
            st.write(f"Ambil Foto {st.session_state.camera_step} dari 2")
            img_file = st.camera_input(f"Ambil Foto {st.session_state.camera_step}")
            if img_file:
                if st.session_state.camera_step == 1:
                    st.session_state.photo1 = img_file
                    st.session_state.camera_step = 2
                    st.experimental_rerun()
                elif st.session_state.camera_step == 2:
                    st.session_state.photo2 = img_file
            # Tampilkan tombol reset jika sudah ada foto
            if st.session_state.photo1 or st.session_state.photo2:
                if st.button("Reset Foto"):
                    st.session_state.photo1 = None
                    st.session_state.photo2 = None
                    st.session_state.camera_step = 1
                    st.experimental_rerun()
        else:  # Deteksi Suku/Etnis
            img_file = st.camera_input("Ambil Foto")
    else:  # File Upload
        if mode == "Face Similarity":
            img_file1 = st.file_uploader("Unggah Gambar 1", type=["jpg", "png"], key="file1")
            img_file2 = st.file_uploader("Unggah Gambar 2", type=["jpg", "png"], key="file2")
        else:
            img_file1 = st.file_uploader("Unggah Gambar", type=["jpg", "png"])
            img_file2 = None

    if st.button("Proses"):
        if mode == "Face Similarity":
            # Ambil dua gambar
            if input_type == "Kamera" and st.session_state.photo1 and st.session_state.photo2:
                img1 = Image.open(st.session_state.photo1)
                img2 = Image.open(st.session_state.photo2)
            elif input_type == "File Upload" and img_file1 and img_file2:
                img1 = Image.open(img_file1)
                img2 = Image.open(img_file2)
            else:
                st.error("Harap masukkan dua gambar!")
                st.stop()

            # Simpan file
            file1_path = os.path.join(UPLOAD_FOLDER, 'file1.jpg')
            file2_path = os.path.join(UPLOAD_FOLDER, 'file2.jpg')
            img1.save(file1_path)
            img2.save(file2_path)

            # Proses face similarity
            face1, _ = detect_face(img1)
            face2, _ = detect_face(img2)
            if face1 is None or face2 is None:
                st.error("Wajah tidak terdeteksi di salah satu gambar!")
            else:
                emb1 = extract_embedding(face1, feature_model)
                emb2 = extract_embedding(face2, feature_model)
                score = cosine_similarity([emb1], [emb2])[0][0]
                is_match = score > 0.6
                st.write(f"Similarity: {(score * 100):.2f}%")
                st.write(f"Status: {'Match' if is_match else 'Tidak Match'}")
                col1, col2 = st.columns(2)
                col1.image(img1, caption="Wajah 1", use_column_width=True)
                col2.image(img2, caption="Wajah 2", use_column_width=True)
                # Reset kamera setelah pemrosesan
                if input_type == "Kamera":
                    st.session_state.photo1 = None
                    st.session_state.photo2 = None
                    st.session_state.camera_step = 1

        elif mode == "Deteksi Suku/Etnis" and (img_file or img_file1):
            # Gunakan kamera atau file
            img = Image.open(img_file if img_file else img_file1)
            file_path = os.path.join(UPLOAD_FOLDER, 'file.jpg')
            img.save(file_path)

            # Proses deteksi suku
            face, _ = detect_face(img)
            if face is None:
                st.error("Wajah tidak terdeteksi!")
            else:
                face_processed = np.expand_dims(face, axis=0)
                probs = etnis_model.predict(face_processed)[0]
                predicted_class = suku_classes[np.argmax(probs)]
                st.write(f"Suku: {predicted_class}")
                st.write("Probabilitas:")
                for suku, prob in zip(suku_classes, probs):
                    st.write(f"{suku}: {(prob * 100):.2f}%")
                st.image(img, caption="Wajah", use_column_width=True)
        else:
            st.error("Masukkan gambar yang diperlukan!")