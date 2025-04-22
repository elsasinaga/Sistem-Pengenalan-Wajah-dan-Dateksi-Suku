import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
from PIL import Image
import os
import seaborn as sns
import matplotlib.pyplot as plt
import io
import pickle

# Konfigurasi
UPLOAD_FOLDER = 'pages/Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
IMG_SIZE = (255, 255)

# Muat model untuk Face Similarity
try:
    feature_model = load_model('feature_extractor_siamese.h5')
    feature_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
except Exception as e:
    st.error(f"Error saat memuat model Face Similarity: {e}")
    st.stop()

# Muat model dan LabelEncoder untuk Deteksi Suku/Etnis
try:
    ethnicity_model = load_model('ethnicity_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
except Exception as e:
    st.error(f"Error saat memuat model Deteksi Suku/Etnis: {e}")
    st.stop()

# Muat metadata untuk Face Similarity
test_df = pd.read_csv('testing.csv', sep=';')

# Fungsi untuk memuat gambar (Face Similarity)
def load_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE).astype('float32') / 255.0
    return img

# Fungsi ekstraksi embedding (Face Similarity)
def extract_embedding(face, model):
    face = np.expand_dims(face, axis=0)
    embedding = model.predict(face)
    return embedding[0]

# Fungsi Euclidean Distance (Face Similarity)
def euclidean_distance_inference(emb1, emb2):
    return np.sqrt(np.sum((emb1 - emb2) ** 2))

# Konversi jarak ke skor kemiripan (Face Similarity)
def distance_to_similarity(distance, max_distance=5.0):
    similarity = max(0, 1 - (distance / max_distance))
    return similarity

# Fungsi evaluasi metrik (Face Similarity)
@st.cache_data
def evaluate_metrics():
    distances = []
    labels = []
    threshold = 1.5
    for i in range(len(test_df)):
        for j in range(i+1, len(test_df)):
            img1 = cv2.imread(test_df['path'][i])
            img2 = cv2.imread(test_df['path'][j])
            if img1 is None or img2 is None:
                continue
            face1 = load_image(Image.fromarray(img1))
            face2 = load_image(Image.fromarray(img2))
            emb1 = extract_embedding(face1, feature_model)
            emb2 = extract_embedding(face2, feature_model)
            dist = euclidean_distance_inference(emb1, emb2)
            distances.append(dist)
            labels.append(1 if test_df['nama'][i] == test_df['nama'][j] else 0)
    
    predictions = [1 if d < threshold else 0 for d in distances]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    tar = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    fig, ax = plt.subplots()
    sns.histplot([d for d, l in zip(distances, labels) if l == 1], label='Same Identity', color='blue', alpha=0.5, ax=ax)
    sns.histplot([d for d, l in zip(distances, labels) if l == 0], label='Different Identity', color='red', alpha=0.5, ax=ax)
    ax.legend()
    ax.set_title('Distribusi Jarak Euclidean (Same vs Different Identity)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return precision, recall, f1, tar, far, frr, buf

# Visualisasi pasangan salah (Face Similarity)
def visualize_errors():
    if os.path.exists('errors.csv'):
        errors_df = pd.read_csv('errors.csv', sep=';')
        st.subheader("Contoh Pasangan Salah")
        for _, row in errors_df.head(2).iterrows():
            if 'image1' in row and 'image2' in row:
                img1 = Image.open(row['image1'])
                img2 = Image.open(row['image2'])
                col1, col2 = st.columns(2)
                col1.image(img1, caption=f"Gambar 1: {row['true']}", use_column_width=True)
                col2.image(img2, caption=f"Gambar 2: {row['predicted']}", use_column_width=True)
                st.write(f"Skor Jarak: {row.get('distance', 'N/A'):.4f}")

# Fungsi untuk preprocess gambar (Deteksi Suku/Etnis)
def preprocess_image(img_rgb, target_size=(255, 255)):
    detector = MTCNN()
    faces = detector.detect_faces(img_rgb)
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        x, y = max(x, 0), max(y, 0)
        face = img_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, target_size) / 255.0
        return face
    else:
        return None

# Inisialisasi session state untuk kamera (Face Similarity)
if 'photo1' not in st.session_state:
    st.session_state.photo1 = None
if 'photo2' not in st.session_state:
    st.session_state.photo2 = None
if 'camera_step' not in st.session_state:
    st.session_state.camera_step = 1

# Sidebar untuk memilih fitur
st.sidebar.title("Pilih Fitur")
feature = st.sidebar.selectbox("Fitur", ["Face Similarity", "Deteksi Suku/Etnis"])

# UI Utama
if feature == "Face Similarity":
    st.title("Pengenalan Wajah (Face Similarity)")
    
    # Pilih input
    input_type = st.radio("Pilih Input", ["File Upload", "Kamera"])
    
    if input_type == "Kamera":
        st.write(f"Ambil Foto {st.session_state.camera_step} dari 2")
        img_file = st.camera_input(f"Ambil Foto {st.session_state.camera_step}")
        if img_file:
            if st.session_state.camera_step == 1:
                st.session_state.photo1 = img_file
                st.session_state.camera_step = 2
                st.experimental_rerun()
            elif st.session_state.camera_step == 2:
                st.session_state.photo2 = img_file
        if st.session_state.photo1 or st.session_state.photo2:
            if st.button("Reset Foto"):
                st.session_state.photo1 = None
                st.session_state.photo2 = None
                st.session_state.camera_step = 1
                st.experimental_rerun()
    else:
        img_file1 = st.file_uploader("Unggah Gambar 1", type=["jpg", "png"], key="file1")
        img_file2 = st.file_uploader("Unggah Gambar 2", type=["jpg", "png"], key="file2")

    if st.button("Proses"):
        if input_type == "Kamera" and st.session_state.photo1 and st.session_state.photo2:
            img1 = Image.open(st.session_state.photo1)
            img2 = Image.open(st.session_state.photo2)
        elif input_type == "File Upload" and img_file1 and img_file2:
            img1 = Image.open(img_file1)
            img2 = Image.open(img_file2)
        else:
            st.error("Harap masukkan dua gambar!")
            st.stop()

        file1_path = os.path.join(UPLOAD_FOLDER, 'file1.jpg')
        file2_path = os.path.join(UPLOAD_FOLDER, 'file2.jpg')
        img1.save(file1_path)
        img2.save(file2_path)

        face1 = load_image(img1)
        face2 = load_image(img2)
        emb1 = extract_embedding(face1, feature_model)
        emb2 = extract_embedding(face2, feature_model)
        dist = euclidean_distance_inference(emb1, emb2)
        threshold = 1.5
        similarity_score = distance_to_similarity(dist)
        is_match = dist < threshold
        st.write(f"Skor Kemiripan: {similarity_score:.4f} (Jarak Euclidean: {dist:.4f})")
        st.write(f"Status: {'Match' if is_match else 'Tidak Match'}")
        col1, col2 = st.columns(2)
        col1.image(img1, caption="Wajah 1", use_column_width=True)
        col2.image(img2, caption="Wajah 2", use_column_width=True)

        # Tampilkan metrik evaluasi
        precision, recall, f1, tar, far, frr, dist_buf = evaluate_metrics()
        st.subheader("Metrik Evaluasi Model")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1-Score: {f1:.4f}")
        st.write(f"TAR: {tar:.4f}")
        st.write(f"FAR: {far:.4f}")
        st.write(f"FRR: {frr:.4f}")
        st.image(dist_buf, caption="Distribusi Jarak Euclidean")

        # Visualisasi pasangan salah
        visualize_errors()

        if input_type == "Kamera":
            st.session_state.photo1 = None
            st.session_state.photo2 = None
            st.session_state.camera_step = 1

elif feature == "Deteksi Suku/Etnis":
    st.title("Deteksi Suku/Etnis")
    st.write("Upload gambar wajah atau gunakan kamera untuk mendeteksi suku/etnis.")

    # Pilih input
    input_method = st.radio("Pilih metode input:", ("Upload File", "Kamera"))

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            face = preprocess_image(img_rgb)
            if face is not None:
                face_input = np.expand_dims(face, axis=0)
                pred = ethnicity_model.predict(face_input)
                pred_class = le.inverse_transform([np.argmax(pred)])[0]
                pred_prob = pred[0][np.argmax(pred)]
                
                st.image(img_rgb, caption="Gambar Input", use_column_width=True)
                st.write(f"**Suku/Etnis:** {pred_class}")
                st.write(f"**Confidence Score:** {pred_prob:.2%}")
                
                # Visualisasi confidence score dengan grafik
                prob_dict = {le.classes_[i]: pred[0][i] for i in range(len(le.classes_))}
                st.write("**Grafik Probabilitas per Suku:**")
                st.bar_chart(prob_dict)

                # Tampilkan probabilitas detail per suku dalam persentase
                st.write("**Detail Probabilitas per Suku:**")
                for suku, prob in prob_dict.items():
                    st.write(f"{suku}: {prob:.2%}")
                
            else:
                st.error("Wajah tidak terdeteksi dalam gambar.")

    elif input_method == "Kamera":
        picture = st.camera_input("Ambil foto")
        if picture is not None:
            file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            face = preprocess_image(img_rgb)
            if face is not None:
                face_input = np.expand_dims(face, axis=0)
                pred = ethnicity_model.predict(face_input)
                pred_class = le.inverse_transform([np.argmax(pred)])[0]
                pred_prob = pred[0][np.argmax(pred)]
                
                st.image(img_rgb, caption="Gambar dari Kamera", use_column_width=True)
                st.write(f"**Suku/Etnis:** {pred_class}")
                st.write(f"**Confidence Score:** {pred_prob:.2%}")
                
                # Visualisasi confidence score dengan grafik
                prob_dict = {le.classes_[i]: pred[0][i] for i in range(len(le.classes_))}
                st.write("**Grafik Probabilitas per Suku:**")
                st.bar_chart(prob_dict)

                # Tampilkan probabilitas detail per suku dalam persentase
                st.write("**Detail Probabilitas per Suku:**")
                for suku, prob in prob_dict.items():
                    st.write(f"{suku}: {prob:.2%}")
                
            else:
                st.error("Wajah tidak terdeteksi dalam gambar.")