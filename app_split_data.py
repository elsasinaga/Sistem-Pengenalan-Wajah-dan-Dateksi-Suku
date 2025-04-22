import streamlit as st
import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance
try:
    import tensorflow as tf
except ModuleNotFoundError as e:
    st.error(f"Gagal mengimpor TensorFlow: {e}. Pastikan TensorFlow terinstal dengan benar.")
    st.stop()

try:
    from mtcnn import MTCNN
except ModuleNotFoundError as e:
    st.error(f"Gagal mengimpor MTCNN: {e}. Pastikan MTCNN terinstal dengan `pip install mtcnn`.")
    st.stop()

import csv

# Initialize session state
if 'image_attributes' not in st.session_state:
    st.session_state.image_attributes = {}
if 'show_form' not in st.session_state:
    st.session_state.show_form = False

# Fungsi untuk menormalkan path file dan case
def normalize_path(path):
    return os.path.normpath(path).replace('\\', '/').lower()

# Fungsi untuk membuat direktori
def create_directory(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        st.error(f"Gagal membuat direktori: {e}")

# Fungsi untuk menyimpan file
def save_file(uploaded_file, folder_path, file_name, attributes):
    try:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        normalized_path = normalize_path(file_path)
        st.session_state.image_attributes[normalized_path] = attributes
        return file_path
    except Exception as e:
        st.error(f"Gagal menyimpan file {file_name}: {e}")
        return None

# Fungsi untuk menghapus folder
def clear_directory(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    except Exception as e:
        st.error(f"Gagal menghapus/membuat direktori {path}: {e}")

# Fungsi untuk mendeteksi wajah
def detect_face(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        detector = MTCNN()
        faces = detector.detect_faces(img_np)
        
        if len(faces) > 0:
            x, y, width, height = faces[0]['box']
            padding = int(max(width, height) * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = width + 2 * padding
            height = height + 2 * padding
            
            crop_size = max(width, height)
            center_x = x + width // 2
            center_y = y + height // 2
            left = max(0, center_x - crop_size // 2)
            top = max(0, center_y - crop_size // 2)
            right = min(img_np.shape[1], left + crop_size)
            bottom = min(img_np.shape[0], top + crop_size)
            
            if right - left < crop_size:
                left = max(0, right - crop_size)
            if bottom - top < crop_size:
                top = max(0, bottom - crop_size)
            
            face_img = img_np[top:bottom, left:right]
            if face_img.size == 0:
                st.warning(f"Area wajah tidak valid pada {image_path}")
                return None
            
            face_img_pil = Image.fromarray(face_img)
            if face_img.shape[0] != face_img.shape[1]:
                square_size = min(face_img.shape[0], face_img.shape[1])
                left = (face_img.shape[1] - square_size) // 2
                top = (face_img.shape[0] - square_size) // 2
                face_img_pil = face_img_pil.crop((left, top, left + square_size, top + square_size))
            
            return face_img_pil
        else:
            st.warning(f"Tidak ada wajah terdeteksi pada {image_path}")
            return None
    except Exception as e:
        st.error(f"Gagal mendeteksi wajah pada {image_path}: {e}")
        return None

# Fungsi untuk resize gambar
def resize_image(image, target_size=(255, 255)):
    try:
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            img = image.convert('RGB')
        img = img.resize(target_size, resample=Image.Resampling.BICUBIC)
        return img
    except Exception as e:
        st.error(f"Gagal meresize gambar: {e}")
        return None

# Fungsi untuk augmentasi gambar
def augment_image(image, original_attributes):
    augmented_images = []
    augmented_attributes = []
    try:
        img = Image.fromarray(image).convert('RGB')
        rotated = img.rotate(15, resample=Image.Resampling.BICUBIC, expand=True)
        rotated = resize_image(rotated, (255, 255))
        augmented_images.append(np.array(rotated))
        augmented_attributes.append({
            'ekspresi': original_attributes['ekspresi'],
            'sudut': 'Rotasi 15',
            'jarak': original_attributes['jarak'],
            'pencahayaan': original_attributes['pencahayaan']
        })
        bright = ImageEnhance.Brightness(img).enhance(1.2)
        contrast = ImageEnhance.Contrast(bright).enhance(1.2)
        contrast = resize_image(contrast, (255, 255))
        augmented_images.append(np.array(contrast))
        augmented_attributes.append({
            'ekspresi': original_attributes['ekspresi'],
            'sudut': original_attributes['sudut'],
            'jarak': original_attributes['jarak'],
            'pencahayaan': 'Terang'
        })
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped = resize_image(flipped, (255, 255))
        augmented_images.append(np.array(flipped))
        augmented_attributes.append({
            'ekspresi': original_attributes['ekspresi'],
            'sudut': 'Flipped',
            'jarak': original_attributes['jarak'],
            'pencahayaan': original_attributes['pencahayaan']
        })
        img_np = np.array(img)
        gaussian_noise = np.random.normal(0, 15, img_np.shape)
        noisy_image = np.clip(img_np + gaussian_noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_image).convert('RGB')
        noisy_img = resize_image(noisy_img, (255, 255))
        augmented_images.append(np.array(noisy_img))
        augmented_attributes.append({
            'ekspresi': original_attributes['ekspresi'],
            'sudut': original_attributes['sudut'],
            'jarak': original_attributes['jarak'],
            'pencahayaan': 'Gaussian Noise'
        })
    except Exception as e:
        st.error(f"Gagal melakukan augmentasi: {e}")
    return augmented_images, augmented_attributes

# Fungsi untuk menentukan ekspresi
def infer_expression(file_name):
    file_name = file_name.lower()
    if 'varian1' in file_name:
        return 'Senyum'
    elif 'varian2' in file_name:
        return 'Serius'
    elif 'varian3' in file_name:
        return 'Terkejut'
    elif 'varian4' in file_name:
        return 'Tertawa'
    return 'Senyum'

# Fungsi untuk menentukan atribut
def infer_attributes(file_name):
    file_name = file_name.lower()
    if 'varian1' in file_name:
        return {'ekspresi': 'Senyum', 'sudut': 'Frontal', 'jarak': 'Jauh', 'pencahayaan': 'Terang'}
    elif 'varian2' in file_name:
        return {'ekspresi': 'Serius', 'sudut': 'Frontal', 'jarak': 'Dekat', 'pencahayaan': 'Redup'}
    elif 'varian3' in file_name:
        return {'ekspresi': 'Terkejut', 'sudut': 'Atas', 'jarak': 'Sedang', 'pencahayaan': 'Redup'}
    elif 'varian4' in file_name:
        return {'ekspresi': 'Tertawa', 'sudut': 'Bawah', 'jarak': 'Sedang', 'pencahayaan': 'Terang'}
    return {'ekspresi': 'Senyum', 'sudut': 'Frontal', 'jarak': 'Sedang', 'pencahayaan': 'Terang'}

# Fungsi untuk menetapkan atribut untuk file yang sudah ada
def assign_attributes_to_existing_files():
    uploads_dir = "Uploads"
    if not os.path.exists(uploads_dir):
        return
    for suku in os.listdir(uploads_dir):
        suku_path = os.path.join(uploads_dir, suku)
        if not os.path.isdir(suku_path):
            continue
        for person in os.listdir(suku_path):
            person_path = os.path.join(suku_path, person)
            if not os.path.isdir(person_path):
                continue
            for file_name in os.listdir(person_path):
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(person_path, file_name)
                    normalized_path = normalize_path(file_path)
                    if normalized_path not in st.session_state.image_attributes:
                        attributes = infer_attributes(file_name)
                        st.session_state.image_attributes[normalized_path] = attributes

# Fungsi untuk generate CSV
def generate_csv():
    try:
        for split in ['training', 'validation', 'testing']:
            csv_file = f"{split}.csv"
            with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['path', 'nama', 'suku', 'ekspresi', 'sudut', 'jarak', 'pencahayaan'])
                
                if not os.path.exists(split):
                    continue
                for suku in os.listdir(split):
                    suku_path = os.path.join(split, suku)
                    if not os.path.isdir(suku_path):
                        continue
                    for nama in os.listdir(suku_path):
                        nama_path = os.path.join(suku_path, nama)
                        if not os.path.isdir(nama_path):
                            continue
                        for file_name in os.listdir(nama_path):
                            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                                file_path = os.path.join(nama_path, file_name)
                                normalized_path = normalize_path(file_path)
                                attributes = st.session_state.image_attributes.get(normalized_path)
                                if attributes is None:
                                    attributes = infer_attributes(file_name)
                                    st.session_state.image_attributes[normalized_path] = attributes
                                writer.writerow([file_path, nama, suku, attributes['ekspresi'], 
                                                attributes['sudut'], attributes['jarak'], attributes['pencahayaan']])
        
            if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
                st.success(f"File CSV '{csv_file}' berhasil digenerated!")
            else:
                st.warning(f"File CSV '{csv_file}' kosong atau tidak dibuat karena tidak ada data di {split}.")
    except Exception as e:
        st.error(f"Gagal mengenerate CSV: {e}")

# Fungsi pembagian proporsi data
def split_counts(total):
    if total < 3:
        return total, 0, 0
    train = round(total * 0.7)
    val = round(total * 0.15)
    test = total - train - val
    while train + val + test < total:
        train += 1
    if val == 0:
        val = 1
        train -= 1
    if test == 0:
        test = 1
        train -= 1
    return train, val, test

# Fungsi split data
def split_data_by_person(resize=True, augment=True):
    try:
        uploads_dir = "Uploads"
        if not os.path.exists(uploads_dir):
            st.error("Folder uploads tidak ditemukan!")
            return

        # Progress bar di luar kolom, 50% width
        with st.container(border=False):
            progress_bar = st.progress(0)
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Tetapkan atribut untuk file yang sudah ada
        assign_attributes_to_existing_files()

        for split in ["training", "validation", "testing"]:
            clear_directory(split)

        suku_folders = [suku for suku in os.listdir(uploads_dir) if os.path.isdir(os.path.join(uploads_dir, suku))]
        total_suku = len(suku_folders)

        # Output hasil split, 50% width
        with st.container(border=False):
            for idx, suku in enumerate(suku_folders):
                suku_path = os.path.join(uploads_dir, suku)
                person_folders = [person for person in os.listdir(suku_path) if os.path.isdir(os.path.join(suku_path, person))]
                if not person_folders:
                    st.warning(f"Tidak ada folder nama ditemukan untuk suku {suku}!")
                    continue

                random.seed(42)
                random.shuffle(person_folders)
                total = len(person_folders)
                train_count, val_count, test_count = split_counts(total)

                train_persons = person_folders[:train_count]
                val_persons = person_folders[train_count:train_count + val_count]
                test_persons = person_folders[train_count + val_count:]

                def copy_persons(person_list, split):
                    split_suku_path = os.path.join(split, suku)
                    create_directory(split_suku_path)
                    for person in person_list:
                        src_path = os.path.join(suku_path, person)
                        dst_path = os.path.join(split_suku_path, person)
                        create_directory(dst_path)
                        for file_name in os.listdir(src_path):
                            src_file = os.path.join(src_path, file_name)
                            dst_file = os.path.join(dst_path, file_name)
                            if os.path.isfile(src_file):
                                face_image = detect_face(src_file)
                                if face_image is not None:
                                    face_img_pil = resize_image(face_image, (255, 255))
                                    face_img_pil.save(dst_file, quality=95)
                                    normalized_src = normalize_path(src_file)
                                    normalized_dst = normalize_path(dst_file)
                                    st.session_state.image_attributes[normalized_dst] = st.session_state.image_attributes[normalized_src]
                                    if augment and split == "training":
                                        face_np = np.array(face_image)
                                        augmented_list, aug_attributes = augment_image(face_np, 
                                            st.session_state.image_attributes[normalized_dst])
                                        for idx, (aug_img, aug_attr) in enumerate(zip(augmented_list, aug_attributes)):
                                            aug_path = os.path.join(dst_path, f"aug{idx+1}_{file_name}")
                                            aug_img_pil = Image.fromarray(aug_img).convert('RGB')
                                            aug_img_pil = resize_image(aug_img_pil, (255, 255))
                                            aug_img_pil.save(aug_path, quality=95)
                                            normalized_aug_path = normalize_path(aug_path)
                                            st.session_state.image_attributes[normalized_aug_path] = aug_attr

                copy_persons(train_persons, "training")
                copy_persons(val_persons, "validation")
                copy_persons(test_persons, "testing")

                st.markdown(f"<h3 style='margin: 0; text-align: center;'>Suku: {suku}</h3>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center;'>Total: {total} orang</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center;'>Training: {len(train_persons)} orang ({', '.join(train_persons)})</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center;'>Validation: {len(val_persons)} orang ({', '.join(val_persons)})</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center;'>Testing: {len(test_persons)} orang ({', '.join(test_persons)})</div>", unsafe_allow_html=True)
                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

                progress_bar.progress((idx + 1) / total_suku)

            generate_csv()

            st.success("Split per orang dalam setiap suku berhasil dilakukan! Gambar di Uploads tetap ukuran asli, gambar di training/validation/testing di-crop tepat di wajah dan diresize ke 255x255 tanpa stretching.")

    except Exception as e:
        st.error(f"Gagal melakukan split data: {e}")
    finally:
        # Kosongkan progress bar setelah selesai
        progress_bar.empty()

# UI Streamlit
with st.container(border=False):
    st.markdown("""
        <style>
        .stButton > button {
            border-radius: 8px;
            padding: 10px;
            width: 100%;
        }
        .stForm {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
        }
        .stProgress {
            width: 150% !important;
            margin: 0 !important;
            position: relative;
            left: 0;
            transform: translateX(-50%);
            box-sizing: border-box;
        }
        .stProgress > div > div {
            margin: 0 !important;
            border-radius: 8px;
        }
        .stMarkdown, .stSuccess, .stWarning, .stError {
            width: 100% !important;
            margin: 0 !important;
            position: relative;
            left: 0;
            transform: translateX(-50%);
            text-align: center;
            box-sizing: border-box;
        }
        .st-emotion-cache-1wmy9hl, .st-emotion-cache-0, .st-emotion-cache-1r4qj8v {
            padding: 0 !important;
            margin: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Aplikasi Upload Foto Varian")
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Layout tombol dalam kolom
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        if st.button("Upload Foto", use_container_width=True):
            st.session_state.show_form = True
    with col2:
        if st.button("Reprocess, Split and Augment Images", use_container_width=True):
            st.session_state.show_form = False
            split_data_by_person(resize=True, augment=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Form upload
    if st.session_state.show_form:
        with st.form(key="upload_form"):
            st.subheader("Form Upload Foto")
            nama = st.text_input("Nama", placeholder="Masukkan nama")
            suku = st.selectbox("Suku", ["Batak", "Cina", "Jawa", "Palembang", "Papua", "Sunda"], placeholder="Pilih suku")
            foto_varian1 = st.file_uploader("Upload Foto 1: Tersenyum", type=["jpg", "jpeg", "png"], key="varian1")
            foto_varian2 = st.file_uploader("Upload Foto 2: Serius", type=["jpg", "jpeg", "png"], key="varian2")
            foto_varian3 = st.file_uploader("Upload Foto 3: Terkejut", type=["jpg", "jpeg", "png"], key="varian3")
            foto_varian4 = st.file_uploader("Upload Foto 4: Tertawa", type=["jpg", "jpeg", "png"], key="varian4")
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(label="Submit", use_container_width=True)

            if submit_button:
                if not nama:
                    st.error("Nama wajib diisi!")
                elif not suku:
                    st.error("Suku wajib dipilih!")
                elif not foto_varian1:
                    st.error("Foto Varian 1 wajib diunggah!")
                elif not foto_varian2:
                    st.error("Foto Varian 2 wajib diunggah!")
                elif not foto_varian3:
                    st.error("Foto Varian 3 wajib diunggah!")
                elif not foto_varian4:
                    st.error("Foto Varian 4 wajib diunggah!")
                else:
                    nama_normalized = nama.lower()
                    folder_path = os.path.join("Uploads", suku, nama_normalized)
                    create_directory(folder_path)
                    results = []
                    for i, (foto, ekspresi, sudut, jarak, pencahayaan) in enumerate([
                        (foto_varian1, 'Senyum', 'Frontal', 'Jauh', 'Terang'),
                        (foto_varian2, 'Serius', 'Frontal', 'Dekat', 'Redup'),
                        (foto_varian3, 'Terkejut', 'Atas', 'Sedang', 'Redup'),
                        (foto_varian4, 'Tertawa', 'Bawah', 'Sedang', 'Terang')
                    ]):
                        file_ext = os.path.splitext(foto.name)[1]
                        file_name = f"{nama_normalized}_varian{i+1}{file_ext}"
                        result = save_file(foto, folder_path, file_name, 
                                          {'ekspresi': ekspresi, 'sudut': sudut, 'jarak': jarak, 'pencahayaan': pencahayaan})
                        results.append(result)
                    if all(r is not None for r in results):
                        st.success(f"Folder {nama_normalized} telah berhasil dibuat!")
                        st.session_state.show_form = False
                    else:
                        st.error("Gagal menyimpan beberapa file, periksa log untuk detail.")