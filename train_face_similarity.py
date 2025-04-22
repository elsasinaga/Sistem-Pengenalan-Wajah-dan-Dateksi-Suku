import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import os

# Fungsi Contrastive Loss
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Fungsi Cosine Distance
def cosine_distance(vects):
    x, y = vects
    x = tf.keras.backend.l2_normalize(x, axis=1)
    y = tf.keras.backend.l2_normalize(y, axis=1)
    cosine_sim = tf.reduce_sum(x * y, axis=1, keepdims=True)
    return 1.0 - cosine_sim  # Cosine distance = 1 - cosine similarity

# Buat base network untuk Siamese Network
def create_base_network(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-15]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    # Normalisasi embedding
    embedding = Dense(512, activation=None, dtype='float32')(x)
    embedding = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1))(embedding)
    return Model(inputs=base_model.input, outputs=embedding)

# Buat Siamese Network
def create_siamese_model(input_shape):
    base_model = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_model(input_a)
    processed_b = base_model(input_b)
    distance = Lambda(cosine_distance)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    return model

# Muat dataset
train_df = pd.read_csv('training.csv', sep=';')
val_df = pd.read_csv('validation.csv', sep=';')

# Verifikasi kolom
expected_columns = ['path', 'nama', 'suku', 'ekspresi', 'sudut', 'jarak', 'pencahayaan']
for df, name in [(train_df, 'Training'), (val_df, 'Validation')]:
    if not all(col in df.columns for col in expected_columns):
        missing = [col for col in expected_columns if col not in df.columns]
        raise ValueError(f"Kolom {missing} tidak ditemukan di {name} CSV")
    counts = df['nama'].value_counts()
    if any(counts < 5):
        print(f"Peringatan: Beberapa individu di {name} memiliki <5 gambar: {counts[counts < 5]}")
    print(f"{name}: {len(df)} gambar, {len(counts)} individu")

# Buat pasangan gambar (tingkatkan pasangan negatif)
def create_pairs(df, max_positive_per_person=10, max_negative_per_person=10):  # Tingkatkan jumlah pasangan
    pairs = []
    labels = []
    name_groups = df.groupby('nama')
    names = list(name_groups.groups.keys())
    
    # Pasangan positif
    for name in names:
        group = name_groups.get_group(name)
        if len(group) > 1:
            indices = np.random.choice(len(group), size=min(len(group), max_positive_per_person*2), replace=False)
            selected_pairs = 0
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    if selected_pairs >= max_positive_per_person:
                        break
                    pairs.append([group.iloc[indices[i]]['path'], group.iloc[indices[j]]['path']])
                    labels.append(1.0)
                    selected_pairs += 1
    
    # Pasangan negatif
    for name in names:
        group = name_groups.get_group(name)
        other_groups = df[df['nama'] != name]
        for i in range(min(len(group), max_negative_per_person)):
            other = other_groups.sample(n=1)
            pairs.append([group.iloc[i]['path'], other.iloc[0]['path']])
            labels.append(0.0)
    
    return np.array(pairs), np.array(labels, dtype=np.float32)

train_pairs, train_labels = create_pairs(train_df, max_positive_per_person=10, max_negative_per_person=10)
val_pairs, val_labels = create_pairs(val_df, max_positive_per_person=10, max_negative_per_person=10)
print(f"Jumlah pasangan training: {len(train_pairs)}, validation: {len(val_pairs)}")

# Fungsi untuk memuat dan preprocess gambar
def load_and_preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Gagal memuat gambar: {img_path}")
        # Normalisasi pencahayaan
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        l = cv2.equalizeHist(l)
        img_lab = cv2.merge((l, a, b))
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        img = cv2.resize(img_rgb, (255, 255)).astype('float32') / 255.0
        return img
    except Exception as e:
        print(f"Error memuat gambar {img_path}: {e}")
        return None

# Fungsi augmentasi
def augment_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.random_brightness(img, 0.4)  # Tingkatkan augmentasi
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    img = tf.image.random_hue(img, 0.1)
    return img.numpy()

# Generator untuk memuat data
def data_generator(pairs, labels, batch_size, augment=False, shuffle=True):
    num_samples = len(pairs)
    indices = np.arange(num_samples)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            img1_batch = []
            img2_batch = []
            label_batch = []
            
            for idx in batch_indices:
                img1 = load_and_preprocess_image(pairs[idx][0])
                img2 = load_and_preprocess_image(pairs[idx][1])
                
                if img1 is None or img2 is None:
                    continue
                    
                if augment:
                    img1 = augment_image(img1)
                    img2 = augment_image(img2)
                    
                img1_batch.append(img1)
                img2_batch.append(img2)
                label_batch.append(labels[idx])
            
            if len(img1_batch) > 0:
                yield [np.array(img1_batch), np.array(img2_batch)], np.array(label_batch)

# Buat Siamese Model
input_shape = (255, 255, 3)
model = create_siamese_model(input_shape)

# Kompilasi model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=contrastive_loss, metrics=['accuracy'])

# Latih model
batch_size = 8  # Tingkatkan batch size untuk stabilitas
steps_per_epoch = max(1, len(train_pairs) // batch_size)
validation_steps = max(1, len(val_pairs) // batch_size)

model.fit(
    data_generator(train_pairs, train_labels, batch_size, augment=True, shuffle=True),
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(val_pairs, val_labels, batch_size, augment=False, shuffle=False),
    validation_steps=validation_steps,
    epochs=10  # Tingkatkan epoch
)

# Simpan model embedding
base_network = model.layers[2]
base_network.save('feature_extractor_siamese.h5')