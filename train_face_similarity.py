import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mtcnn import MTCNN
import os

# Fungsi deteksi wajah
detector = MTCNN()
def detect_face(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if not faces:
        return None
    x, y, w, h = faces[0]['box']
    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, (128, 128))
    return face

# Muat dataset
df = pd.read_csv('all_dataset_metadata(isyan).csv')
valid_images = []
valid_labels = []

# Preprocess gambar
for idx, row in df.iterrows():
    face = detect_face(row['path_gambar'])
    if face is not None:
        valid_images.append(face)
        valid_labels.append(row['nama'])
        # Simpan gambar yang sudah diproses
        os.makedirs('processed_faces', exist_ok=True)
        cv2.imwrite(f'processed_faces/{idx}.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

# Update dataframe
processed_df = pd.DataFrame({
    'image_path': [f'processed_faces/{i}.jpg' for i in range(len(valid_images))],
    'nama': valid_labels
})
processed_df.to_csv('processed_faces.csv', index=False)

# Buat model CNN
num_classes = len(set(valid_labels))
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_dataframe(
    processed_df,
    x_col='image_path',
    y_col='nama',
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_dataframe(
    processed_df,
    x_col='image_path',
    y_col='nama',
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Latih model
model.fit(train_generator, validation_data=validation_generator, epochs=10)
model.save('face_similarity_model.h5')

# Simpan model ekstraksi fitur (tanpa lapisan output)
feature_model = Sequential(model.layers[:-1])
feature_model.save('feature_extractor.h5')