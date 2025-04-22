import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import Counter

# Fungsi untuk memuat dan preprocess gambar
def load_and_preprocess_image(img_path, target_size=(255, 255)):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_processed = preprocess_input(img_resized)
    return img_processed

def load_data(csv_path, base_dir):
    df = pd.read_csv(csv_path, sep=';')
    images, labels = [], []
    for idx, row in df.iterrows():
        img_path = os.path.join(base_dir, row['path'])
        if os.path.exists(img_path):
            img = load_and_preprocess_image(img_path)
            images.append(img)
            labels.append(row['suku'])
        else:
            print(f"Gambar tidak ditemukan: {img_path}")
    return np.array(images), np.array(labels)

# Paths
base_dir = 'D:/SEM4/PCD_BISA/tubes'
train_csv = 'training.csv'
val_csv = 'validation.csv'
test_csv = 'testing.csv'

# Load all data
X_train_full, y_train_full = load_data(train_csv, base_dir)
X_val_ext, y_val_ext = load_data(val_csv, base_dir)
X_test, y_test = load_data(test_csv, base_dir)

# Gabungkan semua label untuk LabelEncoder
all_labels = np.concatenate([y_train_full, y_val_ext, y_test])
le = LabelEncoder()
le.fit(all_labels)

# Encode semua label
y_train_enc = le.transform(y_train_full)
y_val_ext_enc = le.transform(y_val_ext)
y_test_enc = le.transform(y_test)

# Split validation internal dari training set
X_train, X_val, y_train_split, y_val_split = train_test_split(
    X_train_full, y_train_enc, test_size=0.2, stratify=y_train_enc, random_state=42
)

# One-hot encoding
num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train_split, num_classes)
y_val_cat = to_categorical(y_val_split, num_classes)
y_val_ext_cat = to_categorical(y_val_ext_enc, num_classes)
y_test_cat = to_categorical(y_test_enc, num_classes)

# Simpan LabelEncoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Hitung class weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_split), y=y_train_split)
class_weights = dict(enumerate(class_weights))

# Augmentasi data
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.5,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)
datagen.fit(X_train)

# MobileNetV2 + custom layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(255, 255, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Fine-tuning
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=16),
    validation_data=(X_val, y_val_cat),
    epochs=50,
    verbose=1,
    class_weight=class_weights,
    callbacks=[early_stopping, lr_scheduler]
)

model.save('ethnicity_model.h5')

# Evaluasi
def evaluate_model(X, y_cat, dataset_name):
    y_true = np.argmax(y_cat, axis=1)
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(f"\nEvaluasi pada {dataset_name}:")
    print(classification_report(y_true, y_pred_classes, target_names=le.classes_))
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({dataset_name})')
    plt.savefig(f'confusion_matrix_{dataset_name}.png')
    plt.show()

evaluate_model(X_val_ext, y_val_ext_cat, "Validation Eksternal")
evaluate_model(X_test, y_test_cat, "Testing")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()
plt.savefig('training_history.png')
plt.show()
