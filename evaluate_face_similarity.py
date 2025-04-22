from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

IMG_SIZE = (255, 255)
MODEL_PATH = 'feature_extractor_siamese.h5'

def load_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, IMG_SIZE).astype('float32') / 255.0
    return img

def extract_embedding(face, model):
    face = np.expand_dims(face, axis=0)
    embedding = model.predict(face)
    return embedding[0]

def euclidean_distance_inference(emb1, emb2):
    return np.sqrt(np.sum((emb1 - emb2) ** 2))

# Muat model dan data test
feature_model = load_model(MODEL_PATH)
test_df = pd.read_csv('testing.csv', sep=';')
distances = []
labels = []
pairs_info = []  # Untuk menyimpan pasangan gambar (untuk visualisasi TP/FP/FN/TN)

# Buat pasangan positif dan negatif
for i in range(len(test_df)):
    for j in range(i+1, len(test_df)):
        img1 = cv2.imread(test_df['path'][i])
        img2 = cv2.imread(test_df['path'][j])
        if img1 is None or img2 is None:
            continue
        face1 = load_image(img1)
        face2 = load_image(img2)
        emb1 = extract_embedding(face1, feature_model)
        emb2 = extract_embedding(face2, feature_model)
        dist = euclidean_distance_inference(emb1, emb2)
        distances.append(dist)
        label = 1 if test_df['nama'][i] == test_df['nama'][j] else 0
        labels.append(label)
        pairs_info.append({
            'image1': test_df['path'][i],
            'image2': test_df['path'][j],
            'distance': dist,
            'label': label
        })

# Hitung ROC dan threshold dengan FAR rendah
fpr, tpr, thresholds = roc_curve(labels, [-d for d in distances])
auc_score = auc(fpr, tpr)
far_threshold = thresholds[np.where(fpr < 0.05)[0][-1]] if np.any(fpr < 0.05) else thresholds[0]
print(f"Threshold dengan FAR < 0.05: {far_threshold}, AUC: {auc_score}")

# Hitung EER
eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
eer = fpr[eer_idx]
print(f"EER: {eer:.4f}")

# Visualisasi ROC curve
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter(fpr[eer_idx], tpr[eer_idx], marker='o', color='red', label=f'EER = {eer:.4f}')
plt.title('ROC Curve Face Similarity')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

# Visualisasi distribusi jarak
plt.figure()
sns.histplot([d for d, l in zip(distances, labels) if l == 1], label='Same Identity', color='blue', alpha=0.5)
sns.histplot([d for d, l in zip(distances, labels) if l == 0], label='Different Identity', color='red', alpha=0.5)
plt.legend()
plt.title('Distribusi Jarak Euclidean (Same vs Different Identity)')
plt.savefig('distance_distribution.png')
plt.close()

# Hitung metrik
predictions = [1 if d < far_threshold else 0 for d in distances]
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
cm = confusion_matrix(labels, predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix Face Similarity')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_facesimilarity.png')
plt.close()

# Hitung TAR, FAR, FRR
tn, fp, fn, tp = cm.ravel()
tar = tp / (tp + fn) if (tp + fn) > 0 else 0
far = fp / (fp + tn) if (fp + tn) > 0 else 0
frr = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f"TAR: {tar:.4f}, FAR: {far:.4f}, FRR: {frr:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Simpan contoh TP, FP, FN, TN (pilih masing-masing 1 contoh)
examples = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
for pair, pred in zip(pairs_info, predictions):
    true_label = pair['label']
    if true_label == 1 and pred == 1:
        examples['TP'].append(pair)
    elif true_label == 0 and pred == 1:
        examples['FP'].append(pair)
    elif true_label == 1 and pred == 0:
        examples['FN'].append(pair)
    elif true_label == 0 and pred == 0:
        examples['TN'].append(pair)

# Visualisasi contoh TP, FP, FN, TN
for category, pairs in examples.items():
    if pairs:
        pair = pairs[0]  # Ambil contoh pertama
        img1 = cv2.imread(pair['image1'])
        img2 = cv2.imread(pair['image2'])
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img1_rgb)
        plt.title('Gambar 1')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img2_rgb)
        plt.title(f'Gambar 2\nDistance: {pair["distance"]:.4f}')
        plt.axis('off')
        plt.suptitle(f'Contoh {category}')
        plt.savefig(f'{category}_example.png')
        plt.close()

# Visualisasi t-SNE
embeddings = []
tsne_labels = []
for img_path, label in zip(test_df['path'], test_df['nama']):
    img = cv2.imread(img_path)
    if img is None:
        continue
    face = load_image(img)
    emb = extract_embedding(face, feature_model)
    embeddings.append(emb)
    tsne_labels.append(label)

# Sesuaikan perplexity agar lebih kecil dari n_samples
n_samples = len(embeddings)
perplexity_value = min(5, n_samples - 1)  # Pastikan perplexity < n_samples
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
embeddings_2d = tsne.fit_transform(np.array(embeddings))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=pd.factorize(tsne_labels)[0])
plt.title('t-SNE Embedding Face Similarity')
plt.savefig('tsne_embedding.png')
plt.close()