import os
import torch
import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


from model_loader import YoloDanceSystemWeighted

# === PARAMÈTRES ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/classifier_weighted.pth"  # <- Ton modèle entraîné
DATASET_DIR = "data/test"  # <- Dossier avec sous-dossiers de labels
IMAGE_SIZE = (640, 640)

# === INITIALISATION DU SYSTÈME ===
system = YoloDanceSystemWeighted(custom_classifier_path=MODEL_PATH)
system.classifier.to(DEVICE)

# === LECTURE DU DATASET ===
def load_dataset(dataset_dir):
    image_paths = []
    labels = []
    classes = sorted(os.listdir(dataset_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for label in classes:
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(label_dir, fname))
                labels.append(class_to_idx[label])

    return image_paths, labels, class_to_idx

# === EXTRACTION ET PRÉDICTION ===
def predict_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, IMAGE_SIZE)

    # 1. Pose YOLO
    results = system.pose_detector.predict(image_resized, verbose=False)

    # 2. Features
    pose_feat = system.extract_pose_features(results)
    hand_feat = system.extract_hand_features(image_resized)
    context_feat = system.extract_context_features(results, image_resized.shape)

    features = np.concatenate([pose_feat, hand_feat, context_feat], axis=0)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = system.classifier(features_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction

# === ÉVALUATION ===
def evaluate():
    image_paths, true_labels, class_to_idx = load_dataset(DATASET_DIR)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    predictions = []

    print(f"🔎 Évaluation sur {len(image_paths)} images...")
    for path in tqdm(image_paths):
        pred = predict_image(path)
        predictions.append(pred)

    # === METRICS ===
    print("\n📊 Rapport de classification :")
    print(classification_report(true_labels, predictions, target_names=idx_to_class.values()))

    print("📉 Matrice de confusion :")
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=idx_to_class.values(), yticklabels=idx_to_class.values(), cmap="Blues")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
