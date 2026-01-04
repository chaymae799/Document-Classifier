"""
Script pour calculer les métriques détaillées (F1, précision, recall, matrice de confusion)
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import json

# Configuration
MODEL_PATH = "models/cv/model_epoch_01_valacc_67.0635.pth"  # Meilleur checkpoint
VAL_DATA_PATH = "../data_augmented/val"
NUM_CLASSES = 3
BATCH_SIZE = 8

# Transformations (mêmes que l'entraînement)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("🔍 Chargement du dataset de validation...")
val_dataset = ImageFolder(VAL_DATA_PATH, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✅ Dataset chargé : {len(val_dataset)} images")
print(f"📊 Classes : {val_dataset.classes}")
print(f"🔢 Distribution : {dict(zip(val_dataset.classes, np.bincount(val_dataset.targets)))}")

# Chargement du modèle
print("\n🧠 Chargement du modèle...")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

print("✅ Modèle chargé avec succès")

# Inférence sur validation
print("\n🚀 Calcul des prédictions...")
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("✅ Prédictions terminées")

# Calcul des métriques
print("\n" + "="*60)
print("📊 MÉTRIQUES DÉTAILLÉES")
print("="*60)

# Accuracy globale
accuracy = np.mean(all_preds == all_labels)
print(f"\n🎯 Accuracy globale : {accuracy*100:.2f}%")

# Métriques par classe
precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

# Métriques moyennes
precision_avg = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall_avg = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1_avg = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f"\n📈 Métriques moyennes (macro):")
print(f"   Précision : {precision_avg:.4f}")
print(f"   Recall    : {recall_avg:.4f}")
print(f"   F1-score  : {f1_avg:.4f}")

print(f"\n📋 Métriques par classe:")
for i, class_name in enumerate(val_dataset.classes):
    print(f"\n   {class_name}:")
    print(f"      Précision : {precision_per_class[i]:.4f}")
    print(f"      Recall    : {recall_per_class[i]:.4f}")
    print(f"      F1-score  : {f1_per_class[i]:.4f}")
    print(f"      Support   : {np.sum(all_labels == i)} images")

# Matrice de confusion
conf_matrix = confusion_matrix(all_labels, all_preds)
print(f"\n🔢 Matrice de confusion:")
print(f"   Réel \\ Prédit | ", end="")
for class_name in val_dataset.classes:
    print(f"{class_name[:4]:>6}", end=" ")
print()
print("   " + "-"*50)
for i, class_name in enumerate(val_dataset.classes):
    print(f"   {class_name:>15} | ", end="")
    for j in range(len(val_dataset.classes)):
        print(f"{conf_matrix[i][j]:>6}", end=" ")
    print()

# Rapport sklearn
print(f"\n📄 Classification Report (sklearn):")
print(classification_report(all_labels, all_preds, target_names=val_dataset.classes, zero_division=0))

# Sauvegarde des résultats
results = {
    "accuracy": float(accuracy),
    "precision_macro": float(precision_avg),
    "recall_macro": float(recall_avg),
    "f1_macro": float(f1_avg),
    "per_class": {
        val_dataset.classes[i]: {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1_score": float(f1_per_class[i]),
            "support": int(np.sum(all_labels == i))
        }
        for i in range(len(val_dataset.classes))
    },
    "confusion_matrix": conf_matrix.tolist()
}

output_file = "models/cv/detailed_metrics.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n💾 Résultats sauvegardés dans : {output_file}")
print("\n" + "="*60)
print("✅ CALCUL TERMINÉ")
print("="*60)
