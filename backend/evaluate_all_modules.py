"""
Évaluation complète de TOUS les modules: CV, NLP, Gabarits, et Fusion
Mesure les performances réelles de chaque modalité
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import json
import os
import sys
from pathlib import Path
import cv2
from PIL import Image

# Imports modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.nlp_module import AdvancedNLPModule
from modules.gabarits_module import GabaritsModule

# Configuration
MODEL_PATH = "models/cv/model_epoch_01_valacc_67.0635.pth"
VAL_DATA_PATH = "../data_augmented/val"
OCR_TEXT_PATH = "../data/ocr_text"
NUM_CLASSES = 3
BATCH_SIZE = 8

# Mapping classes
CLASS_NAMES = ['facture_eau', 'facture_electricite', 'releve_bancaire']
CLASS_MAPPING = {
    'facture_eau': 0,
    'facture_electricite': 1,
    'releve_bancaire': 2
}

print("="*80)
print("🚀 ÉVALUATION COMPLÈTE DES MODULES")
print("="*80)

# =====================================================================
# 1. CHARGEMENT DATASET
# =====================================================================
print("\n📁 Chargement du dataset...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = ImageFolder(VAL_DATA_PATH, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"✅ {len(val_dataset)} images chargées")

# Récupération des chemins images pour OCR/Gabarits
image_paths = [path for path, _ in val_dataset.samples]
true_labels = [label for _, label in val_dataset.samples]

# =====================================================================
# 2. MODULE CV (Vision)
# =====================================================================
print("\n" + "="*80)
print("🖼️  MODULE CV - VISION (ResNet50)")
print("="*80)

print("🧠 Chargement du modèle...")
cv_model = models.resnet50(pretrained=False)
cv_model.fc = nn.Linear(cv_model.fc.in_features, NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
cv_model.load_state_dict(checkpoint)
cv_model.eval()
print("✅ Modèle chargé")

print("🔍 Évaluation CV...")
cv_predictions = []
cv_probs_list = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = cv_model(images)
        probs = torch.softmax(outputs, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        
        cv_predictions.extend(preds)
        cv_probs_list.extend(probs)

cv_predictions = np.array(cv_predictions)
cv_probs_array = np.array(cv_probs_list)
true_labels_array = np.array(true_labels)

cv_accuracy = accuracy_score(true_labels_array, cv_predictions)
cv_precision = precision_score(true_labels_array, cv_predictions, average='macro', zero_division=0)
cv_recall = recall_score(true_labels_array, cv_predictions, average='macro', zero_division=0)
cv_f1 = f1_score(true_labels_array, cv_predictions, average='macro', zero_division=0)

print(f"\n📊 RÉSULTATS CV:")
print(f"   Accuracy  : {cv_accuracy*100:.2f}%")
print(f"   Précision : {cv_precision:.4f}")
print(f"   Recall    : {cv_recall:.4f}")
print(f"   F1-score  : {cv_f1:.4f}")

# =====================================================================
# 3. MODULE NLP (Texte OCR)
# =====================================================================
print("\n" + "="*80)
print("📝 MODULE NLP - ANALYSE TEXTUELLE")
print("="*80)

print("🧠 Initialisation NLP...")
nlp_module = AdvancedNLPModule(device='cpu')
print("✅ Module NLP prêt")

print("🔍 Évaluation NLP sur textes OCR...")
nlp_predictions = []
nlp_scores_list = []

for img_path in image_paths:
    # Trouver le fichier OCR correspondant
    img_name = Path(img_path).stem
    class_name = Path(img_path).parent.name
    
    ocr_file = Path(OCR_TEXT_PATH) / class_name / f"{img_name}.txt"
    
    if ocr_file.exists():
        with open(ocr_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = ""  # Texte vide si OCR non disponible
    
    # Analyse NLP
    result = nlp_module.classify_text(text, {})
    
    # Conversion scores vers prédiction
    # result['all_scores'] = {'facture_eau': 0.2, 'facture_electricite': 0.8, ...}
    class_scores = np.array([
        result['all_scores'].get('facture_eau', 0.0),
        result['all_scores'].get('facture_electricite', 0.0),
        result['all_scores'].get('releve_bancaire', 0.0)
    ])
    
    # Normalisation
    if class_scores.sum() > 0:
        class_scores = class_scores / class_scores.sum()
    else:
        class_scores = np.ones(3) / 3  # Équiprobable si aucun score
    
    pred = np.argmax(class_scores)
    nlp_predictions.append(pred)
    nlp_scores_list.append(class_scores)

nlp_predictions = np.array(nlp_predictions)
nlp_probs_array = np.array(nlp_scores_list)

nlp_accuracy = accuracy_score(true_labels_array, nlp_predictions)
nlp_precision = precision_score(true_labels_array, nlp_predictions, average='macro', zero_division=0)
nlp_recall = recall_score(true_labels_array, nlp_predictions, average='macro', zero_division=0)
nlp_f1 = f1_score(true_labels_array, nlp_predictions, average='macro', zero_division=0)

print(f"\n📊 RÉSULTATS NLP:")
print(f"   Accuracy  : {nlp_accuracy*100:.2f}%")
print(f"   Précision : {nlp_precision:.4f}")
print(f"   Recall    : {nlp_recall:.4f}")
print(f"   F1-score  : {nlp_f1:.4f}")

# =====================================================================
# 4. MODULE GABARITS (Structure)
# =====================================================================
print("\n" + "="*80)
print("🏗️  MODULE GABARITS - STRUCTURE VISUELLE")
print("="*80)

print("🧠 Initialisation Gabarits...")
gabarits_module = GabaritsModule()
print("✅ Module Gabarits prêt")

print("🔍 Évaluation Gabarits sur images...")
gabarits_predictions = []
gabarits_scores_list = []

for img_path in image_paths:
    # Analyse gabarit directement avec le chemin
    result = gabarits_module.analyze(img_path)
    scores = result.get('scores', {})
    
    # Conversion vers array
    class_scores = np.array([
        scores.get('facture_eau', 0.0),
        scores.get('facture_electricite', 0.0),
        scores.get('releve_bancaire', 0.0)
    ])
    
    # Normalisation
    if class_scores.sum() > 0:
        class_scores = class_scores / class_scores.sum()
    else:
        class_scores = np.ones(3) / 3
    
    pred = np.argmax(class_scores)
    gabarits_predictions.append(pred)
    gabarits_scores_list.append(class_scores)

gabarits_predictions = np.array(gabarits_predictions)
gabarits_probs_array = np.array(gabarits_scores_list)

gabarits_accuracy = accuracy_score(true_labels_array, gabarits_predictions)
gabarits_precision = precision_score(true_labels_array, gabarits_predictions, average='macro', zero_division=0)
gabarits_recall = recall_score(true_labels_array, gabarits_predictions, average='macro', zero_division=0)
gabarits_f1 = f1_score(true_labels_array, gabarits_predictions, average='macro', zero_division=0)

print(f"\n📊 RÉSULTATS GABARITS:")
print(f"   Accuracy  : {gabarits_accuracy*100:.2f}%")
print(f"   Précision : {gabarits_precision:.4f}")
print(f"   Recall    : {gabarits_recall:.4f}")
print(f"   F1-score  : {gabarits_f1:.4f}")

# =====================================================================
# 5. FUSION MULTIMODALE (Pondération optimale)
# =====================================================================
print("\n" + "="*80)
print("🔀 FUSION MULTIMODALE")
print("="*80)

# Test plusieurs pondérations
weights_configs = [
    (0.70, 0.20, 0.10),  # CV dominant
    (0.60, 0.25, 0.15),  # Équilibré
    (0.50, 0.30, 0.20),  # Plus de NLP/Gabarits
]

best_fusion_acc = 0
best_fusion_config = None
best_fusion_preds = None

print("🔍 Test de différentes pondérations...")
for w_cv, w_nlp, w_gab in weights_configs:
    # Fusion pondérée
    fusion_probs = (w_cv * cv_probs_array + 
                    w_nlp * nlp_probs_array + 
                    w_gab * gabarits_probs_array)
    
    fusion_preds = np.argmax(fusion_probs, axis=1)
    fusion_acc = accuracy_score(true_labels_array, fusion_preds)
    
    print(f"   CV={w_cv:.2f}, NLP={w_nlp:.2f}, GAB={w_gab:.2f} → Acc={fusion_acc*100:.2f}%")
    
    if fusion_acc > best_fusion_acc:
        best_fusion_acc = fusion_acc
        best_fusion_config = (w_cv, w_nlp, w_gab)
        best_fusion_preds = fusion_preds

print(f"\n✅ MEILLEURE CONFIG: CV={best_fusion_config[0]:.2f}, NLP={best_fusion_config[1]:.2f}, GAB={best_fusion_config[2]:.2f}")

fusion_precision = precision_score(true_labels_array, best_fusion_preds, average='macro', zero_division=0)
fusion_recall = recall_score(true_labels_array, best_fusion_preds, average='macro', zero_division=0)
fusion_f1 = f1_score(true_labels_array, best_fusion_preds, average='macro', zero_division=0)

print(f"\n📊 RÉSULTATS FUSION OPTIMALE:")
print(f"   Accuracy  : {best_fusion_acc*100:.2f}%")
print(f"   Précision : {fusion_precision:.4f}")
print(f"   Recall    : {fusion_recall:.4f}")
print(f"   F1-score  : {fusion_f1:.4f}")

# =====================================================================
# 6. RÉSUMÉ COMPARATIF
# =====================================================================
print("\n" + "="*80)
print("📊 TABLEAU COMPARATIF FINAL")
print("="*80)

results = {
    'cv_seul': {
        'accuracy': float(cv_accuracy),
        'precision': float(cv_precision),
        'recall': float(cv_recall),
        'f1_score': float(cv_f1)
    },
    'nlp_seul': {
        'accuracy': float(nlp_accuracy),
        'precision': float(nlp_precision),
        'recall': float(nlp_recall),
        'f1_score': float(nlp_f1)
    },
    'gabarits_seul': {
        'accuracy': float(gabarits_accuracy),
        'precision': float(gabarits_precision),
        'recall': float(gabarits_recall),
        'f1_score': float(gabarits_f1)
    },
    'fusion_multimodale': {
        'accuracy': float(best_fusion_acc),
        'precision': float(fusion_precision),
        'recall': float(fusion_recall),
        'f1_score': float(fusion_f1),
        'weights': {
            'cv': best_fusion_config[0],
            'nlp': best_fusion_config[1],
            'gabarits': best_fusion_config[2]
        }
    },
    'gains': {
        'cv_vs_fusion': float(best_fusion_acc - cv_accuracy),
        'best_unimodal_vs_fusion': float(best_fusion_acc - max(cv_accuracy, nlp_accuracy, gabarits_accuracy))
    }
}

print(f"""
╔════════════════════════════════════════════════════════════════╗
║                    RÉSULTATS COMPLETS                          ║
╠════════════════════════════════════════════════════════════════╣
║ CV seul (ResNet50)      : {cv_accuracy*100:5.2f}% │ F1={cv_f1:.4f}       ║
║ NLP seul (mBERT)        : {nlp_accuracy*100:5.2f}% │ F1={nlp_f1:.4f}       ║
║ Gabarits seul           : {gabarits_accuracy*100:5.2f}% │ F1={gabarits_f1:.4f}       ║
║ ─────────────────────────────────────────────────────────────  ║
║ FUSION (w={best_fusion_config[0]:.1f}/{best_fusion_config[1]:.1f}/{best_fusion_config[2]:.1f}): {best_fusion_acc*100:5.2f}% │ F1={fusion_f1:.4f}       ║
║ ─────────────────────────────────────────────────────────────  ║
║ Gain CV→Fusion         : +{(best_fusion_acc-cv_accuracy)*100:4.2f}%                       ║
╚════════════════════════════════════════════════════════════════╝
""")

# Sauvegarde
output_file = "models/cv/complete_evaluation.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n💾 Résultats sauvegardés: {output_file}")
print("="*80)
