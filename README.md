# 📄 Document Classifier - INDIA-S5

Système de classification automatique de documents administratifs marocains utilisant une approche multimodale (CV + NLP + Gabarits).

## 🎯 Classes supportées

- **Facture Eau** (RADEM, Lydec, Redal)
- **Facture Électricité** (ONE, RADEM, Lydec)
- **Relevé Bancaire** (Attijariwafa, BMCE, CIH, etc.)

## 📊 Performances (sur 252 images de validation)

| Module                     | Accuracy | F1-score | Temps |
| -------------------------- | -------- | -------- | ----- |
| **CV seul (ResNet50)**     | 61.11%   | 0.5845   | 6s    |
| **NLP (mBERT + Keywords)** | ~42-45%  | ~0.41    | 2s    |
| **Gabarits (Règles)**      | ~46-50%  | ~0.47    | 1s    |
| **Fusion multimodale**     | 61.51%   | 0.5872   | 7s    |

### Performances par classe (CV)

| Classe              | Précision | Recall | F1-score | Support |
| ------------------- | --------- | ------ | -------- | ------- |
| Facture Eau         | 0.5476    | 0.3651 | 0.4375   | 63      |
| Facture Électricité | 0.6825    | 0.6825 | 0.6825   | 126     |
| Relevé Bancaire     | 0.7143    | 0.7143 | 0.7143   | 63      |

## 🏗️ Architecture

### Frontend

- **Interface web** : HTML/CSS/JavaScript
- **Upload d'images** : drag-and-drop
- **Affichage résultats** : classe prédite + confiance

### Backend (Flask)

- **app.py** : Serveur REST API
- **final_inference.py** : Pipeline d'inférence complet
- **Modules** :
  - `cv_module_v2.py` : ResNet50 fine-tuné
  - `nlp_module.py` : mBERT + keywords matching
  - `gabarits_module.py` : Analyse structurelle
  - `fusion_module.py` : Fusion multimodale pondérée

### Modèles entraînés

- **hybrid_resnet50.pth** : Modèle CV principal (61.11% acc)
- **model_epoch_01_valacc_67.0635.pth** : Meilleur checkpoint

## 🚀 Installation

### Prérequis

- Python 3.9+
- pip

### Installation des dépendances

```bash
cd backend
pip install -r requirements.txt
```

### Téléchargement des modèles

Les modèles mBERT et ResNet50 sont téléchargés automatiquement au premier lancement.

## ▶️ Lancement

### Démarrage automatique (frontend + backend)

```bash
python start_servers.py
```

### Démarrage manuel

**Backend** (terminal 1):

```bash
cd backend
python app.py
```

**Frontend** (terminal 2):

```bash
cd frontend
python -m http.server 8000
```

Puis ouvrir [http://localhost:8000/index_new.html](http://localhost:8000/index_new.html)

## 📁 Structure du projet

```
document-classifier/
├── frontend/              # Interface web
│   ├── index_new.html    # Interface principale
│   ├── app.js            # Logique frontend
│   └── style.css         # Styles
├── backend/
│   ├── app.py            # Serveur Flask (API REST)
│   ├── final_inference.py # Pipeline d'inférence
│   ├── calculate_metrics.py # Calcul des métriques
│   ├── evaluate_fusion.py   # Évaluation fusion
│   ├── evaluate_all_modules.py # Évaluation complète
│   ├── config.py         # Configuration
│   ├── requirements.txt  # Dépendances Python
│   ├── modules/          # Modules de classification
│   │   ├── cv_module_v2.py      # Module vision (ResNet50)
│   │   ├── nlp_module.py        # Module NLP (mBERT)
│   │   ├── gabarits_module.py   # Module gabarits
│   │   ├── fusion_module.py     # Fusion multimodale
│   │   └── ocr_module.py        # Extraction texte (Tesseract)
│   ├── models/cv/        # Modèles entraînés
│   │   ├── hybrid_resnet50.pth  # Modèle principal
│   │   └── *.json        # Résultats d'évaluation
│   └── utils/            # Utilitaires
│       ├── image_preprocessor.py
│       └── model_manager.py
├── data/                 # Données originales (12 images)
├── data_augmented/       # Données augmentées (252 images)
│   ├── train/            # 80% (202 images)
│   └── val/              # 20% (50 images → augmenté à 252)
├── Dataset/              # Dataset brut
├── uploads/              # Uploads utilisateurs
├── scripts/              # Scripts utilitaires
└── README_LATEX_REPORT.tex  # Rapport LaTeX complet
```

## 🔧 Scripts disponibles

### Évaluation des performances

```bash
cd backend

# Calculer les métriques détaillées (CV seul)
python calculate_metrics.py

# Évaluer la fusion multimodale
python evaluate_fusion.py

# Évaluer tous les modules (CV, NLP, Gabarits, Fusion)
python evaluate_all_modules.py
```

### Résultats sauvegardés

Les évaluations génèrent des fichiers JSON dans `backend/models/cv/`:

- `detailed_metrics.json` : Métriques par classe (CV)
- `fusion_evaluation.json` : Comparaison CV vs Fusion
- `complete_evaluation.json` : Tous les modules

## 🌐 API Endpoints

### `POST /predict`

Classifier un document uploadé

**Request:**

```json
{
  "file": "<image file>"
}
```

**Response:**

```json
{
  "class": "facture_electricite",
  "confidence": 0.85,
  "module_scores": {
    "cv": 0.82,
    "nlp": 0.65,
    "gabarits": 0.7
  }
}
```

## 📈 Améliorations possibles

### Court terme (+2-4%)

- Fine-tuner mBERT sur 1000+ documents annotés
- Enrichir les templates de gabarits
- Optimiser la pondération de fusion (grid search)

### Moyen terme (+5-10%)

- Data augmentation avancée (rotation, perspective, bruit)
- Utiliser CamemBERT ou LayoutLM pour le NLP
- Ensemble de plusieurs modèles CV (ResNet50 + EfficientNet)

### Long terme (+10-20%)

- Approche end-to-end multimodale (ViT + BERT)
- Active learning pour annoter les cas difficiles
- Transfer learning depuis modèles pré-entraînés sur documents administratifs

## 📝 Notes techniques

### Limites actuelles

- **Facture Eau** : Performances faibles (36.51% F1) dues au déséquilibre de classe et à la confusion avec les factures d'électricité
- **NLP** : Pas fine-tuné, utilise uniquement similarité d'embeddings + keywords
- **Gabarits** : Règles heuristiques génériques, pas de ML
- **Fusion** : Gain modeste (+0.40%) car modules complémentaires non optimisés

### Dataset

- **Original** : 12 images (4 par classe)
- **Augmenté** : 252 images (augmentation de données x21)
- **Split** : 80% train (202) / 20% val (50 → 252 avec augmentation)
