# Document Classifier - Multimodal AI System

Système de classification multimodal de documents administratifs marocains utilisant PyTorch, Transformers et OpenCV.

## 📋 Catégories Supportées

- 🏦 Relevés bancaires
- ⚡ Factures d'électricité
- 💧 Factures d'eau

## 🏗️ Architecture

- **Vision (CV)**: ResNet50 fine-tuné + 36 features structurelles
- **Texte (NLP)**: Analyse de keywords + Embeddings mBERT
- **Layout (Gabarits)**: Détection tableaux, logos, signatures
- **Fusion**: Combinaison multimodale intelligente (CV 40% + NLP 35% + Gabarits 25%)

## 🚀 Installation

### Prérequis

- Python 3.9+
- Tesseract OCR
- Poppler (pour PDF)

### Dépendances

```bash
pip install -r backend/requirements.txt
```

### Tesseract (Windows)

1. Télécharger: https://github.com/UB-Mannheim/tesseract/wiki
2. Installer dans `C:\Program Files\Tesseract-OCR\`

### Poppler (Windows)

1. Télécharger: http://blog.alivate.com.au/poppler-windows/
2. Extraire dans `C:\poppler\`

## 📁 Structure du Projet

```
document-classifier/
├── backend/
│   ├── app.py              # API Flask
│   ├── config.py           # Configuration
│   ├── train_hybrid.py     # Entraînement modèle
│   ├── modules/            # Modules de classification
│   │   ├── ocr_module.py
│   │   ├── nlp_module.py
│   │   ├── cv_module_v2.py
│   │   ├── gabarits_module.py
│   │   └── fusion_module.py
│   └── utils/
├── frontend/
│   ├── index_new.html      # Interface web
│   └── style.css
├── data/                   # Datasets (gitignored)
└── models/                 # Modèles entraînés (gitignored)
```

## 🎯 Utilisation

### 1. Démarrer le Backend

```bash
cd backend
python app.py
```

Backend accessible sur: http://localhost:5000

### 2. Démarrer le Frontend

```bash
cd frontend
python -m http.server 8000
```

Interface accessible sur: http://localhost:8000/index_new.html

### 3. Utiliser l'API

```python
import requests

with open('document.pdf', 'rb') as f:
    response = requests.post('http://localhost:5000/api/classify',
                           files={'file': f})
    result = response.json()
    print(f"Classe: {result['classification']}")
    print(f"Confiance: {result['confidence_globale']:.2%}")
```

## 📊 Entraînement du Modèle

### Avec Augmentation de Données

```bash
python backend/train_hybrid.py \
    --data-dir data_augmented \
    --epochs 60 \
    --batch-size 16 \
    --lr 0.001
```

### Sur Google Colab

Utiliser le notebook `COLAB_TRAINING.ipynb` pour entraînement GPU gratuit.

## 🔧 Configuration

Modifier `backend/config.py`:

- Chemins Tesseract/Poppler
- Seuils de confiance
- Poids de fusion multimodale
- Catégories de documents

## 📈 Performance

Avec dataset augmenté (~900 images):

- **Accuracy attendue**: 60-75%
- **Temps de traitement**: 3-5s (image), 8-12s (PDF)
- **Seuil d'acceptation**: 50%

## 🔍 Endpoints API

- `GET /api/health` - État du système
- `POST /api/classify` - Classification d'un document
- `POST /api/batch` - Classification par batch
- `GET /api/categories` - Liste des catégories

## 🛠️ Technologies

- **Backend**: Flask, PyTorch, Transformers, OpenCV, Tesseract
- **Frontend**: HTML/CSS/JS, Chart.js
- **Models**: ResNet50, mBERT (bert-base-multilingual-cased)

## 📝 Licence

Projet académique - INDIA-S5

## 👥 Contribution

Projet développé dans le cadre du module INDIA à l'ENSIAS.
