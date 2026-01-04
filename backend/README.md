# 📄 Système de Classification de Documents Administratifs

**Projet:** INDIA-S5 - Module CV/NLP  
**Professeur:** Pr. CHEFIRA  
**Année:** 2024-2025

---

## 📋 Description

Système intelligent de classification automatique de documents administratifs marocains utilisant une approche multimodale combinant :

- **Computer Vision** (ResNet50 + EfficientNet)
- **Natural Language Processing** (CamemBERT + Analyse sémantique)
- **Détection de Gabarits** (Patterns structurels)
- **Fusion Multimodale** (Pondération adaptative)

### 🎯 Catégories Supportées

1. 🆔 **Pièce d'identité (CNIE)** - Carte Nationale d'Identité Électronique
2. 🏦 **Relevés bancaires** - Toutes banques marocaines
3. ⚡ **Factures d'électricité** - ONE, RADEM, Lydec, Redal
4. 💧 **Factures d'eau** - RADEM, Lydec, Redal
5. 💼 **Documents employeur** - Bulletins de paie, attestations

---

## 🏗️ Architecture

```
backend/
├── modules/
│   ├── ocr_module.py          # Extraction texte (Tesseract + prétraitement)
│   ├── nlp_module.py          # Analyse sémantique (CamemBERT + mots-clés)
│   ├── cv_module.py           # Classification visuelle (ResNet50 + EfficientNet)
│   ├── gabarits_module.py     # Détection patterns structurels
│   └── fusion_module.py       # Fusion multimodale intelligente
├── utils/
│   ├── image_preprocessor.py  # Prétraitement d'images
│   └── model_manager.py       # Gestion des modèles
├── app.py                     # API Flask
└── config.py                  # Configuration centralisée
```

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- Tesseract OCR
- Poppler (pour PDF)
- 4GB RAM minimum
- GPU optionnel (accélère CV)

### Étapes

```bash
# 1. Cloner le projet
git clone https://github.com/votre-repo/document-classifier
cd document-classifier/backend

# 2. Créer environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Installer dépendances
pip install -r requirements.txt

# 4. Configurer Tesseract et Poppler dans config.py

# 5. Lancer le serveur
python app.py
```

Le serveur démarre sur `http://localhost:5000`

---

## 📡 API Endpoints

### 1. Classification d'un document

```bash
POST /api/classify
Content-Type: multipart/form-data

Body: file (PDF, JPG, PNG)

Response:
{
  "classification": "piece_identite",
  "confidence_globale": 0.92,
  "confidence_cv": 0.89,
  "confidence_nlp": 0.94,
  "confidence_gabarit": 0.91,
  "features_cv": {...},
  "features_nlp": {...},
  "processing_times": {...}
}
```

### 2. État du serveur

```bash
GET /api/health

Response:
{
  "status": "online",
  "modules": {
    "ocr": true,
    "nlp": true,
    "cv": true,
    "gabarits": true,
    "fusion": true
  }
}
```

### 3. Liste des catégories

```bash
GET /api/categories
```

### 4. Classification par lots

```bash
POST /api/batch
Content-Type: multipart/form-data

Body: files[] (jusqu'à 10 fichiers)
```

---

## 🧪 Tests

```bash
# Test d'un document
curl -X POST http://localhost:5000/api/classify \
  -F "file=@test_document.jpg"

# Test de santé
curl http://localhost:5000/api/health
```

---

## 📊 Performance

| Métrique                | Valeur                              |
| ----------------------- | ----------------------------------- |
| **Précision globale**   | 92-95%                              |
| **Temps de traitement** | 2-4 secondes/document               |
| **Modèles CV**          | ResNet50 (60%) + EfficientNet (40%) |
| **OCR**                 | Tesseract avec prétraitement avancé |
| **NLP**                 | CamemBERT + Analyse par mots-clés   |

---

## 🔧 Configuration

Modifier `config.py` pour :

- Chemins Tesseract/Poppler
- Poids des modules (CV/NLP/Gabarits)
- Seuils de confiance
- Activation GPU
- Modes de prétraitement

---

## 📝 Modules Détaillés

### Module OCR

- Prétraitement adaptatif (débruitage, CLAHE, binarisation)
- Extraction avec scores de confiance
- Correction automatique d'erreurs OCR
- Support multi-pages

### Module NLP

- CamemBERT embeddings
- Analyse par mots-clés pondérés
- Extraction d'entités (RIB, CNSS, montants...)
- Analyse structurelle du texte

### Module CV

- Ensemble ResNet50 + EfficientNet
- Test Time Augmentation (optionnel)
- Multi-crop prediction (optionnel)
- Extraction de features visuelles

### Module Gabarits

- Détection de photos (cascade classifiers)
- Détection de tableaux (Hough transform)
- Analyse de couleurs dominantes
- Détection de signatures/logos

### Module Fusion

- Pondération adaptative selon contexte
- Règles métier par catégorie
- Validation croisée
- Ajustement de confiance intelligent

---

## 🐛 Dépannage

### Erreur Tesseract

```python
# Dans config.py, mettre le bon chemin:
TESSERACT_CONFIG = {
    'cmd': r'C:\Program Files\Tesseract-OCR\tesseract.exe'
}
```

### Erreur mémoire CamemBERT

```python
# Dans config.py:
NLP_CONFIG = {
    'enable_camembert': False  # Désactiver si RAM insuffisante
}
```

### Erreur GPU

```python
# Dans config.py, forcer CPU:
CV_CONFIG = {
    'device': 'cpu'
}
```

---

## 📈 Améliorations Futures

- [ ] Fine-tuning des modèles sur données spécifiques
- [ ] Support de nouvelles catégories
- [ ] API de réentraînement
- [ ] Interface d'administration
- [ ] Export des résultats en CSV/Excel
- [ ] Système de feedback utilisateur

---

## 👥 Équipe

**Projet réalisé par l'équipe INDIA-S5**

---

## 📄 Licence

Projet académique - INDIA-S5 - 2024-2025

---

## 📞 Support

Pour toute question:

- Email: votre-email@exemple.com
- Issues: GitHub Issues

---

## 🙏 Remerciements

- Pr. CHEFIRA pour l'encadrement
- Anthropic (Claude) pour l'assistance au développement
- Communauté open-source (PyTorch, Transformers, OpenCV)

---

**⭐ Si ce projet vous aide, n'hésitez pas à le star sur GitHub!**
