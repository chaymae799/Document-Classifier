# 📤 Guide: Pusher le projet sur GitHub

## Option 1: Avec Git LFS (Recommandé pour les modèles)

### 1. Installer Git LFS

```bash
# Windows: Télécharger depuis https://git-lfs.github.com/
# Ou avec chocolatey:
choco install git-lfs

# Initialiser Git LFS
git lfs install
```

### 2. Configurer Git LFS pour les modèles

```bash
cd c:\Users\lenovo\Desktop\document-classifier

# Traquer les fichiers .pth (modèles PyTorch)
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.bin"

# Ajouter .gitattributes
git add .gitattributes
```

### 3. Créer/Mettre à jour .gitignore

```bash
# Créer ou éditer .gitignore pour exclure les gros fichiers inutiles
```

### 4. Commit et Push

```bash
# Ajouter tous les fichiers
git add .

# Commit
git commit -m "feat: projet document-classifier complet avec modèles entraînés"

# Ajouter remote GitHub (si nouveau repo)
git remote add origin https://github.com/TON_USERNAME/document-classifier.git

# Push (Git LFS gère automatiquement les gros fichiers)
git push -u origin main
```

---

## Option 2: Sans les modèles (GitHub standard)

Si tu ne veux PAS pusher les modèles (trop lourds pour GitHub gratuit):

### 1. Ajouter modèles au .gitignore

```bash
# Ajouter ces lignes dans .gitignore:
backend/models/cv/*.pth
backend/models/cv/*.pt
backend/models/nlp/camembert/*.bin
.venv/
__pycache__/
*.pyc
```

### 2. Push normal

```bash
git add .
git commit -m "feat: projet sans modèles (à télécharger séparément)"
git push -u origin main
```

### 3. Documenter où télécharger les modèles

Ajouter dans README.md:

```markdown
## 📥 Téléchargement des modèles

Les modèles pré-entraînés sont trop volumineux pour GitHub.
Télécharger depuis: [lien Google Drive / OneDrive]

Placer dans:

- `backend/models/cv/hybrid_resnet50.pth`
- `backend/models/cv/model_epoch_01_valacc_67.0635.pth`
```

---

## Option 3: GitHub + Google Drive/OneDrive

### 1. Exclure modèles du Git

```bash
# .gitignore
backend/models/cv/*.pth
```

### 2. Uploader modèles sur Drive

- Compresser `backend/models/cv/*.pth` en ZIP
- Upload sur Google Drive / OneDrive
- Partager le lien

### 3. Script de téléchargement automatique

Créer `backend/download_models.py`:

```python
import requests
import os

MODELS = {
    'hybrid_resnet50.pth': 'https://drive.google.com/uc?id=YOUR_FILE_ID',
    'model_epoch_01.pth': 'https://drive.google.com/uc?id=YOUR_FILE_ID2'
}

def download_models():
    for filename, url in MODELS.items():
        path = f'models/cv/{filename}'
        if not os.path.exists(path):
            print(f'Téléchargement {filename}...')
            # Code download...
```

---

## 🎯 Commandes rapides (Option 1 recommandée)

```bash
# 1. Installer Git LFS
git lfs install

# 2. Traquer les modèles
git lfs track "*.pth"
git lfs track "*.pt"

# 3. Commit tout
git add .
git commit -m "feat: projet complet avec modèles CV fine-tuné + NLP mBERT + Gabarits"

# 4. Push
git push -u origin main
```

---

## ⚠️ Limites GitHub

- **GitHub gratuit**: 1 GB stockage LFS / mois
- **Fichier max**: 2 GB avec LFS
- **Repo max**: 5 GB recommandé

Si tes modèles dépassent ces limites, utilise **Option 2 ou 3**.

---

## 📊 Taille des fichiers

Vérifier la taille avant de pusher:

```bash
# Taille des modèles
du -sh backend/models/cv/*.pth

# Taille totale du projet
du -sh .
```

Si > 1 GB, considère **Option 2** (sans modèles) ou **Option 3** (Drive).
