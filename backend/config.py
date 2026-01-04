"""
Configuration centralisée du système
Projet: Classification de Documents - INDIA-S5
"""

import os
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

# Dossiers
UPLOAD_FOLDER = PROJECT_ROOT / 'uploads'
RESULTS_FOLDER = PROJECT_ROOT / 'results'
MODELS_FOLDER = PROJECT_ROOT / 'models'
DATA_FOLDER = PROJECT_ROOT / 'data'
LOGS_FOLDER = PROJECT_ROOT / 'logs'

# Créer les dossiers s'ils n'existent pas
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, MODELS_FOLDER, DATA_FOLDER, LOGS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Configuration Flask
FLASK_CONFIG = {
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'MAX_CONTENT_LENGTH': 10 * 1024 * 1024,  # 10 MB
}

# Extensions de fichiers autorisées
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Configuration Tesseract (à adapter selon votre OS)
TESSERACT_CONFIG = {
    # Windows:
    'cmd': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    # Mac/Linux: (commenter la ligne Windows et décommenter celle-ci)
    # 'cmd': '/usr/bin/tesseract',
    'lang': 'fra+ara',  # French + Arabic
    'config': r'--oem 3 --psm 6'
}

# Configuration Poppler (pour conversion PDF)
POPPLER_CONFIG = {
    # Windows:
    'path': r'C:\poppler\bin',
    # Mac/Linux: (laisser None)
    # 'path': None,
    'dpi': 300
}

# Configuration des modèles CV
CV_CONFIG = {
    'device': 'cpu',
    'models': {
        'resnet50': {
            'enabled': True,
            'weight': 1.0,  # ← 100% ResNet (change 0.6 en 1.0)
            'input_size': (224, 224)
        },
        'efficientnet': {
            'enabled': False,  # ← CHANGE True en False
            'weight': 0.0,  # ← CHANGE 0.4 en 0.0
            'input_size': (224, 224)
        }
    },
}

# Configuration NLP
NLP_CONFIG = {
    'device': 'cpu',
    'camembert_model': 'camembert-base',
    'max_length': 512,
    'enable_camembert': True,  # Désactiver si problème de mémoire
}

# Configuration OCR
OCR_CONFIG = {
    'preprocessing_method': 'adaptive',  # 'adaptive', 'otsu', 'gaussian'
    'min_confidence': 60,
    'enable_deskew': True,  # Correction d'inclinaison
    'enable_denoising': True,
}

# Configuration Gabarits
GABARITS_CONFIG = {
    'enable_face_detection': True,
    'enable_table_detection': True,
    'enable_color_analysis': True,
    'min_table_confidence': 0.3,
}

# Configuration Fusion
FUSION_CONFIG = {
    'default_weights': {
        'cv': 0.40,
        'nlp': 0.35,
        'gabarits': 0.25
    },
    'confidence_threshold': 0.6,  # Seuil de rejet
    'enable_business_rules': True,
    'enable_adaptive_weights': True,
}

# Catégories de documents (3 classes actives)
CATEGORIES = {
    'releve_bancaire': {
        'name': 'Relevé bancaire',
        'description': 'Relevés de compte bancaires',
        'icon': '🏦',
        'color': '#10B981'
    },
    'facture_electricite': {
        'name': 'Facture d\'électricité',
        'description': 'Factures ONE, RADEM, Lydec, Redal',
        'icon': '⚡',
        'color': '#F59E0B'
    },
    'facture_eau': {
        'name': 'Facture d\'eau',
        'description': 'Factures RADEM, Lydec, Redal',
        'icon': '💧',
        'color': '#06B6D4'
    }
}

# Nombre de classes
NUM_CLASSES = 3

# Configuration du logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'default',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(LOGS_FOLDER / 'app.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

# Benchmarking
BENCHMARK_CONFIG = {
    'enable': False,
    'iterations': 10,
    'save_results': True,
    'output_file': str(RESULTS_FOLDER / 'benchmark_results.json')
}

# Mode de développement
DEV_MODE = True

# Version
VERSION = '1.0.0'
PROJECT_NAME = 'Document Classifier INDIA-S5'
AUTHOR = 'Équipe INDIA-S5'