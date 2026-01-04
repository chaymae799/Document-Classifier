"""
SYSTÈME DE CLASSIFICATION DE DOCUMENTS ADMINISTRATIFS
Projet: INDIA-S5 - Pr. CHEFIRA

Architecture Modulaire:
- Module OCR: Extraction de texte optimisée
- Module NLP: Analyse sémantique et CamemBERT
- Module CV: ResNet50 + EfficientNet en ensemble
- Module Gabarits: Détection de patterns structurels
- Module Fusion: Combinaison intelligente multimodale

API Flask pour interface web
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time
import logging
from typing import Dict

# PDF handling: Use PyMuPDF (fitz) instead of pdf2image
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    # Fallback to pdf2image if available
    try:
        from pdf2image import convert_from_path
        HAS_PDF2IMAGE = True
    except ImportError:
        HAS_PDF2IMAGE = False

# Import des modules custom
from modules.ocr_module import AdvancedOCRModule
from modules.nlp_module import AdvancedNLPModule
from modules.cv_module_v2 import HybridCVModule  # Version fine-tuning
from modules.gabarits_module import GabaritsModule
from modules.fusion_module import MultimodalFusionModule

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation Flask
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '../uploads'
RESULTS_FOLDER = '../results'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ============================================
# INITIALISATION DES MODULES
# ============================================
logger.info("="*60)
logger.info("INITIALISATION DU SYSTÈME DE CLASSIFICATION")
logger.info("="*60)

try:
    # Device (GPU si disponible, sinon CPU)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Initialisation des modules
    logger.info("Chargement des modules...")
    
    ocr_module = AdvancedOCRModule()
    nlp_module = AdvancedNLPModule(device=device)
    cv_module = HybridCVModule(device=device)
    gabarits_module = GabaritsModule()
    fusion_module = MultimodalFusionModule()
    
    logger.info("="*60)
    logger.info("✓ TOUS LES MODULES CHARGÉS AVEC SUCCÈS")
    logger.info("="*60)
    
    MODULES_LOADED = True
    
except Exception as e:
    logger.error(f"ERREUR lors de l'initialisation: {e}")
    MODULES_LOADED = False

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def allowed_file(filename: str) -> bool:
    """Vérifie si le fichier est autorisé"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_pdf_to_images(pdf_path: str) -> list:
    """Convertit un PDF en images using PyMuPDF"""
    try:
        # Try PyMuPDF first (preferred, no external dependencies)
        if HAS_PYMUPDF:
            doc = fitz.open(pdf_path)
            image_paths = []
            for i, page in enumerate(doc):
                # Render page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)  # 2x zoom for better quality
                image_path = pdf_path.replace('.pdf', f'_page_{i}.jpg')
                pix.save(image_path)
                image_paths.append(image_path)
                logger.info(f"  ✓ Page {i+1} convertie")
            doc.close()
            return image_paths
        
        # Fallback to pdf2image if available
        elif HAS_PDF2IMAGE:
            images = convert_from_path(pdf_path, dpi=300)
            image_paths = []
            for i, image in enumerate(images):
                image_path = pdf_path.replace('.pdf', f'_page_{i}.jpg')
                image.save(image_path, 'JPEG', quality=95)
                image_paths.append(image_path)
                logger.info(f"  ✓ Page {i+1} convertie")
            return image_paths
        else:
            logger.error("Aucune library PDF disponible (PyMuPDF ni pdf2image)")
            return []
            
    except Exception as e:
        logger.error(f"Erreur conversion PDF: {e}")
        return []

def classify_document(image_path: str) -> Dict:
    """
    Pipeline complet de classification d'un document
    
    Args:
        image_path: Chemin vers l'image du document
    
    Returns:
        Résultat complet de classification
    """
    start_time = time.time()
    logger.info("="*60)
    logger.info(f"CLASSIFICATION: {os.path.basename(image_path)}")
    logger.info("="*60)
    
    try:
        # ÉTAPE 1: OCR - Extraction de texte
        logger.info("ÉTAPE 1/5: Extraction OCR...")
        ocr_start = time.time()
        ocr_result = ocr_module.extract_with_confidence(image_path)
        text = ocr_result['text']
        ocr_time = time.time() - ocr_start
        logger.info(f"  ✓ OCR terminé en {ocr_time:.2f}s")
        logger.info(f"  → {len(text)} caractères extraits")
        logger.info(f"  → Confiance OCR: {ocr_result['confidence']:.2f}")
        
        # ÉTAPE 2: Analyse Gabarits
        logger.info("ÉTAPE 2/5: Analyse des gabarits...")
        gabarit_start = time.time()
        gabarit_result = gabarits_module.analyze(image_path)
        gabarit_time = time.time() - gabarit_start
        logger.info(f"  ✓ Gabarits analysés en {gabarit_time:.2f}s")
        
        # ÉTAPE 3: Classification Computer Vision
        logger.info("ÉTAPE 3/5: Classification CV (ResNet50 + EfficientNet)...")
        cv_start = time.time()
        cv_result = cv_module.classify(image_path)
        cv_time = time.time() - cv_start
        logger.info(f"  ✓ CV terminé en {cv_time:.2f}s")
        logger.info(f"  → Prédiction: {cv_result['class']} ({cv_result['confidence']:.2f})")
        
        # ÉTAPE 4: Classification NLP
        logger.info("ÉTAPE 4/5: Classification NLP (CamemBERT + Mots-clés)...")
        nlp_start = time.time()
        entities = nlp_module.extract_entities(text)
        nlp_result = nlp_module.classify_text(text, entities)
        nlp_time = time.time() - nlp_start
        logger.info(f"  ✓ NLP terminé en {nlp_time:.2f}s")
        logger.info(f"  → Prédiction: {nlp_result['class']} ({nlp_result['confidence']:.2f})")
        logger.info(f"  → Entités: {list(entities.keys())}")
        
        # ÉTAPE 5: Fusion Multimodale
        logger.info("ÉTAPE 5/5: Fusion multimodale...")
        fusion_start = time.time()
        final_result = fusion_module.fuse(
            cv_result,
            nlp_result,
            gabarit_result,
            image_path
        )
        fusion_time = time.time() - fusion_start
        logger.info(f"  ✓ Fusion terminée en {fusion_time:.2f}s")
        
        # Résultat final
        total_time = time.time() - start_time
        
        logger.info("="*60)
        logger.info(f"RÉSULTAT FINAL: {final_result['classification']}")
        logger.info(f"Confiance globale: {final_result['confidence_globale']:.2f}")
        logger.info(f"  - CV: {final_result['confidence_cv']:.2f}")
        logger.info(f"  - NLP: {final_result['confidence_nlp']:.2f}")
        logger.info(f"  - Gabarits: {final_result['confidence_gabarits']:.2f}")
        logger.info(f"Temps total: {total_time:.2f}s")
        logger.info("="*60)
        
        # Déterminer si rejeter
        should_reject = fusion_module.should_reject(final_result, threshold=0.6)
        confidence_level = fusion_module.get_confidence_level(final_result['confidence_globale'])
        
        # Construire la réponse
        response = {
            'classification': final_result['classification'],
            'confidence_cv': float(final_result['confidence_cv']),
            'confidence_nlp': float(final_result['confidence_nlp']),
            'confidence_gabarit': float(final_result['confidence_gabarits']),
            'confidence_globale': float(final_result['confidence_globale']),
            'confidence_level': confidence_level,
            'should_reject': should_reject,
            
            # Scores détaillés
            'all_scores': final_result['all_scores'],
            
            # Features détectées
            'features_cv': {
                'models_agree': cv_result.get('models_agree', False),
                'resnet_prediction': cv_result.get('resnet_prediction'),
                'efficientnet_prediction': cv_result.get('efficientnet_prediction')
            },
            'features_nlp': {
                'matched_keywords': nlp_result.get('matched_keywords', [])[:10],
                'entities': {k: v[:3] for k, v in entities.items()},
                'text_quality': nlp_result.get('quality', {}),
                'camembert_used': nlp_result.get('camembert_available', False)
            },
            'features_gabarits': final_result['details']['gabarits']['top_features'],
            
            # Métriques de fusion
            'fusion': {
                'weights_used': final_result['weights_used'],
                'coherence': final_result['coherence'],
                'validation': final_result['validation']
            },
            
            # Temps de traitement
            'processing_times': {
                'ocr': round(ocr_time, 2),
                'cv': round(cv_time, 2),
                'nlp': round(nlp_time, 2),
                'gabarits': round(gabarit_time, 2),
                'fusion': round(fusion_time, 2),
                'total': round(total_time, 2)
            },
            
            # Métadonnées
            'metadata': {
                'image_path': os.path.basename(image_path),
                'text_length': len(text),
                'ocr_confidence': float(ocr_result['confidence']),
                'timestamp': time.time()
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"ERREUR lors de la classification: {e}", exc_info=True)
        return {
            'classification': 'error',
            'confidence_globale': 0.0,
            'error': str(e),
            'processing_times': {'total': time.time() - start_time}
        }

# ============================================
# ROUTES API
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérification de l'état du serveur"""
    return jsonify({
        'status': 'online' if MODULES_LOADED else 'error',
        'modules': {
            'ocr': MODULES_LOADED,
            'nlp': MODULES_LOADED,
            'cv': MODULES_LOADED,
            'gabarits': MODULES_LOADED,
            'fusion': MODULES_LOADED
        },
        'timestamp': time.time()
    }), 200 if MODULES_LOADED else 500

@app.route('/api/classify', methods=['POST'])
def classify():
    """
    Endpoint principal de classification
    
    Accepte: multipart/form-data avec file
    Retourne: Résultat JSON de classification
    """
    if not MODULES_LOADED:
        return jsonify({'error': 'Modules non chargés'}), 500
    
    # Vérifier la présence du fichier
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Format de fichier non supporté. Utilisez PDF, JPG ou PNG'}), 400
    
    try:
        # Sauvegarder le fichier
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logger.info(f"Fichier reçu: {filename}")
        
        # Convertir PDF en image si nécessaire
        if filename.lower().endswith('.pdf'):
            logger.info("Conversion PDF en cours...")
            image_paths = convert_pdf_to_images(filepath)
            if not image_paths:
                return jsonify({
                    'error': 'Impossible de convertir le PDF. Poppler est requis sur Windows.',
                    'solution': 'Télécharger et installer: https://github.com/oschwartz10612/poppler-windows/releases/',
                    'alternative': 'Convertir le PDF en JPG/PNG avant de télécharger'
                }), 500
            filepath = image_paths[0]  # Prendre la première page
            logger.info(f"PDF converti: {len(image_paths)} page(s)")
        
        # Vérifier la taille du fichier
        file_size = os.path.getsize(filepath)
        if file_size > MAX_FILE_SIZE:
            os.remove(filepath)
            return jsonify({'error': 'Fichier trop volumineux (max 10 MB)'}), 400
        
        # Classification
        result = classify_document(filepath)
        
        # Ajouter le nom du fichier
        result['filename'] = filename
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Erreur API: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def batch_classify():
    """
    Classification par lots
    
    Accepte: plusieurs fichiers
    Retourne: Liste de résultats
    """
    if not MODULES_LOADED:
        return jsonify({'error': 'Modules non chargés'}), 500
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    results = []
    
    for file in files[:10]:  # Limiter à 10 fichiers par batch
        if not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'error': 'Format non supporté'
            })
            continue
        
        try:
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            if filename.lower().endswith('.pdf'):
                image_paths = convert_pdf_to_images(filepath)
                if image_paths:
                    filepath = image_paths[0]
            
            result = classify_document(filepath)
            result['filename'] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return jsonify({
        'total': len(results),
        'results': results
    }), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Statistiques globales du système
    """
    # TODO: Implémenter la persistance des stats
    return jsonify({
        'total_processed': 0,
        'by_category': {},
        'avg_processing_time': 0.0,
        'avg_confidence': 0.0
    }), 200

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """
    Liste des catégories supportées (3 classes actives)
    """
    categories = {
        'releve_bancaire': {
            'name': 'Relevé bancaire',
            'description': 'Relevés de compte de toutes banques marocaines',
            'icon': '🏦'
        },
        'facture_electricite': {
            'name': 'Facture d\'électricité',
            'description': 'Factures ONE, RADEM, Lydec, Redal',
            'icon': '⚡'
        },
        'facture_eau': {
            'name': 'Facture d\'eau',
            'description': 'Factures RADEM, Lydec, Redal',
            'icon': '💧'
        }
    }
    
    return jsonify(categories), 200

# ============================================
# LANCEMENT DU SERVEUR
# ============================================

if __name__ == '__main__':
    logger.info("")
    logger.info("="*60)
    logger.info("SERVEUR DE CLASSIFICATION DEMARRÉ")
    logger.info("="*60)
    logger.info("Modules:")
    logger.info("  - OCR: Tesseract avec prétraitement avancé")
    logger.info("  - NLP: CamemBERT + Motifs sémantiques")
    logger.info("  - CV: ResNet50 (EfficientNet disabled)")
    logger.info("  - Gabarits: Détection de patterns structurels")
    logger.info("  - Fusion: Combinaison multimodale intelligente")
    logger.info("="*60)
    logger.info("Endpoints disponibles:")
    logger.info("  - GET  /api/health      -> État du serveur")
    logger.info("  - POST /api/classify    -> Classifier un document")
    logger.info("  - POST /api/batch       -> Classifier plusieurs documents")
    logger.info("  - GET  /api/categories  -> Liste des catégories")
    logger.info("  - GET  /api/stats       -> Statistiques")
    logger.info("="*60)
    logger.info("")
    
    app.run(debug=False, port=5000, host='0.0.0.0')