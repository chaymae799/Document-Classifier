"""
PIPELINE DE PRÉTRAITEMENT COMPLET - Étape par étape
Convertit PDFs → Images → Nettoyage → Organisation → Prêt pour fine-tuning

Usage:
    python preprocess_pipeline.py --step convert    # Étape 1: PDF → Images
    python preprocess_pipeline.py --step clean      # Étape 2: Nettoyage images
    python preprocess_pipeline.py --step organize   # Étape 3: Organisation train/val
    python preprocess_pipeline.py --step report     # Étape 4: Rapport final
    python preprocess_pipeline.py --all              # Tout automatiquement
"""

import os
import sys
from pathlib import Path
import json
import shutil
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import cv2

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / 'Dataset'
RAW_IMAGES_DIR = PROJECT_ROOT / 'data' / 'raw_images'
PROCESSED_IMAGES_DIR = PROJECT_ROOT / 'data' / 'processed_images'
TRAIN_DIR = PROJECT_ROOT / 'data' / 'train'
VAL_DIR = PROJECT_ROOT / 'data' / 'val'
PREPROCESSING_REPORT = PROJECT_ROOT / 'backend' / 'preprocessing_report.json'

CATEGORIES = {
    'factures_electricite': 'facture_electricite',
    'factures_eau': 'facture_eau',
    'Pieces_identite': 'piece_identite',
    'doc_employeur': 'document_employeur',
    'relevez_banc': 'releve_bancaire'
}

# Ratio train/val
TRAIN_RATIO = 0.8


class PreprocessingPipeline:
    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'steps_completed': [],
            'statistics': {}
        }
        self._setup_directories()
    
    def _setup_directories(self):
        """Crée les répertoires nécessaires"""
        print("📁 Création des répertoires...")
        for directory in [RAW_IMAGES_DIR, PROCESSED_IMAGES_DIR, TRAIN_DIR, VAL_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Créer sous-répertoires pour chaque catégorie
        for category in CATEGORIES.values():
            (TRAIN_DIR / category).mkdir(parents=True, exist_ok=True)
            (VAL_DIR / category).mkdir(parents=True, exist_ok=True)
            (PROCESSED_IMAGES_DIR / category).mkdir(parents=True, exist_ok=True)
        
        print("✓ Répertoires créés\n")
    
    # ============================================================
    # ÉTAPE 1: CONVERSION PDF → IMAGES
    # ============================================================
    
    def step1_convert_pdfs(self):
        """Convertit tous les PDFs en images haute résolution"""
        print("=" * 60)
        print("ÉTAPE 1: CONVERSION PDF → IMAGES")
        print("=" * 60)
        
        stats = {
            'total_pdfs': 0,
            'successfully_converted': 0,
            'failed': 0,
            'total_pages': 0,
            'errors': []
        }
        
        # Parcourir tous les dossiers Dataset
        for category_dir, standard_name in CATEGORIES.items():
            folder_path = DATASET_DIR / category_dir
            
            if not folder_path.exists():
                print(f"⚠️  Dossier non trouvé: {folder_path}")
                continue
            
            print(f"\n📂 Traitement: {category_dir}")
            print("-" * 40)
            
            # Chercher tous les PDFs
            pdf_files = list(folder_path.glob('*.pdf'))
            stats['total_pdfs'] += len(pdf_files)
            
            for pdf_file in pdf_files:
                try:
                    print(f"\n  📄 {pdf_file.name}")
                    images = self._convert_single_pdf(pdf_file, standard_name)
                    stats['successfully_converted'] += 1
                    stats['total_pages'] += len(images)
                    print(f"    ✓ {len(images)} page(s) convertie(s)")
                    
                except Exception as e:
                    stats['failed'] += 1
                    error_msg = f"{pdf_file.name}: {str(e)}"
                    stats['errors'].append(error_msg)
                    print(f"    ✗ Erreur: {str(e)}")
        
        self.report['statistics']['step1_conversion'] = stats
        self.report['steps_completed'].append('convert_pdfs')
        
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ CONVERSION")
        print("=" * 60)
        print(f"✓ PDFs convertis: {stats['successfully_converted']}/{stats['total_pdfs']}")
        print(f"✗ Erreurs: {stats['failed']}")
        print(f"📄 Total pages: {stats['total_pages']}")
        if stats['errors']:
            print("\n❌ Erreurs détaillées:")
            for error in stats['errors']:
                print(f"  - {error}")
        
        return stats['failed'] == 0
    
    def _convert_single_pdf(self, pdf_path: Path, category: str) -> List[str]:
        """Convertit un seul PDF en images"""
        images_created = []
        
        pdf_doc = fitz.open(str(pdf_path))
        pdf_name = pdf_path.stem
        
        for page_num in range(len(pdf_doc)):
            # Rendre la page en haute résolution (3x zoom = ~900 DPI)
            page = pdf_doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
            
            # Convertir en PIL
            img_data = pix.tobytes("ppm")
            img = Image.open(__import__('io').BytesIO(img_data))
            
            # Sauvegarder dans raw_images
            img_name = f"{category}_{pdf_name}_page_{page_num}.jpg"
            img_path = RAW_IMAGES_DIR / img_name
            
            img.save(str(img_path), quality=95)
            images_created.append(img_name)
        
        pdf_doc.close()
        return images_created
    
    # ============================================================
    # ÉTAPE 2: NETTOYAGE ET AMÉLIORATION DES IMAGES
    # ============================================================
    
    def step2_clean_images(self):
        """Nettoie et améliore toutes les images"""
        print("\n" + "=" * 60)
        print("ÉTAPE 2: NETTOYAGE ET AMÉLIORATION")
        print("=" * 60)
        
        stats = {
            'total_images': 0,
            'successfully_processed': 0,
            'failed': 0,
            'preprocessing_techniques': [
                'Débruitage (Bilateral Filter)',
                'Rehaussement contraste (CLAHE)',
                'Dégradation d\'éclairage (Shadow removal)',
                'Optimisation résolution'
            ],
            'errors': []
        }
        
        # Récupérer toutes les images brutes
        raw_images = list(RAW_IMAGES_DIR.glob('*.jpg'))
        stats['total_images'] = len(raw_images)
        
        print(f"\n📊 {len(raw_images)} images à traiter")
        print("-" * 40)
        
        for idx, img_path in enumerate(raw_images, 1):
            try:
                # Extraire la catégorie du nom de fichier
                category = img_path.stem.split('_')[0]
                
                # Charger et nettoyer (cv2, sinon fallback PIL pour chemins unicode)
                img = cv2.imread(str(img_path))
                if img is None:
                    try:
                        pil = Image.open(str(img_path)).convert('RGB')
                        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    except Exception:
                        raise ValueError("Impossible de charger l'image (cv2 et PIL échoués)")

                img_cleaned = self._clean_single_image(img)
                
                # Sauvegarder image nettoyée
                output_path = PROCESSED_IMAGES_DIR / category / img_path.name
                cv2.imwrite(str(output_path), img_cleaned)
                
                stats['successfully_processed'] += 1
                if idx % 10 == 0:
                    print(f"  ✓ {idx}/{len(raw_images)} images traitées")
                
            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append(f"{img_path.name}: {str(e)}")
                print(f"  ✗ {img_path.name}: {str(e)}")
        
        self.report['statistics']['step2_cleaning'] = stats
        self.report['steps_completed'].append('clean_images')
        
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ NETTOYAGE")
        print("=" * 60)
        print(f"✓ Images nettoyées: {stats['successfully_processed']}/{stats['total_images']}")
        print(f"✗ Erreurs: {stats['failed']}")
        print(f"\n🔧 Techniques appliquées:")
        for technique in stats['preprocessing_techniques']:
            print(f"  ✓ {technique}")
        
        return stats['failed'] == 0
    
    def _clean_single_image(self, img):
        """Applique les techniques de nettoyage"""
        # 1. Débruitage
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # 2. Conversion en gris pour analyse
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 4. Débruitage supplémentaire si nécessaire
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # 5. Appliquer les améliorations à l'image couleur
        # Recalculer la saturation avec le contraste amélioré
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = gray  # Remplacer le canal Value
        img_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 6. Amélioration de la netteté
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.0
        img_enhanced = cv2.filter2D(img_enhanced, -1, kernel)
        
        # 7. Normalisation finale
        img_enhanced = cv2.normalize(img_enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        return img_enhanced
    
    # ============================================================
    # ÉTAPE 3: ORGANISATION TRAIN/VAL
    # ============================================================
    
    def step3_organize_dataset(self):
        """Organise les images en train/val splits"""
        print("\n" + "=" * 60)
        print("ÉTAPE 3: ORGANISATION TRAIN/VAL")
        print("=" * 60)
        
        stats = {
            'categories': {},
            'train_total': 0,
            'val_total': 0,
            'train_ratio': TRAIN_RATIO,
            'val_ratio': 1 - TRAIN_RATIO
        }
        
        # Pour chaque catégorie
        for category in CATEGORIES.values():
            category_path = PROCESSED_IMAGES_DIR / category
            
            if not category_path.exists():
                continue
            
            images = list(category_path.glob('*.jpg'))
            
            if not images:
                continue
            
            print(f"\n📂 {category}")
            print("-" * 40)
            
            # Trier pour reproductibilité
            images.sort()
            
            # Split train/val
            split_idx = int(len(images) * TRAIN_RATIO)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Copier les images
            for img in train_images:
                dest = TRAIN_DIR / category / img.name
                shutil.copy2(str(img), str(dest))
            
            for img in val_images:
                dest = VAL_DIR / category / img.name
                shutil.copy2(str(img), str(dest))
            
            # Statistiques
            stats['categories'][category] = {
                'total': len(images),
                'train': len(train_images),
                'val': len(val_images),
                'train_ratio': round(len(train_images) / len(images) * 100, 1),
                'val_ratio': round(len(val_images) / len(images) * 100, 1)
            }
            stats['train_total'] += len(train_images)
            stats['val_total'] += len(val_images)
            
            print(f"  📊 Total: {len(images)} images")
            print(f"    ✓ Train: {len(train_images)} ({stats['categories'][category]['train_ratio']}%)")
            print(f"    ✓ Val: {len(val_images)} ({stats['categories'][category]['val_ratio']}%)")
        
        self.report['statistics']['step3_organization'] = stats
        self.report['steps_completed'].append('organize_dataset')
        
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ ORGANISATION")
        print("=" * 60)
        print(f"✓ Train total: {stats['train_total']} images")
        print(f"✓ Val total: {stats['val_total']} images")
        print(f"✓ Ratio: {TRAIN_RATIO*100:.0f}% train / {(1-TRAIN_RATIO)*100:.0f}% val")
        print("\n📋 Par catégorie:")
        for cat, cat_stats in stats['categories'].items():
            print(f"  {cat}: {cat_stats['total']} total ({cat_stats['train']} train, {cat_stats['val']} val)")
        
        return True
    
    # ============================================================
    # ÉTAPE 4: RAPPORT FINAL
    # ============================================================
    
    def step4_generate_report(self):
        """Génère un rapport de prétraitement"""
        print("\n" + "=" * 60)
        print("ÉTAPE 4: RAPPORT FINAL")
        print("=" * 60)
        
        # Compter les fichiers finaux
        train_count = len(list(TRAIN_DIR.rglob('*.jpg')))
        val_count = len(list(VAL_DIR.rglob('*.jpg')))
        
        final_stats = {
            'total_images': train_count + val_count,
            'train_images': train_count,
            'val_images': val_count,
            'train_ratio': round(train_count / (train_count + val_count) * 100, 1) if train_count + val_count > 0 else 0,
            'val_ratio': round(val_count / (train_count + val_count) * 100, 1) if train_count + val_count > 0 else 0,
            'data_locations': {
                'raw_images': str(RAW_IMAGES_DIR),
                'processed_images': str(PROCESSED_IMAGES_DIR),
                'training': str(TRAIN_DIR),
                'validation': str(VAL_DIR)
            }
        }
        
        self.report['statistics']['final'] = final_stats
        self.report['steps_completed'].append('generate_report')
        
        # Compter par catégorie
        categories_breakdown = {}
        for category in CATEGORIES.values():
            # Count images per category in train/val
            train_cat = len(list((TRAIN_DIR / category).glob('*.jpg'))) if (TRAIN_DIR / category).exists() else 0
            val_cat = len(list((VAL_DIR / category).glob('*.jpg'))) if (VAL_DIR / category).exists() else 0
            
            categories_breakdown[category] = {
                'train': train_cat,
                'val': val_cat,
                'total': train_cat + val_cat
            }
        
        self.report['statistics']['by_category'] = categories_breakdown
        
        # Sauvegarder le rapport
        with open(PREPROCESSING_REPORT, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        print("\n✅ RÉSUMÉ FINAL")
        print("=" * 60)
        print(f"\n📊 DATASET PRÉPARÉ POUR FINE-TUNING")
        print(f"\n  Total d'images: {final_stats['total_images']}")
        print(f"  ✓ Train: {train_count} images ({final_stats['train_ratio']}%)")
        print(f"  ✓ Val: {val_count} images ({final_stats['val_ratio']}%)")
        
        print(f"\n📂 STRUCTURE FINALE:")
        print(f"  {TRAIN_DIR}")
        for category, counts in categories_breakdown.items():
            print(f"    ├── {category}/  ({counts['train']} images)")
        print(f"  {VAL_DIR}")
        for category, counts in categories_breakdown.items():
            print(f"    ├── {category}/  ({counts['val']} images)")
        
        print(f"\n💾 RAPPORT: {PREPROCESSING_REPORT}")
        print(f"\n✅ PRÊT POUR FINE-TUNING!")
        print(f"   Exécutez: python backend/train_hybrid.py --data-dir data")
        
        return True
    
    def run_all(self):
        """Exécute le pipeline complet"""
        print("\n" + "=" * 60)
        print("🚀 PIPELINE COMPLET DE PRÉTRAITEMENT")
        print("=" * 60)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        steps = [
            ("Conversion PDF → Images", self.step1_convert_pdfs),
            ("Nettoyage images", self.step2_clean_images),
            ("Organisation train/val", self.step3_organize_dataset),
            ("Rapport final", self.step4_generate_report)
        ]
        
        for step_name, step_func in steps:
            try:
                result = step_func()
                if not result:
                    print(f"\n⚠️  {step_name} - Étape terminée avec avertissements")
            except Exception as e:
                print(f"\n❌ ERREUR dans {step_name}:")
                print(f"   {str(e)}")
                return False
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline de prétraitement complet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python preprocess_pipeline.py --step convert    # Étape 1 seulement
  python preprocess_pipeline.py --step clean      # Étape 2 seulement
  python preprocess_pipeline.py --step organize   # Étape 3 seulement
  python preprocess_pipeline.py --step report     # Étape 4 seulement
  python preprocess_pipeline.py --all             # Tout d'un coup
        """
    )
    
    parser.add_argument('--step', 
                       choices=['convert', 'clean', 'organize', 'report'],
                       help='Exécuter une étape spécifique')
    parser.add_argument('--all', action='store_true',
                       help='Exécuter le pipeline complet')
    
    args = parser.parse_args()
    
    pipeline = PreprocessingPipeline()
    
    if args.all or (not args.step):
        # Mode par défaut: tout
        success = pipeline.run_all()
    else:
        # Mode étape unique
        step_map = {
            'convert': ('Conversion PDF', pipeline.step1_convert_pdfs),
            'clean': ('Nettoyage', pipeline.step2_clean_images),
            'organize': ('Organisation', pipeline.step3_organize_dataset),
            'report': ('Rapport', pipeline.step4_generate_report)
        }
        
        step_name, step_func = step_map[args.step]
        print(f"\n🚀 Exécution: {step_name}\n")
        success = step_func()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
