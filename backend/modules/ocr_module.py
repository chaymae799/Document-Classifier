"""
MODULE OCR - Extraction de Texte Avancée
Utilise Tesseract OCR avec prétraitement d'images optimal
Projet: Classification de Documents - INDIA-S5
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedOCRModule:
    """
    Module OCR avancé avec prétraitement intelligent
    Optimisé pour les documents administratifs marocains
    """
    
    def __init__(self):
        # Configuration Tesseract (décommenter selon votre OS)
        # Windows:
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Configurations OCR pour différents types de documents
        # Allow multiple languages via env var TESS_LANGS (e.g. 'fra+eng+ara')
        import os
        # desired languages from env (e.g. 'fra+eng+ara')
        desired_langs = os.environ.get('TESS_LANGS', 'fra+ara')

        # detect installed languages from tesseract via pytesseract
        try:
            installed = pytesseract.get_languages(config='')
        except Exception:
            installed = []

        self.available_langs = installed

        # choose effective languages: prefer desired if installed, else fall back to English if present
        eff_langs = None
        if 'fra' in installed:
            # keep requested order but ensure fra present
            eff_langs = desired_langs
        elif 'eng' in installed:
            eff_langs = 'eng'
        elif installed:
            eff_langs = '+'.join(installed)
        else:
            eff_langs = ''

        if eff_langs:
            lang_flag = f"-l {eff_langs}"
        else:
            lang_flag = ''

        self.ocr_configs = {
            'default': fr'--oem 3 --psm 6 {lang_flag}',
            'single_block': fr'--oem 3 --psm 6 {lang_flag}',
            'sparse': fr'--oem 3 --psm 11 {lang_flag}',
            'single_line': fr'--oem 3 --psm 7 {lang_flag}',
            'fallback_eng': '--oem 3 --psm 6 -l eng',
            'no_lang': '--oem 3 --psm 6'
        }

        if 'fra' not in installed:
            logger.warning("Tesseract 'fra' traineddata not found; available langs=%s", installed)
        
        logger.info("✓ Module OCR initialisé")
    
    def preprocess_for_ocr(self, image_path: str, method: str = 'adaptive') -> Image.Image:
        """
        Prétraitement avancé d'image pour OCR optimal
        
        Args:
            image_path: Chemin vers l'image
            method: 'adaptive', 'otsu', 'gaussian'
        
        Returns:
            Image PIL prétraitée
        """
        # Charger l'image (cv2 may fail on certain unicode paths) - try cv2 then PIL fallback
        img = cv2.imread(image_path)
        if img is None:
            try:
                pil_raw = Image.open(image_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil_raw), cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # 1. Conversion en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Débruitage avec filtre non-local means
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 3. Amélioration du contraste avec CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 4. Binarisation selon la méthode choisie
        if method == 'adaptive':
            # Seuillage adaptatif (meilleur pour éclairage inégal)
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == 'otsu':
            # Méthode d'Otsu (bon pour contraste uniforme)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Seuillage gaussien
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Opérations morphologiques pour nettoyer le bruit
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 6. Redimensionnement si l'image est trop petite
        height, width = binary.shape
        if width < 1500:
            scale_factor = 1800 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            binary = cv2.resize(binary, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Conversion en PIL Image
        pil_image = Image.fromarray(binary)
        
        # 7. Amélioration finale avec PIL
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        return pil_image
    
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Correction de l'inclinaison du document (deskew)
        
        Args:
            image: Image numpy array
        
        Returns:
            Image redressée
        """
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Détection des bords
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Détection de lignes avec Hough
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None:
            # Calculer l'angle moyen
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.rad2deg(theta) - 90
                if -45 < angle < 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                
                # Rotation de l'image
                if abs(median_angle) > 0.5:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(
                        image, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    return rotated
        
        return image
    
    def extract_text(self, image_path: str, config_type: str = 'default') -> str:
        """
        Extraction de texte avec OCR optimisé
        
        Args:
            image_path: Chemin vers l'image
            config_type: Type de configuration OCR
        
        Returns:
            Texte extrait
        """
        try:
            logger.info(f"Extraction OCR de: {image_path}")
            
            # Prétraitement
            preprocessed = self.preprocess_for_ocr(image_path)
            
            # OCR avec configuration spécifique
            config = self.ocr_configs.get(config_type, self.ocr_configs['default'])
            text = pytesseract.image_to_string(preprocessed, config=config)

            # If OCR empty or very short, try fallbacks: English-only, then no language flag
            if not text or len(text.strip()) < 5:
                logger.debug('OCR empty with config %s, trying fallback_eng', config)
                try:
                    text = pytesseract.image_to_string(preprocessed, config=self.ocr_configs.get('fallback_eng'))
                except Exception:
                    text = ''

            if not text or len(text.strip()) < 5:
                logger.debug('Still empty, trying no_lang fallback')
                try:
                    text = pytesseract.image_to_string(preprocessed, config=self.ocr_configs.get('no_lang'))
                except Exception:
                    text = ''

            # Post-traitement du texte
            text = self._postprocess_text(text)

            logger.info(f"✓ OCR: {len(text)} caractères extraits (langs available={self.available_langs})")
            return text
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction OCR: {e}")
            return ""
    
    def extract_with_confidence(self, image_path: str) -> Dict:
        """
        Extraction avec scores de confiance par mot
        
        Args:
            image_path: Chemin vers l'image
        
        Returns:
            Dictionnaire avec texte et scores de confiance
        """
        try:
            preprocessed = self.preprocess_for_ocr(image_path)
            
            # OCR avec données détaillées
            data = pytesseract.image_to_data(
                preprocessed,
                config=self.ocr_configs['default'],
                output_type=pytesseract.Output.DICT
            )
            
            # Filtrer les mots avec bonne confiance
            high_confidence_words = []
            low_confidence_words = []
            
            for i, word in enumerate(data['text']):
                if word.strip():
                    conf = int(data['conf'][i])
                    if conf > 60:
                        high_confidence_words.append(word)
                    elif conf > 0:
                        low_confidence_words.append(word)
            
            full_text = ' '.join(high_confidence_words + low_confidence_words)
            avg_confidence = np.mean([int(c) for c in data['conf'] if int(c) > 0]) if data['conf'] else 0

            # If nothing found or no confidence, retry with English fallback then no-lang
            if (len(full_text.strip()) == 0) or (avg_confidence == 0):
                logger.debug('No OCR words/confidence, retrying with fallback_eng')
                try:
                    data = pytesseract.image_to_data(
                        preprocessed,
                        config=self.ocr_configs.get('fallback_eng'),
                        output_type=pytesseract.Output.DICT
                    )
                except Exception:
                    data = {'text': [], 'conf': []}

                # recompute lists
                high_confidence_words = []
                low_confidence_words = []
                for i, word in enumerate(data.get('text', [])):
                    if str(word).strip():
                        try:
                            conf = int(data['conf'][i])
                        except Exception:
                            conf = 0
                        if conf > 60:
                            high_confidence_words.append(word)
                        elif conf > 0:
                            low_confidence_words.append(word)

                full_text = ' '.join(high_confidence_words + low_confidence_words)
                try:
                    avg_confidence = np.mean([int(c) for c in data.get('conf', []) if int(c) > 0]) if data.get('conf') else 0
                except Exception:
                    avg_confidence = 0
            
            return {
                'text': self._postprocess_text(full_text),
                'confidence': float(avg_confidence) / 100.0,
                'high_confidence_words': high_confidence_words[:50],
                'word_count': len(high_confidence_words) + len(low_confidence_words)
            }
            
        except Exception as e:
            logger.error(f"Erreur OCR avec confiance: {e}")
            return {'text': '', 'confidence': 0.0, 'high_confidence_words': [], 'word_count': 0}
    
    def extract_structured_data(self, image_path: str) -> Dict:
        """
        Extraction de données structurées (tableaux, lignes)
        
        Args:
            image_path: Chemin vers l'image
        
        Returns:
            Dictionnaire avec données structurées
        """
        try:
            preprocessed = self.preprocess_for_ocr(image_path)
            
            # OCR avec structure TSV
            tsv_data = pytesseract.image_to_data(
                preprocessed,
                config=self.ocr_configs['default'],
                output_type=pytesseract.Output.DICT
            )

            # If no structured lines detected, retry with English fallback
            if not any(tsv_data.get('text', [])):
                try:
                    tsv_data = pytesseract.image_to_data(
                        preprocessed,
                        config=self.ocr_configs.get('fallback_eng'),
                        output_type=pytesseract.Output.DICT
                    )
                except Exception:
                    tsv_data = {'text': [], 'block_num': [], 'line_num': []}
            
            # Organiser par lignes
            lines = {}
            for i in range(len(tsv_data['text'])):
                if tsv_data['text'][i].strip():
                    block_num = tsv_data['block_num'][i]
                    line_num = tsv_data['line_num'][i]
                    key = f"{block_num}_{line_num}"
                    
                    if key not in lines:
                        lines[key] = []
                    lines[key].append(tsv_data['text'][i])
            
            # Reconstruire les lignes
            structured_lines = [' '.join(words) for words in lines.values()]
            
            return {
                'lines': structured_lines,
                'line_count': len(structured_lines),
                'full_text': '\n'.join(structured_lines)
            }
            
        except Exception as e:
            logger.error(f"Erreur extraction structurée: {e}")
            return {'lines': [], 'line_count': 0, 'full_text': ''}
    
    def extract_numbers(self, text: str) -> List[str]:
        """
        Extraction des nombres du texte
        
        Args:
            text: Texte source
        
        Returns:
            Liste des nombres trouvés
        """
        # Pattern pour nombres avec séparateurs
        pattern = r'\b\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d+)?\b'
        numbers = re.findall(pattern, text)
        return numbers
    
    def extract_dates(self, text: str) -> List[str]:
        """
        Extraction des dates du texte
        
        Args:
            text: Texte source
        
        Returns:
            Liste des dates trouvées
        """
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # JJ/MM/AAAA
            r'\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'  # AAAA/MM/JJ
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return dates
    
    def _postprocess_text(self, text: str) -> str:
        """
        Post-traitement du texte OCR
        
        Args:
            text: Texte brut OCR
        
        Returns:
            Texte nettoyé
        """
        # Conversion en minuscules
        text = text.lower()
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les caractères spéciaux isolés
        text = re.sub(r'\s[^\w\s]\s', ' ', text)
        
        # Corrections courantes d'erreurs OCR
        ocr_corrections = {
            'identit6': 'identité',
            'nationaie': 'nationale',
            'numer0': 'numero',
            'kwfi': 'kwh',
            'm3': 'm³',
            'dirfiams': 'dirhams',
            'emp1oyeur': 'employeur',
            'sa1aire': 'salaire'
        }
        
        for wrong, correct in ocr_corrections.items():
            text = text.replace(wrong, correct)
        
        return text.strip()
    
    def get_ocr_quality_score(self, image_path: str) -> float:
        """
        Évalue la qualité de l'image pour l'OCR
        
        Args:
            image_path: Chemin vers l'image
        
        Returns:
            Score de qualité (0-1)
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Critères de qualité
        # 1. Résolution
        height, width = img.shape
        resolution_score = min((width * height) / (1500 * 2000), 1.0)
        
        # 2. Contraste
        contrast = img.std()
        contrast_score = min(contrast / 80, 1.0)
        
        # 3. Netteté (variance du Laplacien)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(sharpness / 500, 1.0)
        
        # Score global
        quality_score = (resolution_score + contrast_score + sharpness_score) / 3
        
        return float(quality_score)
    
    def batch_extract(self, image_paths: List[str]) -> List[Dict]:
        """
        Extraction OCR par lots
        
        Args:
            image_paths: Liste des chemins d'images
        
        Returns:
            Liste des résultats OCR
        """
        results = []
        
        for image_path in image_paths:
            result = self.extract_with_confidence(image_path)
            result['image_path'] = image_path
            result['quality_score'] = self.get_ocr_quality_score(image_path)
            results.append(result)
        
        return results