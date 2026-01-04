"""
MODULE GABARITS - Détection de Patterns Structurels
Analyse la structure visuelle des documents
Projet: Classification de Documents - INDIA-S5
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class GabaritsModule:
    """
    Module de détection de gabarits (patterns structurels)
    Analyse sans lire le texte: structure, layout, éléments visuels
    """
    
    def __init__(self):
        # Configuration des gabarits par catégorie
        self.gabarit_rules = {
            'piece_identite': {
                'aspect_ratio': (1.50, 1.70),  # Format carte
                'has_photo': True,
                'text_density': (0.15, 0.45),
                'color_zones': ['blue', 'red'],
                'weights': {
                    'aspect_ratio': 0.25,
                    'photo': 0.35,
                    'text_density': 0.20,
                    'colors': 0.20
                }
            },
            'releve_bancaire': {
                'aspect_ratio': (1.35, 1.50),  # Format A4
                'has_table': True,
                'text_density': (0.45, 0.75),
                'has_logo': True,
                'weights': {
                    'aspect_ratio': 0.15,
                    'table': 0.40,
                    'text_density': 0.25,
                    'logo': 0.20
                }
            },
            'facture_electricite': {
                'aspect_ratio': (1.35, 1.50),
                'has_table': True,
                'text_density': (0.40, 0.70),
                'has_graphs': True,
                'weights': {
                    'aspect_ratio': 0.15,
                    'table': 0.35,
                    'text_density': 0.25,
                    'graphs': 0.25
                }
            },
            'facture_eau': {
                'aspect_ratio': (1.35, 1.50),
                'has_table': True,
                'text_density': (0.40, 0.70),
                'has_graphs': True,
                'weights': {
                    'aspect_ratio': 0.15,
                    'table': 0.35,
                    'text_density': 0.25,
                    'graphs': 0.25
                }
            },
            'document_employeur': {
                'aspect_ratio': (1.35, 1.50),
                'text_density': (0.50, 0.80),
                'has_signature': True,
                'has_sections': True,
                'weights': {
                    'aspect_ratio': 0.15,
                    'text_density': 0.30,
                    'signature': 0.30,
                    'sections': 0.25
                }
            }
        }
        
        # Cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        logger.info("✓ Module Gabarits initialisé")
    
    def analyze(self, image_path: str) -> Dict:
        """
        Analyse complète des gabarits d'un document
        
        Args:
            image_path: Chemin vers l'image
        
        Returns:
            Features détectées et scores par catégorie
        """
        logger.info(f"Analyse gabarits: {image_path}")
        
        # load image; cv2.imread may fail on unicode paths - try PIL fallback
        img = cv2.imread(image_path)
        if img is None:
            try:
                from PIL import Image
                pil = Image.open(image_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                logger.error(f"Impossible de charger: {image_path}")
                return self._empty_result()
        
        # Extraction de toutes les features
        features = {
            'aspect_ratio': self._get_aspect_ratio(img),
            'has_photo': self._detect_photo(img),
            'photo_confidence': self._detect_photo_confidence(img),
            'has_table': self._detect_table(img),
            'table_metrics': self._get_table_metrics(img),
            'text_density': self._calculate_text_density(img),
            'color_analysis': self._analyze_colors(img),
            'has_logo': self._detect_logo(img),
            'has_graphs': self._detect_graphs(img),
            'has_signature': self._detect_signature(img),
            'has_sections': self._detect_sections(img),
            'layout_structure': self._analyze_layout(img)
        }
        
        # Calcul des scores par catégorie
        scores = {}
        for category, rules in self.gabarit_rules.items():
            score = self._calculate_category_score(features, category, rules)
            scores[category] = score
        
        logger.info(f"  ✓ Gabarits: {max(scores, key=scores.get)} ({max(scores.values()):.3f})")
        
        return {
            'features': features,
            'scores': scores
        }
    
    def _get_aspect_ratio(self, img: np.ndarray) -> float:
        """Calcule le ratio hauteur/largeur"""
        h, w = img.shape[:2]
        return h / w if w > 0 else 0
    
    def _detect_photo(self, img: np.ndarray) -> bool:
        """Détecte la présence d'une photo (visage)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return len(faces) > 0
    
    def _detect_photo_confidence(self, img: np.ndarray) -> float:
        """Score de confiance pour la détection de photo"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return 0.0
        
        # Score basé sur la taille et position du visage
        h, w = img.shape[:2]
        best_score = 0.0
        
        for (x, y, face_w, face_h) in faces:
            size_ratio = (face_w * face_h) / (w * h)
            position_score = 1.0 if x < w * 0.4 else 0.5  # Gauche pour CNIE
            score = min(size_ratio * 20 * position_score, 1.0)
            best_score = max(best_score, score)
        
        return float(best_score)
    
    def _detect_table(self, img: np.ndarray) -> bool:
        """Détecte une structure tabulaire"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Détection de lignes avec Hough
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=100,
            minLineLength=100, maxLineGap=10
        )
        
        if lines is None:
            return False
        
        # Compter lignes horizontales et verticales
        h_lines = 0
        v_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:
                h_lines += 1
            elif 80 < angle < 100:
                v_lines += 1
        
        return h_lines >= 3 and v_lines >= 2
    
    def _get_table_metrics(self, img: np.ndarray) -> Dict:
        """Métriques détaillées sur la structure tabulaire"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=100,
            minLineLength=100, maxLineGap=10
        )
        
        if lines is None:
            return {'horizontal': 0, 'vertical': 0, 'intersections': 0}
        
        h_count = 0
        v_count = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:
                h_count += 1
            elif 80 < angle < 100:
                v_count += 1
        
        return {
            'horizontal': h_count,
            'vertical': v_count,
            'intersections': h_count * v_count,
            'confidence': min((h_count * v_count) / 50.0, 1.0)
        }
    
    def _calculate_text_density(self, img: np.ndarray) -> float:
        """Calcule la densité de texte"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarisation adaptative
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphologie pour connecter le texte
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        text_pixels = cv2.countNonZero(binary)
        total_pixels = binary.shape[0] * binary.shape[1]
        
        return text_pixels / total_pixels
    
    def _analyze_colors(self, img: np.ndarray) -> Dict:
        """Analyse des couleurs dominantes"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Détection de bleu (CNIE)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_ratio = cv2.countNonZero(blue_mask) / (img.shape[0] * img.shape[1])
        
        # Détection de rouge (CNIE)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, red_lower1, red_upper1),
            cv2.inRange(hsv, red_lower2, red_upper2)
        )
        red_ratio = cv2.countNonZero(red_mask) / (img.shape[0] * img.shape[1])
        
        # Détection de vert
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_ratio = cv2.countNonZero(green_mask) / (img.shape[0] * img.shape[1])
        
        return {
            'blue': float(blue_ratio),
            'red': float(red_ratio),
            'green': float(green_ratio),
            'blue_red_combined': float(blue_ratio + red_ratio)
        }
    
    def _detect_logo(self, img: np.ndarray) -> bool:
        """Détecte la présence d'un logo (en haut)"""
        h, w = img.shape[:2]
        top_region = img[:int(h * 0.2), :]
        
        gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Rechercher des formes compactes (logos)
        logo_candidates = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:  # Taille typique d'un logo
                logo_candidates += 1
        
        return logo_candidates > 0
    
    def _detect_graphs(self, img: np.ndarray) -> bool:
        """Détecte la présence de graphiques/diagrammes"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Détection de cercles (graphiques circulaires)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=20, maxRadius=100
        )
        
        has_circles = circles is not None
        
        # Détection de courbes (graphiques linéaires)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        curved_contours = 0
        for contour in contours:
            if len(contour) > 50:  # Contours longs
                hull = cv2.convexHull(contour)
                if len(hull) > 10:
                    curved_contours += 1
        
        return has_circles or curved_contours > 2
    
    def _detect_signature(self, img: np.ndarray) -> bool:
        """Détecte une zone de signature (bas du document)"""
        h, w = img.shape[:2]
        bottom_third = img[int(h * 0.66):, :]
        
        gray = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Rechercher des formes allongées horizontales
        signature_candidates = 0
        for contour in contours:
            x, y, w_cont, h_cont = cv2.boundingRect(contour)
            aspect_ratio = w_cont / h_cont if h_cont > 0 else 0
            
            if 2 < aspect_ratio < 8 and w_cont > 50:
                signature_candidates += 1
        
        return signature_candidates > 0
    
    def _detect_sections(self, img: np.ndarray) -> bool:
        """Détecte la présence de sections distinctes"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Profil de projection horizontale
        horizontal_proj = np.sum(gray < 200, axis=1)
        
        # Détecter les zones blanches (séparateurs)
        threshold = np.mean(horizontal_proj) * 0.3
        separators = horizontal_proj < threshold
        
        # Compter les transitions (sections)
        transitions = np.sum(np.diff(separators.astype(int)) != 0)
        
        return transitions >= 4  # Au moins 2 sections
    
    def _analyze_layout(self, img: np.ndarray) -> Dict:
        """Analyse la structure du layout"""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Diviser en régions
        top_region = gray[:int(h * 0.33), :]
        middle_region = gray[int(h * 0.33):int(h * 0.66), :]
        bottom_region = gray[int(h * 0.66):, :]
        
        # Densité par région
        top_density = np.sum(top_region < 200) / top_region.size
        middle_density = np.sum(middle_region < 200) / middle_region.size
        bottom_density = np.sum(bottom_region < 200) / bottom_region.size
        
        return {
            'top_density': float(top_density),
            'middle_density': float(middle_density),
            'bottom_density': float(bottom_density),
            'is_balanced': abs(top_density - bottom_density) < 0.2
        }
    
    def _calculate_category_score(
        self, 
        features: Dict, 
        category: str, 
        rules: Dict
    ) -> float:
        """Calcule le score pour une catégorie donnée"""
        score = 0.0
        weights = rules['weights']
        
        # Vérifier aspect ratio
        if 'aspect_ratio' in rules:
            min_r, max_r = rules['aspect_ratio']
            if min_r <= features['aspect_ratio'] <= max_r:
                score += weights.get('aspect_ratio', 0.2)
        
        # Vérifier photo
        if rules.get('has_photo'):
            if features['has_photo']:
                score += weights.get('photo', 0.3) * features['photo_confidence']
        
        # Vérifier table
        if rules.get('has_table'):
            if features['has_table']:
                table_conf = features['table_metrics'].get('confidence', 0.5)
                score += weights.get('table', 0.3) * table_conf
        
        # Vérifier densité de texte
        if 'text_density' in rules:
            min_d, max_d = rules['text_density']
            if min_d <= features['text_density'] <= max_d:
                score += weights.get('text_density', 0.2)
        
        # Vérifier couleurs (pour CNIE)
        if 'color_zones' in rules:
            if 'blue' in rules['color_zones'] or 'red' in rules['color_zones']:
                color_score = features['color_analysis']['blue_red_combined']
                if color_score > 0.05:
                    score += weights.get('colors', 0.15) * min(color_score * 5, 1.0)
        
        # Vérifier logo
        if rules.get('has_logo'):
            if features['has_logo']:
                score += weights.get('logo', 0.2)
        
        # Vérifier graphiques
        if rules.get('has_graphs'):
            if features['has_graphs']:
                score += weights.get('graphs', 0.2)
        
        # Vérifier signature
        if rules.get('has_signature'):
            if features['has_signature']:
                score += weights.get('signature', 0.2)
        
        # Vérifier sections
        if rules.get('has_sections'):
            if features['has_sections']:
                score += weights.get('sections', 0.2)
        
        return min(score, 1.0)
    
    def _empty_result(self) -> Dict:
        """Résultat vide en cas d'erreur"""
        return {
            'features': {},
            'scores': {cat: 0.0 for cat in self.gabarit_rules.keys()}
        }