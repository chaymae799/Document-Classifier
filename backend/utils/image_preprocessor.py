"""
Module de prétraitement d'images pour améliorer la qualité OCR et CV
"""

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Classe pour le prétraitement avancé d'images"""
    
    def __init__(self):
        self.target_size = (224, 224)  # Pour les modèles CV
        
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraitement optimisé pour OCR
        
        Pipeline:
        1. Conversion niveaux de gris
        2. Débruitage
        3. Augmentation contraste (CLAHE)
        4. Binarisation adaptative
        5. Suppression du bruit résiduel
        
        Args:
            image: Image numpy array (BGR ou RGB)
            
        Returns:
            Image prétraitée pour OCR
        """
        try:
            # Conversion niveaux de gris
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Redimensionner si trop grande (optimisation)
            h, w = gray.shape
            if w > 2000:
                scale = 2000 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 1. Débruitage non-local means
            denoised = cv2.fastNlMeansDenoising(
                gray, 
                None, 
                h=10, 
                templateWindowSize=7, 
                searchWindowSize=21
            )
            
            # 2. Augmentation du contraste avec CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # 3. Binarisation adaptative
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # 4. Morphologie pour nettoyer le bruit
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            logger.debug("Prétraitement OCR terminé")
            return cleaned
            
        except Exception as e:
            logger.error(f"Erreur prétraitement OCR: {str(e)}")
            return image
    
    def preprocess_for_cv(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraitement pour modèles Computer Vision
        
        Args:
            image: Image numpy array
            
        Returns:
            Image redimensionnée et normalisée
        """
        try:
            # Convertir en RGB si nécessaire
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Redimensionner
            resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Normalisation ImageNet (pour ResNet/EfficientNet)
            normalized = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            
            logger.debug("Prétraitement CV terminé")
            return normalized
            
        except Exception as e:
            logger.error(f"Erreur prétraitement CV: {str(e)}")
            return image
    
    def detect_and_correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Détecte et corrige l'inclinaison d'une image
        
        Args:
            image: Image à corriger
            
        Returns:
            Image corrigée
        """
        try:
            # Convertir en niveaux de gris
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Binarisation
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Trouver les coordonnées des pixels non nuls
            coords = np.column_stack(np.where(binary > 0))
            
            # Calculer l'angle de rotation avec minAreaRect
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                
                # Ajuster l'angle
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90
                
                # Rotation si nécessaire
                if abs(angle) > 0.5:
                    logger.info(f"Correction d'inclinaison: {angle:.2f}°")
                    return self.rotate_image(image, angle)
            
            return image
            
        except Exception as e:
            logger.error(f"Erreur correction inclinaison: {str(e)}")
            return image
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Fait pivoter une image
        
        Args:
            image: Image à pivoter
            angle: Angle en degrés
            
        Returns:
            Image pivotée
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Matrice de rotation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculer nouvelles dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Ajuster la translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Appliquer la rotation
        rotated = cv2.warpAffine(
            image, 
            M, 
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def remove_borders(self, image: np.ndarray, threshold: int = 10) -> np.ndarray:
        """
        Supprime les bordures blanches/noires d'une image
        
        Args:
            image: Image à traiter
            threshold: Seuil pour détecter les bordures
            
        Returns:
            Image sans bordures
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Trouver les contours non blancs
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(binary)
            
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                cropped = image[y:y+h, x:x+w]
                return cropped
            
            return image
            
        except Exception as e:
            logger.error(f"Erreur suppression bordures: {str(e)}")
            return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore le contraste d'une image
        
        Args:
            image: Image à améliorer
            
        Returns:
            Image avec contraste amélioré
        """
        try:
            if len(image.shape) == 3:
                # Pour images couleur, convertir en LAB
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Appliquer CLAHE sur le canal L
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)
                
                # Recombiner
                lab_enhanced = cv2.merge([l_enhanced, a, b])
                enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
                
            else:
                # Pour niveaux de gris
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Erreur amélioration contraste: {str(e)}")
            return image
    
    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Débruite une image
        
        Args:
            image: Image à débruiter
            strength: Force du débruitage (1-30)
            
        Returns:
            Image débruitée
        """
        try:
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(
                    image, None, strength, strength, 7, 21
                )
            else:
                denoised = cv2.fastNlMeansDenoising(
                    image, None, strength, 7, 21
                )
            
            return denoised
            
        except Exception as e:
            logger.error(f"Erreur débruitage: {str(e)}")
            return image
    
    def full_preprocess_pipeline(self, image: np.ndarray, 
                                 for_ocr: bool = True,
                                 correct_skew: bool = True,
                                 remove_borders: bool = True) -> dict:
        """
        Pipeline complet de prétraitement
        
        Args:
            image: Image d'entrée
            for_ocr: Si True, optimise pour OCR
            correct_skew: Corriger l'inclinaison
            remove_borders: Supprimer les bordures
            
        Returns:
            Dict avec différentes versions de l'image
        """
        results = {
            'original': image.copy()
        }
        
        try:
            # Suppression des bordures
            if remove_borders:
                image = self.remove_borders(image)
                results['no_borders'] = image.copy()
            
            # Correction de l'inclinaison
            if correct_skew:
                image = self.detect_and_correct_skew(image)
                results['corrected'] = image.copy()
            
            # Prétraitement pour OCR
            if for_ocr:
                results['ocr_ready'] = self.preprocess_for_ocr(image)
            
            # Prétraitement pour CV (toujours)
            results['cv_ready'] = self.preprocess_for_cv(image)
            
            logger.info("Pipeline de prétraitement complet terminé")
            
        except Exception as e:
            logger.error(f"Erreur pipeline prétraitement: {str(e)}")
        
        return results
    
    def get_image_quality_metrics(self, image: np.ndarray) -> dict:
        """
        Calcule des métriques de qualité d'image
        
        Args:
            image: Image à analyser
            
        Returns:
            Dict avec métriques de qualité
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Netteté (variance du Laplacien)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Contraste
            contrast = gray.std()
            
            # Luminosité moyenne
            brightness = gray.mean()
            
            # Exposition (histogramme)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            overexposed = (hist[250:].sum() / hist.sum()) * 100
            underexposed = (hist[:5].sum() / hist.sum()) * 100
            
            metrics = {
                'sharpness': float(laplacian_var),
                'contrast': float(contrast),
                'brightness': float(brightness),
                'overexposed_percent': float(overexposed),
                'underexposed_percent': float(underexposed),
                'quality_score': self._calculate_quality_score(
                    laplacian_var, contrast, brightness
                )
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques: {str(e)}")
            return {}
    
    def _calculate_quality_score(self, sharpness: float, 
                                 contrast: float, 
                                 brightness: float) -> float:
        """
        Calcule un score de qualité global (0-100)
        
        Args:
            sharpness: Netteté
            contrast: Contraste
            brightness: Luminosité
            
        Returns:
            Score de qualité (0-100)
        """
        # Normaliser les valeurs
        sharpness_score = min(sharpness / 1000, 1.0) * 40  # Max 40 points
        contrast_score = min(contrast / 100, 1.0) * 30     # Max 30 points
        
        # Luminosité optimale autour de 127
        brightness_score = (1 - abs(brightness - 127) / 127) * 30  # Max 30 points
        
        total_score = sharpness_score + contrast_score + brightness_score
        
        return min(total_score, 100.0)


# Fonction utilitaire pour usage rapide
def preprocess_image(image: np.ndarray, 
                    for_ocr: bool = True,
                    for_cv: bool = True) -> dict:
    """
    Fonction helper pour prétraitement rapide
    
    Args:
        image: Image à prétraiter
        for_ocr: Inclure prétraitement OCR
        for_cv: Inclure prétraitement CV
        
    Returns:
        Dict avec images prétraitées
    """
    preprocessor = ImagePreprocessor()
    results = {}
    
    if for_ocr:
        results['ocr'] = preprocessor.preprocess_for_ocr(image)
    
    if for_cv:
        results['cv'] = preprocessor.preprocess_for_cv(image)
    
    return results