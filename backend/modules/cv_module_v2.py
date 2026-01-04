"""
MODULE CV AMÉLIORÉ - Utilise le modèle fine-tuné Hybride (ResNet50 + Gabarits)
Si le modèle fine-tuné n'existe pas, fallback sur ResNet50 pretrained seul
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import cv2
from typing import Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GabaritsExtractor:
    """Extrait les 36 features structurelles (30 vision + 6 OCR-derived)"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.ocr_dir = Path(__file__).parent.parent / 'data' / 'ocr_text'
    
    def extract(self, image_path):
        """Extrait 36 features structurelles d'une image (30 vision + 6 OCR-derived)"""
        img = cv2.imread(str(image_path))
        if img is None:
            return np.zeros(30, dtype=np.float32)
        
        h, w = img.shape[:2]
        features = []
        
        # 1-2: Aspect ratio
        aspect_ratio = h / w if w > 0 else 0
        features.append(min(aspect_ratio, 2.0))
        features.append(abs(1 - aspect_ratio))
        
        # 3-5: Photo detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        features.append(1.0 if len(faces) > 0 else 0.0)
        features.append(float(len(faces)) / 10.0)
        
        photo_conf = 0.0
        for (x, y, fw, fh) in faces:
            size_ratio = (fw * fh) / (w * h)
            photo_conf = max(photo_conf, min(size_ratio * 20, 1.0))
        features.append(photo_conf)
        
        # 6-7: Text density
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        text_density = cv2.countNonZero(binary) / (h * w)
        features.append(text_density)
        features.append(1 - text_density)
        
        # 8-10: Table detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        h_lines = v_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:
                    h_lines += 1
                elif 80 < angle < 100:
                    v_lines += 1
        
        features.append(min(h_lines / 20.0, 1.0))
        features.append(min(v_lines / 20.0, 1.0))
        features.append(1.0 if (h_lines >= 3 and v_lines >= 2) else 0.0)
        
        # 11-14: Color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_ratio = cv2.countNonZero(cv2.inRange(hsv, blue_lower, blue_upper)) / (h * w)
        features.append(blue_ratio)
        
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(cv2.inRange(hsv, red_lower1, red_upper1),
                                  cv2.inRange(hsv, red_lower2, red_upper2))
        red_ratio = cv2.countNonZero(red_mask) / (h * w)
        features.append(red_ratio)
        
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_ratio = cv2.countNonZero(cv2.inRange(hsv, green_lower, green_upper)) / (h * w)
        features.append(green_ratio)
        
        dominant_color = max([blue_ratio, red_ratio, green_ratio])
        features.append(dominant_color)
        
        # 15-17: Logo detection
        top_region = img[:int(h * 0.2), :]
        gray_top = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
        _, binary_top = cv2.threshold(gray_top, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logo_candidates = sum(1 for c in contours if 1000 < cv2.contourArea(c) < 50000)
        features.append(1.0 if logo_candidates > 0 else 0.0)
        features.append(float(logo_candidates) / 5.0)
        
        logo_density = cv2.countNonZero(binary_top) / (top_region.shape[0] * top_region.shape[1])
        features.append(logo_density)
        
        # 18-20: Graphs
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30,
                                   minRadius=20, maxRadius=100)
        has_circles = 1.0 if circles is not None else 0.0
        features.append(has_circles)
        
        curved_contours = sum(1 for c in contours if len(c) > 50)
        features.append(min(curved_contours / 5.0, 1.0))
        
        graph_conf = (has_circles * 0.5) + (min(curved_contours / 5.0, 1.0) * 0.5)
        features.append(graph_conf)
        
        # 21-24: Layout
        top_third = gray[:int(h * 0.33), :]
        mid_third = gray[int(h * 0.33):int(h * 0.66), :]
        bot_third = gray[int(h * 0.66):, :]
        
        top_density = np.sum(top_third < 200) / max(top_third.size, 1)
        mid_density = np.sum(mid_third < 200) / max(mid_third.size, 1)
        bot_density = np.sum(bot_third < 200) / max(bot_third.size, 1)
        
        features.append(top_density)
        features.append(mid_density)
        features.append(bot_density)
        features.append(1.0 - abs(top_density - bot_density))
        
        # 25-27: Signature
        bottom_third = img[int(h * 0.66):, :]
        gray_bottom = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
        edges_bottom = cv2.Canny(gray_bottom, 30, 100)
        contours_bottom, _ = cv2.findContours(edges_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sig_candidates = 0
        for c in contours_bottom:
            x, y, w_cont, h_cont = cv2.boundingRect(c)
            if h_cont > 0:
                ar = w_cont / h_cont
                if 2 < ar < 8 and w_cont > 50:
                    sig_candidates += 1
        
        features.append(1.0 if sig_candidates > 0 else 0.0)
        features.append(float(sig_candidates) / 5.0)
        features.append(bot_density)
        
        # 28-30: Quality
        contrast = gray.std()
        features.append(min(contrast / 80.0, 1.0))
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        features.append(min(sharpness / 500.0, 1.0))
        
        quality = (min(contrast / 80.0, 1.0) + min(sharpness / 500.0, 1.0)) / 2
        features.append(quality)
        
        # 31-36: OCR-derived features (6 features)
        try:
            ocr_file = self.ocr_dir / (Path(image_path).stem + '.txt')
            if ocr_file.exists():
                text = ocr_file.read_text(encoding='utf-8')
                words = [w for w in text.split() if len(w) > 0]
                word_count = len(words)
                avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
                chars = list(text)
                digit_ratio = sum(c.isdigit() for c in chars) / max(len(chars), 1)
                lower = text.lower()
                has_facture = 1.0 if ('facture' in lower or 'factures' in lower) else 0.0
                has_montant = 1.0 if ('montant' in lower or 'total' in lower or 'tds' in lower) else 0.0
                has_iban = 1.0 if ('iban' in lower or 'bank' in lower or 'compte' in lower) else 0.0
            else:
                word_count = 0.0
                avg_word_len = 0.0
                digit_ratio = 0.0
                has_facture = 0.0
                has_montant = 0.0
                has_iban = 0.0
        except Exception:
            word_count = 0.0
            avg_word_len = 0.0
            digit_ratio = 0.0
            has_facture = 0.0
            has_montant = 0.0
            has_iban = 0.0

        features.append(min(word_count / 500.0, 1.0))
        features.append(min(avg_word_len / 10.0, 1.0))
        features.append(digit_ratio)
        features.append(has_facture)
        features.append(has_montant)
        features.append(has_iban)
        
        return np.array(features, dtype=np.float32)


class HybridResNet50(nn.Module):
    """Modèle hybride: ResNet50 (image) + Dense (gabarits) + Fusion"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.resnet50 = resnet50(weights='IMAGENET1K_V1')
        self.resnet50.fc = nn.Identity()
        
        self.gabarit_net = nn.Sequential(
            nn.Linear(36, 128),  # Updated to 36 to match trained checkpoint
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, gabarit):
        cv_features = self.resnet50(image)
        gab_features = self.gabarit_net(gabarit)
        combined = torch.cat([cv_features, gab_features], dim=1)
        logits = self.fusion(combined)
        return logits


class HybridCVModule:
    """Module CV amélioré utilisant le modèle fine-tuné"""
    
    CLASSES = ['facture_eau', 'facture_electricite', 'piece_identite', 'releve_bancaire', 'document_employeur']
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.gabarit_extractor = GabaritsExtractor()
        
        # Transforms
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Charger le modèle fine-tuné si disponible, sinon ResNet50 pretrained
        self.hybrid_model = None
        self.resnet_only = None
        self._load_model()
        
        logger.info("✓ Module CV initialisé")
    
    def _load_model(self):
        """Charge le modèle fine-tuné ou ResNet50 seul"""
        # Essayer plusieurs chemins possibles
        possible_paths = [
            Path(__file__).parent.parent / 'models' / 'cv' / 'hybrid_resnet50.pth',  # Depuis modules/
            Path('/backend/models/cv/hybrid_resnet50.pth'),
            Path('backend/models/cv/hybrid_resnet50.pth'),
            Path('models/cv/hybrid_resnet50.pth'),
        ]
        
        hybrid_path = None
        for path in possible_paths:
            if path.exists():
                hybrid_path = path
                break
        
        if hybrid_path is None:
            # Essayer relative au répertoire courant
            import os
            if os.path.exists('models/cv/hybrid_resnet50.pth'):
                hybrid_path = Path('models/cv/hybrid_resnet50.pth')
        
        if hybrid_path and hybrid_path.exists():
            try:
                logger.info(f"Chargement du modèle fine-tuné: {hybrid_path}")
                checkpoint = torch.load(hybrid_path, map_location=self.device)
                
                self.hybrid_model = HybridResNet50(num_classes=3).to(self.device)
                # Handle both formats: checkpoint dict or direct state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.hybrid_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.hybrid_model.load_state_dict(checkpoint)
                self.hybrid_model.eval()
                
                logger.info("✓ Modèle fine-tuné chargé avec succès")
            except Exception as e:
                logger.warning(f"Impossible de charger le modèle fine-tuné: {e}")
                self.hybrid_model = None
        else:
            logger.info("Modèle fine-tuné non trouvé, utilisation de ResNet50 pretrained")
        
        # Fallback: ResNet50 seul
        if self.hybrid_model is None:
            self.resnet_only = resnet50(weights='IMAGENET1K_V1')
            self.resnet_only.fc = nn.Linear(self.resnet_only.fc.in_features, 5)
            self.resnet_only = self.resnet_only.to(self.device)
            self.resnet_only.eval()
            logger.info("✓ ResNet50 pretrained chargé en fallback")
    
    def classify(self, image_path: str) -> Dict:
        """Classifie une image"""
        try:
            # Charger image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
            
            # Extraire features gabarits
            gab_features = self.gabarit_extractor.extract(image_path)
            gab_tensor = torch.from_numpy(gab_features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.hybrid_model is not None:
                    # Utiliser le modèle hybride fine-tuné
                    logits = self.hybrid_model(img_tensor, gab_tensor)
                else:
                    # Fallback: ResNet50 seul
                    logits = self.resnet_only(img_tensor)
                
                probs = torch.softmax(logits, dim=1)
                scores = probs[0].cpu().numpy()
                predicted_idx = torch.argmax(probs, dim=1).item()
                predicted_class = self.CLASSES[predicted_idx]
                confidence = float(scores[predicted_idx])
            
            # Scores pour toutes les classes
            class_scores = {self.CLASSES[i]: float(scores[i]) for i in range(len(self.CLASSES))}
            
            return {
                'class': predicted_class,
                'prediction': predicted_class,
                'confidence': confidence,
                'scores': class_scores,
                'use_hybrid': self.hybrid_model is not None
            }
        
        except Exception as e:
            logger.error(f"Erreur classification CV: {e}")
            return {
                'prediction': None,
                'confidence': 0.0,
                'scores': {cls: 0.0 for cls in self.CLASSES},
                'use_hybrid': False,
                'error': str(e)
            }
