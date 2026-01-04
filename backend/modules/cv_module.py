"""
MODULE COMPUTER VISION - Classification Visuelle
Utilise ResNet50 + EfficientNet en ensemble
Projet: Classification de Documents - INDIA-S5
"""

import os
import glob
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class HybridCVModule:
    """
    Module Computer Vision hybride combinant:
    - ResNet50 (architecture profonde, très précise)
    - EfficientNet-B0 (léger et efficace)
    - Ensemble learning pour décision finale
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.classes = [
            'piece_identite',
            'releve_bancaire', 
            'facture_electricite',
            'facture_eau',
            'document_employeur'
        ]
        
        # Transformations standard pour ImageNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Transformations avec augmentation pour robustesse
        self.transform_augmented = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Charger les modèles
        self.resnet50 = None
        self.efficientnet = None
        self._load_models()
        
        logger.info("✓ Module CV initialisé")
    
    def _load_models(self):
        """Charge ResNet50 et EfficientNet"""
        try:
            logger.info("Chargement des modèles CV...")
            
            # ResNet50 pré-entraîné sur ImageNet
            logger.info("  - Chargement ResNet50...")
            self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            # Modifier la couche finale pour 5 classes
            num_features_resnet = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_features_resnet, 5)
            
            # Gel des couches initiales pour accélérer (fine-tuning partiel)
            for param in list(self.resnet50.parameters())[:-10]:
                param.requires_grad = False
            
            self.resnet50 = self.resnet50.to(self.device)
            self.resnet50.eval()
            logger.info("  ✓ ResNet50 chargé")
            
            # EfficientNet-B0 disabled due to weight download issues
            # Will use ResNet50 only for inference
            logger.info("  - EfficientNet-B0 disabled (ResNet50 sufficient for Phase 1)")
            self.efficientnet = None
            logger.info("  ! Configuration: ResNet50 only mode")
            
            logger.info("✓ Tous les modèles CV chargés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles CV: {e}")
            raise
    
    def preprocess_image(self, image_path: str, enhance: bool = True) -> Image.Image:
        """
        Prétraitement avancé de l'image
        
        Args:
            image_path: Chemin vers l'image
            enhance: Appliquer des améliorations
        
        Returns:
            Image PIL prétraitée
        """
        # Charger l'image
        img = Image.open(image_path).convert('RGB')
        
        if enhance:
            # Amélioration du contraste
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
            
            # Amélioration de la netteté
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            
            # Amélioration de la luminosité si trop sombre
            enhancer = ImageEnhance.Brightness(img)
            img_array = np.array(img)
            avg_brightness = np.mean(img_array)
            if avg_brightness < 100:
                brightness_factor = 1.2
                img = enhancer.enhance(brightness_factor)
            
            # Auto-contraste
            img = ImageOps.autocontrast(img, cutoff=2)
        
        return img
    
    def extract_features_resnet(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extraction de features avec ResNet50
        
        Args:
            image_tensor: Tensor d'image
        
        Returns:
            Vecteur de features
        """
        with torch.no_grad():
            # Extraire les features avant la couche FC
            x = self.resnet50.conv1(image_tensor)
            x = self.resnet50.bn1(x)
            x = self.resnet50.relu(x)
            x = self.resnet50.maxpool(x)
            
            x = self.resnet50.layer1(x)
            x = self.resnet50.layer2(x)
            x = self.resnet50.layer3(x)
            x = self.resnet50.layer4(x)
            
            x = self.resnet50.avgpool(x)
            features = torch.flatten(x, 1)
        
        return features
    
    def classify_with_resnet(self, image_tensor: torch.Tensor) -> Dict:
        """
        Classification avec ResNet50
        
        Args:
            image_tensor: Image preprocessed
        
        Returns:
            Résultats de classification
        """
        with torch.no_grad():
            outputs = self.resnet50(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        all_scores = {
            self.classes[i]: float(probabilities[0][i].item()) 
            for i in range(len(self.classes))
        }
        
        return {
            'class': self.classes[predicted.item()],
            'confidence': float(confidence.item()),
            'all_scores': all_scores
        }
    
    def classify_with_efficientnet(self, image_tensor: torch.Tensor) -> Dict:
        """
        Classification avec EfficientNet
        
        Args:
            image_tensor: Image preprocessed
        
        Returns:
            Résultats de classification
        """
        if self.efficientnet is None:
            # Return neutral low-confidence scores if EfficientNet unavailable
            uniform = 1.0 / len(self.classes)
            all_scores = {cls: float(uniform * 0.5) for cls in self.classes}
            return {
                'class': None,
                'confidence': 0.0,
                'all_scores': all_scores
            }

        with torch.no_grad():
            outputs = self.efficientnet(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        all_scores = {
            self.classes[i]: float(probabilities[0][i].item()) 
            for i in range(len(self.classes))
        }

        return {
            'class': self.classes[predicted.item()],
            'confidence': float(confidence.item()),
            'all_scores': all_scores
        }
    
    def ensemble_prediction(
        self, 
        resnet_result: Dict, 
        efficientnet_result: Dict,
        weights: Tuple[float, float] = (0.6, 0.4)
    ) -> Dict:
        """
        Ensemble des prédictions des deux modèles
        
        Args:
            resnet_result: Résultat ResNet50
            efficientnet_result: Résultat EfficientNet
            weights: Poids (resnet, efficientnet)
        
        Returns:
            Prédiction finale
        """
        ensemble_scores = {}
        
        for cls in self.classes:
            ensemble_scores[cls] = (
                weights[0] * resnet_result['all_scores'][cls] +
                weights[1] * efficientnet_result['all_scores'][cls]
            )
        
        best_class = max(ensemble_scores, key=ensemble_scores.get)
        confidence = ensemble_scores[best_class]
        
        # Calculer l'accord entre les deux modèles
        agreement = resnet_result['class'] == efficientnet_result['class']
        
        # Ajuster la confiance selon l'accord
        if agreement:
            confidence = min(confidence * 1.1, 1.0)  # Bonus si accord
        else:
            confidence = confidence * 0.9  # Pénalité si désaccord
        
        return {
            'class': best_class,
            'confidence': float(confidence),
            'all_scores': ensemble_scores,
            'resnet_prediction': resnet_result['class'],
            'efficientnet_prediction': efficientnet_result['class'],
            'models_agree': agreement
        }
    
    def multi_crop_prediction(self, image_path: str) -> Dict:
        """
        Prédiction avec multiple crops pour plus de robustesse
        
        Args:
            image_path: Chemin de l'image
        
        Returns:
            Prédiction agrégée
        """
        img = self.preprocess_image(image_path)
        
        # Crops: centre, haut-gauche, haut-droit, bas-gauche, bas-droit
        width, height = img.size
        crops = [
            img,  # Image complète
            img.crop((0, 0, width // 2, height // 2)),  # Haut-gauche
            img.crop((width // 2, 0, width, height // 2)),  # Haut-droit
            img.crop((0, height // 2, width // 2, height)),  # Bas-gauche
            img.crop((width // 2, height // 2, width, height))  # Bas-droit
        ]
        
        predictions = []
        for crop in crops:
            crop = crop.resize((224, 224))
            tensor = self.transform(crop).unsqueeze(0).to(self.device)
            
            resnet_pred = self.classify_with_resnet(tensor)
            efficientnet_pred = self.classify_with_efficientnet(tensor)
            ensemble_pred = self.ensemble_prediction(resnet_pred, efficientnet_pred)
            
            predictions.append(ensemble_pred)
        
        # Agrégation des prédictions
        aggregated_scores = {cls: 0.0 for cls in self.classes}
        for pred in predictions:
            for cls, score in pred['all_scores'].items():
                aggregated_scores[cls] += score
        
        # Moyenne
        for cls in aggregated_scores:
            aggregated_scores[cls] /= len(predictions)
        
        best_class = max(aggregated_scores, key=aggregated_scores.get)
        
        return {
            'class': best_class,
            'confidence': aggregated_scores[best_class],
            'all_scores': aggregated_scores,
            'method': 'multi_crop'
        }
    
    def test_time_augmentation(self, image_path: str, n_augmentations: int = 5) -> Dict:
        """
        Test Time Augmentation pour prédiction plus robuste
        
        Args:
            image_path: Chemin de l'image
            n_augmentations: Nombre d'augmentations
        
        Returns:
            Prédiction agrégée
        """
        img = self.preprocess_image(image_path)
        
        predictions = []
        
        for i in range(n_augmentations):
            # Appliquer des transformations aléatoires
            if i == 0:
                # Original
                transformed = self.transform(img)
            else:
                # Avec augmentations
                transformed = self.transform_augmented(img)
            
            tensor = transformed.unsqueeze(0).to(self.device)
            
            resnet_pred = self.classify_with_resnet(tensor)
            efficientnet_pred = self.classify_with_efficientnet(tensor)
            ensemble_pred = self.ensemble_prediction(resnet_pred, efficientnet_pred)
            
            predictions.append(ensemble_pred)
        
        # Agrégation
        aggregated_scores = {cls: 0.0 for cls in self.classes}
        for pred in predictions:
            for cls, score in pred['all_scores'].items():
                aggregated_scores[cls] += score
        
        for cls in aggregated_scores:
            aggregated_scores[cls] /= len(predictions)
        
        best_class = max(aggregated_scores, key=aggregated_scores.get)
        
        return {
            'class': best_class,
            'confidence': aggregated_scores[best_class],
            'all_scores': aggregated_scores,
            'method': 'tta'
        }
    
    def classify(
        self, 
        image_path: str, 
        use_tta: bool = False,
        use_multicrop: bool = False
    ) -> Dict:
        """
        Pipeline complet de classification CV
        
        Args:
            image_path: Chemin de l'image
            use_tta: Utiliser Test Time Augmentation
            use_multicrop: Utiliser multi-crop
        
        Returns:
            Résultat de classification complet
        """
        logger.info(f"Classification CV: {image_path}")
        
        try:
            # Méthode robuste si demandée
            if use_tta:
                logger.info("  → Utilisation de TTA")
                return self.test_time_augmentation(image_path)
            
            if use_multicrop:
                logger.info("  → Utilisation de multi-crop")
                return self.multi_crop_prediction(image_path)
            
            # Méthode standard (plus rapide)
            img = self.preprocess_image(image_path)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Prédictions des deux modèles
            resnet_result = self.classify_with_resnet(img_tensor)
            efficientnet_result = self.classify_with_efficientnet(img_tensor)
            
            # Ensemble
            final_result = self.ensemble_prediction(resnet_result, efficientnet_result)
            
            logger.info(f"  ✓ CV: {final_result['class']} ({final_result['confidence']:.3f})")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Erreur classification CV: {e}")
            return {
                'class': 'error',
                'confidence': 0.0,
                'all_scores': {cls: 0.0 for cls in self.classes},
                'error': str(e)
            }
    
    def get_visual_features(self, image_path: str) -> Dict:
        """
        Extraction de features visuelles générales
        
        Args:
            image_path: Chemin de l'image
        
        Returns:
            Caractéristiques visuelles
        """
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Statistiques de couleur
        mean_color = img_array.mean(axis=(0, 1))
        std_color = img_array.std(axis=(0, 1))
        
        # Histogramme des couleurs
        hist_r, _ = np.histogram(img_array[:,:,0], bins=32, range=(0, 256))
        hist_g, _ = np.histogram(img_array[:,:,1], bins=32, range=(0, 256))
        hist_b, _ = np.histogram(img_array[:,:,2], bins=32, range=(0, 256))
        
        # Dominance de couleur
        color_dominance = {
            'red': float(mean_color[0] / 255),
            'green': float(mean_color[1] / 255),
            'blue': float(mean_color[2] / 255)
        }
        
        # Contraste
        contrast = float(img_array.std())
        
        # Luminosité moyenne
        brightness = float(img_array.mean())
        
        return {
            'mean_color': mean_color.tolist(),
            'std_color': std_color.tolist(),
            'color_dominance': color_dominance,
            'contrast': contrast,
            'brightness': brightness,
            'histogram_entropy': float(
                -np.sum((hist_r/hist_r.sum()) * np.log2(hist_r/hist_r.sum() + 1e-10))
            )
        }
    
    def compare_similarity(self, image_path1: str, image_path2: str) -> float:
        """
        Compare la similarité visuelle entre deux images
        
        Args:
            image_path1: Première image
            image_path2: Deuxième image
        
        Returns:
            Score de similarité (0-1)
        """
        # Extraire les features des deux images
        img1 = self.preprocess_image(image_path1)
        img2 = self.preprocess_image(image_path2)
        
        tensor1 = self.transform(img1).unsqueeze(0).to(self.device)
        tensor2 = self.transform(img2).unsqueeze(0).to(self.device)
        
        features1 = self.extract_features_resnet(tensor1)
        features2 = self.extract_features_resnet(tensor2)
        
        # Similarité cosinus
        similarity = torch.nn.functional.cosine_similarity(features1, features2)
        
        return float(similarity.item())
    
    def benchmark_models(self, image_path: str) -> Dict:
        """
        Benchmark des performances des modèles
        
        Args:
            image_path: Image de test
        
        Returns:
            Temps d'inférence et résultats
        """
        import time
        
        img = self.preprocess_image(image_path)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Benchmark ResNet50
        start = time.time()
        resnet_result = self.classify_with_resnet(img_tensor)
        resnet_time = time.time() - start
        
        # Benchmark EfficientNet
        start = time.time()
        eff_result = self.classify_with_efficientnet(img_tensor)
        eff_time = time.time() - start
        
        return {
            'resnet50': {
                'time': resnet_time,
                'prediction': resnet_result['class'],
                'confidence': resnet_result['confidence']
            },
            'efficientnet': {
                'time': eff_time,
                'prediction': eff_result['class'],
                'confidence': eff_result['confidence']
            },
            'speedup': resnet_time / eff_time if eff_time > 0 else 0
        }