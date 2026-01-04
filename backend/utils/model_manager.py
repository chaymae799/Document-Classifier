"""
Gestionnaire de modèles pour chargement offline et cache
"""

import os
import pickle
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """Gestionnaire centralisé des modèles ML"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Args:
            models_dir: Dossier racine des modèles
        """
        self.models_dir = Path(models_dir)
        self.cache = {}
        self.configs = {}
        
        # Vérifier que le dossier existe
        if not self.models_dir.exists():
            logger.warning(f"Dossier modèles n'existe pas: {self.models_dir}")
            self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelManager initialisé: {self.models_dir}")
    
    def load_cv_model(self, model_name: str = "resnet50", 
                     device: str = "cpu") -> Optional[torch.nn.Module]:
        """
        Charge un modèle Computer Vision
        
        Args:
            model_name: Nom du modèle (resnet50, efficientnet)
            device: Device PyTorch (cpu, cuda)
            
        Returns:
            Modèle PyTorch chargé
        """
        cache_key = f"cv_{model_name}_{device}"
        
        # Vérifier le cache
        if cache_key in self.cache:
            logger.info(f"Modèle CV chargé depuis cache: {model_name}")
            return self.cache[cache_key]
        
        try:
            import torchvision.models as models
            
            model_path = self.models_dir / "cv" / f"{model_name}_pretrained.pth"
            
            # Charger le modèle
            if model_name == "resnet50":
                model = models.resnet50(pretrained=False)
                if model_path.exists():
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    logger.info(f"✓ ResNet50 chargé depuis {model_path}")
                else:
                    logger.warning(f"Fichier non trouvé: {model_path}, utilisation modèle aléatoire")
            
            elif model_name == "efficientnet":
                model = models.efficientnet_b0(pretrained=False)
                if model_path.exists():
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    logger.info(f"✓ EfficientNet chargé depuis {model_path}")
                else:
                    logger.warning(f"Fichier non trouvé: {model_path}, utilisation modèle aléatoire")
            
            else:
                logger.error(f"Modèle inconnu: {model_name}")
                return None
            
            # Mettre en mode évaluation
            model.eval()
            model.to(device)
            
            # Mettre en cache
            self.cache[cache_key] = model
            
            return model
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle CV: {str(e)}")
            return None
    
    def load_nlp_model(self, model_name: str = "camembert") -> Optional[Any]:
        """
        Charge un modèle NLP
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Tuple (model, tokenizer)
        """
        cache_key = f"nlp_{model_name}"
        
        # Vérifier le cache
        if cache_key in self.cache:
            logger.info(f"Modèle NLP chargé depuis cache: {model_name}")
            return self.cache[cache_key]
        
        try:
            from transformers import CamembertModel, CamembertTokenizer
            
            model_path = self.models_dir / "nlp" / model_name
            
            if model_path.exists():
                # Charger depuis le dossier local
                model = CamembertModel.from_pretrained(str(model_path))
                tokenizer = CamembertTokenizer.from_pretrained(str(model_path))
                logger.info(f"✓ CamemBERT chargé depuis {model_path}")
            else:
                logger.warning(f"Modèle non trouvé: {model_path}")
                logger.info("Téléchargement depuis HuggingFace...")
                model = CamembertModel.from_pretrained('camembert-base')
                tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
                
                # Sauvegarder pour usage offline futur
                model_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(model_path))
                tokenizer.save_pretrained(str(model_path))
                logger.info(f"✓ Modèle sauvegardé dans {model_path}")
            
            model.eval()
            
            # Mettre en cache
            self.cache[cache_key] = (model, tokenizer)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle NLP: {str(e)}")
            return None
    
    def load_keywords_dict(self) -> Dict[str, list]:
        """
        Charge le dictionnaire de mots-clés
        
        Returns:
            Dict des mots-clés par catégorie
        """
        cache_key = "keywords_dict"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            keywords_path = self.models_dir / "nlp" / "keywords.pkl"
            
            if keywords_path.exists():
                with open(keywords_path, 'rb') as f:
                    keywords = pickle.load(f)
                logger.info(f"✓ Dictionnaire de mots-clés chargé")
            else:
                # Dictionnaire par défaut
                keywords = {
                    'piece_identite': [
                        'carte nationale', 'CNIE', 'identité', 'né(e) le', 
                        'nationalité', 'date de naissance', 'royaume du maroc'
                    ],
                    'releve_bancaire': [
                        'solde', 'débit', 'crédit', 'compte', 'banque', 
                        'opération', 'IBAN', 'RIB', 'virement'
                    ],
                    'facture_electricite': [
                        'kWh', 'électricité', 'ONE', 'RADEM', 'Lydec',
                        'consommation', 'puissance', 'compteur'
                    ],
                    'facture_eau': [
                        'm³', 'eau', 'ONEE', 'Lydec', 'REDAL',
                        'consommation', 'assainissement'
                    ],
                    'document_employeur': [
                        'salaire', 'bulletin', 'paie', 'employeur', 
                        'cotisations', 'CNSS', 'net à payer'
                    ]
                }
                
                # Sauvegarder
                keywords_path.parent.mkdir(parents=True, exist_ok=True)
                with open(keywords_path, 'wb') as f:
                    pickle.dump(keywords, f)
                
                logger.info(f"✓ Dictionnaire par défaut créé: {keywords_path}")
            
            self.cache[cache_key] = keywords
            return keywords
            
        except Exception as e:
            logger.error(f"Erreur chargement dictionnaire: {str(e)}")
            return {}
    
    def load_gabarit_templates(self) -> Dict[str, Any]:
        """
        Charge les templates de gabarits
        
        Returns:
            Dict des templates par catégorie
        """
        cache_key = "gabarit_templates"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            templates_path = self.models_dir / "gabarits" / "templates.pkl"
            
            if templates_path.exists():
                with open(templates_path, 'rb') as f:
                    templates = pickle.load(f)
                logger.info(f"✓ Templates de gabarits chargés")
            else:
                # Templates par défaut
                templates = {
                    'piece_identite': {
                        'aspect_ratio': (1.58, 1.62),  # Format carte ID-1
                        'has_photo': True,
                        'orientation': 'landscape',
                        'color_dominance': 'varied'
                    },
                    'releve_bancaire': {
                        'has_table': True,
                        'min_rows': 5,
                        'orientation': 'portrait',
                        'color_dominance': 'white'
                    },
                    'facture_electricite': {
                        'has_table': True,
                        'has_logo': True,
                        'orientation': 'portrait',
                        'color_dominance': 'white'
                    },
                    'facture_eau': {
                        'has_table': True,
                        'has_logo': True,
                        'orientation': 'portrait',
                        'color_dominance': 'white'
                    },
                    'document_employeur': {
                        'has_header': True,
                        'has_table': True,
                        'orientation': 'portrait',
                        'color_dominance': 'white'
                    }
                }
                
                # Sauvegarder
                templates_path.parent.mkdir(parents=True, exist_ok=True)
                with open(templates_path, 'wb') as f:
                    pickle.dump(templates, f)
                
                logger.info(f"✓ Templates par défaut créés: {templates_path}")
            
            self.cache[cache_key] = templates
            return templates
            
        except Exception as e:
            logger.error(f"Erreur chargement templates: {str(e)}")
            return {}
    
    def save_model(self, model: Any, model_type: str, model_name: str):
        """
        Sauvegarde un modèle
        
        Args:
            model: Modèle à sauvegarder
            model_type: Type (cv, nlp, etc.)
            model_name: Nom du fichier
        """
        try:
            save_dir = self.models_dir / model_type
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = save_dir / f"{model_name}.pth"
            
            if isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), save_path)
            else:
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info(f"✓ Modèle sauvegardé: {save_path}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle: {str(e)}")
    
    def clear_cache(self):
        """Vide le cache des modèles"""
        self.cache.clear()
        logger.info("Cache vidé")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Retourne des infos sur le cache
        
        Returns:
            Dict avec statistiques du cache
        """
        info = {
            'cached_models': list(self.cache.keys()),
            'cache_size': len(self.cache),
            'models_directory': str(self.models_dir)
        }
        return info
    
    def check_models_availability(self) -> Dict[str, bool]:
        """
        Vérifie la disponibilité des modèles
        
        Returns:
            Dict avec statut de chaque modèle
        """
        availability = {}
        
        # CV models
        cv_dir = self.models_dir / "cv"
        availability['resnet50'] = (cv_dir / "resnet50_pretrained.pth").exists()
        availability['efficientnet'] = (cv_dir / "efficientnet_b0.pth").exists()
        
        # NLP models
        nlp_dir = self.models_dir / "nlp" / "camembert"
        availability['camembert'] = nlp_dir.exists()
        
        # Keywords
        availability['keywords'] = (self.models_dir / "nlp" / "keywords.pkl").exists()
        
        # Gabarits
        availability['gabarits'] = (self.models_dir / "gabarits" / "templates.pkl").exists()
        
        return availability


# Instance globale singleton
_model_manager_instance = None

def get_model_manager(models_dir: str = "models") -> ModelManager:
    """
    Retourne l'instance singleton du ModelManager
    
    Args:
        models_dir: Dossier des modèles
        
    Returns:
        Instance ModelManager
    """
    global _model_manager_instance
    
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager(models_dir)
    
    return _model_manager_instance