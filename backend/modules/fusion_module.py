"""
MODULE FUSION MULTIMODALE - Combinaison Intelligente
Fusionne les prédictions CV + NLP + Gabarits
Projet: Classification de Documents - INDIA-S5
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MultimodalFusionModule:
    """
    Fusion intelligente des prédictions des 3 modules:
    - Computer Vision (ResNet50 + EfficientNet)
    - NLP (OCR + CamemBERT + Mots-clés)
    - Gabarits (Patterns structurels)
    
    Utilise des règles métier et des pondérations adaptatives
    """
    
    def __init__(self):
        # Classes supportées
        self.classes = [
            'piece_identite',
            'releve_bancaire',
            'facture_electricite',
            'facture_eau',
            'document_employeur'
        ]
        
        # Poids par défaut pour chaque module
        self.default_weights = {
            'cv': 0.40,
            'nlp': 0.35,
            'gabarits': 0.25
        }
        
        # Règles métier par catégorie
        self.business_rules = {
            'piece_identite': {
                'required_gabarit_features': ['has_photo'],
                'min_gabarit_score': 0.3,
                'keywords_required': ['identité', 'carte'],
                'cv_confidence_boost': 1.1
            },
            'releve_bancaire': {
                'required_gabarit_features': ['has_table'],
                'min_gabarit_score': 0.4,
                'keywords_required': ['compte', 'banque'],
                'nlp_confidence_boost': 1.1
            },
            'facture_electricite': {
                'required_gabarit_features': ['has_table'],
                'min_gabarit_score': 0.3,
                'keywords_required': ['électricité', 'kwh'],
                'nlp_confidence_boost': 1.15
            },
            'facture_eau': {
                'required_gabarit_features': ['has_table'],
                'min_gabarit_score': 0.3,
                'keywords_required': ['eau', 'm3'],
                'nlp_confidence_boost': 1.15
            },
            'document_employeur': {
                'required_gabarit_features': [],
                'min_gabarit_score': 0.2,
                'keywords_required': ['salaire', 'employeur'],
                'nlp_confidence_boost': 1.2
            }
        }
        
        logger.info("✓ Module Fusion Multimodale initialisé")
    
    def fuse(
        self,
        cv_result: Dict,
        nlp_result: Dict,
        gabarit_result: Dict,
        image_path: str = ""
    ) -> Dict:
        """
        Fusion complète des 3 modules
        
        Args:
            cv_result: Résultat du module CV
            nlp_result: Résultat du module NLP
            gabarit_result: Résultat du module Gabarits
            image_path: Chemin de l'image (pour logs)
        
        Returns:
            Décision finale avec tous les détails
        """
        logger.info(f"Fusion multimodale pour: {image_path}")
        
        # Récupérer les scores de chaque module
        cv_scores = cv_result.get('all_scores', {})
        nlp_scores = nlp_result.get('all_scores', {})
        gabarit_scores = gabarit_result.get('scores', {})
        
        # Ajuster les poids selon le contexte
        weights = self._adaptive_weighting(cv_result, nlp_result, gabarit_result)
        
        # Calcul des scores combinés
        combined_scores = {}
        for cls in self.classes:
            cv_score = cv_scores.get(cls, 0.0)
            nlp_score = nlp_scores.get(cls, 0.0)
            gabarit_score = gabarit_scores.get(cls, 0.0)
            
            # Pondération de base
            combined = (
                weights['cv'] * cv_score +
                weights['nlp'] * nlp_score +
                weights['gabarits'] * gabarit_score
            )
            
            combined_scores[cls] = combined
        
        # Décision initiale
        best_class = max(combined_scores, key=combined_scores.get)
        base_confidence = combined_scores[best_class]
        
        # Application des règles métier
        best_class, final_confidence, validation = self._apply_business_rules(
            best_class,
            base_confidence,
            cv_result,
            nlp_result,
            gabarit_result
        )
        
        # Vérification de cohérence
        coherence = self._check_coherence(cv_result, nlp_result, gabarit_result, best_class)
        
        # Ajustement final de la confiance
        final_confidence = self._adjust_confidence(
            final_confidence,
            coherence,
            validation
        )
        
        logger.info(f"  ✓ Fusion: {best_class} ({final_confidence:.3f})")
        
        return {
            'classification': best_class,
            'confidence_cv': cv_result.get('confidence', 0.0),
            'confidence_nlp': nlp_result.get('confidence', 0.0),
            'confidence_gabarits': gabarit_scores.get(best_class, 0.0),
            'confidence_globale': float(final_confidence),
            'weights_used': weights,
            'all_scores': combined_scores,
            'coherence': coherence,
            'validation': validation,
            'details': {
                'cv': {
                    'prediction': cv_result.get('class'),
                    'confidence': cv_result.get('confidence', 0.0)
                },
                'nlp': {
                    'prediction': nlp_result.get('class'),
                    'confidence': nlp_result.get('confidence', 0.0),
                    'keywords': nlp_result.get('matched_keywords', [])[:10]
                },
                'gabarits': {
                    'top_features': self._get_top_features(gabarit_result)
                }
            }
        }
    
    def _adaptive_weighting(
        self,
        cv_result: Dict,
        nlp_result: Dict,
        gabarit_result: Dict
    ) -> Dict:
        """
        Ajustement adaptatif des poids selon les confidences
        
        Args:
            cv_result, nlp_result, gabarit_result: Résultats des modules
        
        Returns:
            Poids ajustés
        """
        weights = self.default_weights.copy()
        
        cv_conf = cv_result.get('confidence', 0.0)
        nlp_conf = nlp_result.get('confidence', 0.0)
        
        # Si un module est très confiant, augmenter son poids
        if cv_conf > 0.9:
            weights['cv'] = 0.50
            weights['nlp'] = 0.30
            weights['gabarits'] = 0.20
        elif nlp_conf > 0.9:
            weights['cv'] = 0.30
            weights['nlp'] = 0.50
            weights['gabarits'] = 0.20
        
        # Si qualité OCR faible, réduire le poids NLP
        nlp_quality = nlp_result.get('quality', {}).get('overall_quality', 1.0)
        if nlp_quality < 0.5:
            weights['cv'] += 0.15
            weights['nlp'] -= 0.10
            weights['gabarits'] -= 0.05
        
        # If gabarit indicates a table (likely invoice), boost gabarits weight
        try:
            features = gabarit_result.get('features', {})
            if features.get('has_table'):
                weights['gabarits'] += 0.15
                # slightly reduce CV weight to compensate
                weights['cv'] -= 0.05
        except Exception:
            pass
        # Normaliser pour que la somme = 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _apply_business_rules(
        self,
        predicted_class: str,
        confidence: float,
        cv_result: Dict,
        nlp_result: Dict,
        gabarit_result: Dict
    ) -> Tuple[str, float, Dict]:
        """
        Application des règles métier spécifiques
        
        Args:
            predicted_class: Classe prédite
            confidence: Confiance de base
            cv_result, nlp_result, gabarit_result: Résultats des modules
        
        Returns:
            (classe finale, confiance ajustée, validation)
        """
        validation = {
            'rules_met': [],
            'rules_violated': [],
            'is_valid': True
        }
        
        if predicted_class not in self.business_rules:
            return predicted_class, confidence, validation
        
        rules = self.business_rules[predicted_class]
        features = gabarit_result.get('features', {})
        nlp_keywords = nlp_result.get('matched_keywords', [])
        
        # 1. Vérifier features gabarits requises
        for required_feature in rules.get('required_gabarit_features', []):
            if features.get(required_feature, False):
                validation['rules_met'].append(f"Feature requise: {required_feature}")
                confidence *= 1.05
            else:
                validation['rules_violated'].append(f"Feature manquante: {required_feature}")
                confidence *= 0.85
                validation['is_valid'] = False
        
        # 2. Vérifier score minimal gabarits
        min_gabarit = rules.get('min_gabarit_score', 0.0)
        gabarit_score = gabarit_result.get('scores', {}).get(predicted_class, 0.0)
        
        if gabarit_score >= min_gabarit:
            validation['rules_met'].append(f"Score gabarits OK: {gabarit_score:.2f}")
        else:
            validation['rules_violated'].append(
                f"Score gabarits insuffisant: {gabarit_score:.2f} < {min_gabarit}"
            )
            confidence *= 0.9
        
        # 3. Vérifier mots-clés requis
        keywords_found = []
        for keyword in rules.get('keywords_required', []):
            if any(keyword in str(kw).lower() for kw in nlp_keywords):
                keywords_found.append(keyword)
        
        if keywords_found:
            validation['rules_met'].append(f"Mots-clés: {', '.join(keywords_found)}")
        else:
            validation['rules_violated'].append("Mots-clés requis manquants")
            confidence *= 0.85
        
        # 4. Boost de confiance selon le module le plus pertinent
        if 'cv_confidence_boost' in rules and cv_result.get('confidence', 0) > 0.8:
            confidence *= rules['cv_confidence_boost']
            validation['rules_met'].append("Boost CV appliqué")
        
        if 'nlp_confidence_boost' in rules and nlp_result.get('confidence', 0) > 0.8:
            confidence *= rules['nlp_confidence_boost']
            validation['rules_met'].append("Boost NLP appliqué")
        
        # Limiter la confiance à 1.0
        confidence = min(confidence, 1.0)
        
        return predicted_class, confidence, validation
    
    def _check_coherence(
        self,
        cv_result: Dict,
        nlp_result: Dict,
        gabarit_result: Dict,
        final_class: str
    ) -> Dict:
        """
        Vérification de la cohérence entre les modules
        
        Args:
            cv_result, nlp_result, gabarit_result: Résultats
            final_class: Classe finale prédite
        
        Returns:
            Métriques de cohérence
        """
        cv_pred = cv_result.get('class')
        nlp_pred = nlp_result.get('class')
        gabarit_scores = gabarit_result.get('scores', {})
        gabarit_pred = max(gabarit_scores, key=gabarit_scores.get) if gabarit_scores else None
        
        # Accord entre modules
        agreements = []
        if cv_pred == nlp_pred:
            agreements.append('cv_nlp')
        if cv_pred == gabarit_pred:
            agreements.append('cv_gabarits')
        if nlp_pred == gabarit_pred:
            agreements.append('nlp_gabarits')
        
        # Tous d'accord ?
        perfect_agreement = (cv_pred == nlp_pred == gabarit_pred)
        
        # Majorité ?
        predictions = [cv_pred, nlp_pred, gabarit_pred]
        majority = predictions.count(final_class) >= 2
        
        # Score de cohérence global
        coherence_score = len(agreements) / 3.0
        if perfect_agreement:
            coherence_score = 1.0
        
        return {
            'perfect_agreement': perfect_agreement,
            'majority_agreement': majority,
            'agreements': agreements,
            'coherence_score': float(coherence_score),
            'predictions': {
                'cv': cv_pred,
                'nlp': nlp_pred,
                'gabarits': gabarit_pred
            }
        }
    
    def _adjust_confidence(
        self,
        confidence: float,
        coherence: Dict,
        validation: Dict
    ) -> float:
        """
        Ajustement final de la confiance
        
        Args:
            confidence: Confiance de base
            coherence: Métriques de cohérence
            validation: Résultats de validation
        
        Returns:
            Confiance ajustée
        """
        # Bonus si accord parfait
        if coherence['perfect_agreement']:
            confidence *= 1.15
        elif coherence['majority_agreement']:
            confidence *= 1.05
        else:
            confidence *= 0.85
        
        # Bonus/pénalité selon validation
        if validation['is_valid']:
            confidence *= 1.05
        else:
            penalty = 1.0 - (len(validation['rules_violated']) * 0.05)
            confidence *= max(penalty, 0.7)
        
        # Limiter entre 0 et 1
        return max(0.0, min(confidence, 1.0))
    
    def _get_top_features(self, gabarit_result: Dict) -> List[str]:
        """
        Extrait les principales features détectées
        
        Args:
            gabarit_result: Résultat gabarits
        
        Returns:
            Liste des features principales
        """
        features = gabarit_result.get('features', {})
        top_features = []
        
        if features.get('has_photo'):
            top_features.append('photo détectée')
        if features.get('has_table'):
            top_features.append('tableau détecté')
        if features.get('has_logo'):
            top_features.append('logo présent')
        if features.get('has_signature'):
            top_features.append('signature détectée')
        if features.get('has_graphs'):
            top_features.append('graphiques présents')
        
        text_density = features.get('text_density', 0)
        if text_density > 0.6:
            top_features.append('densité texte élevée')
        elif text_density < 0.3:
            top_features.append('densité texte faible')
        
        return top_features[:5]
    
    def should_reject(self, result: Dict, threshold: float = 0.50) -> bool:
        """
        Détermine si la prédiction doit être rejetée (confiance trop faible)
        
        Args:
            result: Résultat de fusion
            threshold: Seuil de rejet (default 0.50 = 50%)
        
        Returns:
            True si doit être rejeté
        """
        confidence = result.get('confidence_globale', 0.0)
        coherence_score = result.get('coherence', {}).get('coherence_score', 0.0)
        is_valid = result.get('validation', {}).get('is_valid', False)
        
        # Rejet si confiance trop faible
        if confidence < threshold:
            return True
        
        # Rejet si aucune cohérence entre modules (moins strict)
        if coherence_score < 0.1:
            return True
        
        # Rejet si règles métier violées et confiance PAS TRÈS élevée (assouplir à 0.6)
        if not is_valid and confidence < 0.6:
            return True
        
        return False
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Catégorise le niveau de confiance
        
        Args:
            confidence: Score de confiance
        
        Returns:
            Niveau: 'très élevée', 'élevée', 'moyenne', 'faible'
        """
        if confidence >= 0.9:
            return 'très élevée'
        elif confidence >= 0.75:
            return 'élevée'
        elif confidence >= 0.6:
            return 'moyenne'
        else:
            return 'faible'