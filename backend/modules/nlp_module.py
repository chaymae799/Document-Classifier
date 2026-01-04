"""
MODULE NLP - Traitement du Langage Naturel
Analyse sémantique et classification textuelle
Projet: Classification de Documents - INDIA-S5
"""

import re
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class AdvancedNLPModule:
    """
    Module NLP avancé combinant:
    - Analyse par motifs sémantiques
    - Embeddings XLM-RoBERTa (multilingual: French + Arabic)
    - Détection d'entités nommées
    - Scoring pondéré intelligent
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Dictionnaires de mots-clés enrichis et pondérés (3 classes actives)
        self.keywords = {
            'releve_bancaire': {
                'obligatoires': ['compte', 'banque', 'solde'],
                'importants': ['débit', 'crédit', 'opération', 'rib', 'iban'],
                'secondaires': ['virement', 'date', 'montant', 'libellé', 'agence', 
                               'attijariwafa', 'bmce', 'bmci', 'cih', 'populaire', 'agricole'],
                'exclusions': ['kwh', 'eau', 'salaire'],
                'poids': {'obligatoires': 3.0, 'importants': 2.0, 'secondaires': 1.0, 'exclusions': -2.0}
            },
            'facture_electricite': {
                'obligatoires': ['électricité', 'kwh', 'consommation'],
                'importants': ['one', 'radem', 'lydec', 'redal', 'puissance', 'compteur'],
                'secondaires': ['abonnement', 'watt', 'ampère', 'index', 'période', 
                               'facture', 'ttc', 'hva', 'basse', 'tension'],
                'exclusions': ['eau', 'compte', 'salaire'],
                'poids': {'obligatoires': 3.0, 'importants': 2.5, 'secondaires': 1.0, 'exclusions': -2.0}
            },
            'facture_eau': {
                'obligatoires': ['eau', 'm3', 'consommation'],
                'importants': ['radem', 'lydec', 'redal', 'régie', 'compteur'],
                'secondaires': ['index', 'période', 'facture', 'ttc', 'litre', 
                               'mètre', 'cube', 'assainissement'],
                'exclusions': ['électricité', 'kwh', 'compte', 'salaire'],
                'poids': {'obligatoires': 3.0, 'importants': 2.5, 'secondaires': 1.0, 'exclusions': -2.0}
            }
        }
        
        # Patterns regex pour entités nommées
        self.entity_patterns = {
            'montant_dh': r'\d+[.,\s]?\d*\s*(dh|mad|dirham|dirhams)',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'numero_compte': r'\b\d{10,24}\b',
            'rib': r'\b\d{24}\b',
            'numero_identite': r'\b[A-Z]{1,2}\d{5,8}\b',
            'kwh': r'\d+[.,]?\d*\s*kwh',
            'm3': r'\d+[.,]?\d*\s*m[³3]',
            'cnss': r'\b\d{8,10}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'telephone': r'\b0[567]\d{8}\b'
        }
        
        # Charger CamemBERT - TEMPORAIREMENT DÉSACTIVÉ POUR TESTS
        self.camembert_model = None
        self.tokenizer = None
        # self._load_camembert()  # Désactivé temporairement
        
        logger.info("✓ Module NLP initialisé (sans mBERT - mode keywords only)")
    
    def _load_camembert(self):
        """Charge le modèle mBERT (multilingual: French + Arabic + 100+ languages)"""
        try:
            logger.info("Chargement de mBERT (multilingual, 110 languages)...")
            # Using mBERT instead of XLM-R for faster download (~420MB vs 1.1GB)
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.camembert_model = AutoModel.from_pretrained('bert-base-multilingual-cased')
            self.camembert_model.to(self.device)
            self.camembert_model.eval()
            logger.info("✓ CamemBERT chargé avec succès")
        except Exception as e:
            logger.warning(f"Impossible de charger CamemBERT: {e}")
            self.camembert_model = None
            self.tokenizer = None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extraction d'entités nommées avec regex
        
        Args:
            text: Texte source
        
        Returns:
            Dictionnaire d'entités par type
        """
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))[:5]  # Limiter à 5 occurrences uniques
        
        return entities
    
    def analyze_keywords(self, text: str) -> Dict:
        """
        Analyse complète par mots-clés avec scoring pondéré
        
        Args:
            text: Texte à analyser
        
        Returns:
            Scores par catégorie et mots-clés trouvés
        """
        # Normaliser le texte
        text_lower = text.lower()
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        word_counter = Counter(re.findall(r'\b\w+\b', text_lower))
        
        category_scores = {}
        category_matches = {}
        
        for category, keywords_config in self.keywords.items():
            score = 0.0
            matched = []
            
            poids = keywords_config['poids']
            
            # 1. Mots-clés obligatoires (poids x3)
            obligatoires_found = 0
            for keyword in keywords_config['obligatoires']:
                if keyword in text_lower:
                    freq = word_counter.get(keyword, 0)
                    score += poids['obligatoires'] * (1 + np.log1p(freq) * 0.5)
                    matched.append(f"✓ {keyword}")
                    obligatoires_found += 1
            
            # Pénalité importante si mots obligatoires manquants
            if obligatoires_found < len(keywords_config['obligatoires']) * 0.4:
                score *= 0.3
            
            # 2. Mots-clés importants (poids x2)
            for keyword in keywords_config['importants']:
                if keyword in text_lower:
                    freq = word_counter.get(keyword, 0)
                    score += poids['importants'] * (1 + np.log1p(freq) * 0.3)
                    matched.append(f"+ {keyword}")
            
            # 3. Mots-clés secondaires (poids x1)
            for keyword in keywords_config['secondaires']:
                if keyword in text_lower:
                    freq = word_counter.get(keyword, 0)
                    score += poids['secondaires'] * (1 + np.log1p(freq) * 0.2)
                    matched.append(keyword)
            
            # 4. Mots exclus (pénalité)
            for keyword in keywords_config['exclusions']:
                if keyword in text_lower:
                    score += poids['exclusions']
                    matched.append(f"✗ {keyword}")
            
            # Normalisation du score
            max_possible = (
                len(keywords_config['obligatoires']) * poids['obligatoires'] * 2 +
                len(keywords_config['importants']) * poids['importants'] * 1.5 +
                len(keywords_config['secondaires']) * poids['secondaires']
            )
            
            normalized_score = max(0.0, min(score / max_possible, 1.0))
            
            category_scores[category] = normalized_score
            category_matches[category] = matched[:15]  # Top 15 mots-clés
        
        best_category = max(category_scores, key=category_scores.get)
        
        return {
            'scores': category_scores,
            'best_category': best_category,
            'confidence': category_scores[best_category],
            'matched_keywords': category_matches[best_category]
        }
    
    def get_camembert_embeddings(self, text: str, max_length: int = 512) -> Optional[np.ndarray]:
        """
        Obtenir les embeddings CamemBERT d'un texte
        
        Args:
            text: Texte source
            max_length: Longueur max des tokens
        
        Returns:
            Vecteur d'embeddings ou None
        """
        if self.camembert_model is None or self.tokenizer is None:
            return None
        
        try:
            # Tokenization
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length'
            ).to(self.device)
            
            # Inférence
            with torch.no_grad():
                outputs = self.camembert_model(**inputs)
            
            # Utiliser [CLS] token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return cls_embedding[0]
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction d'embeddings: {e}")
            return None
    
    def classify_with_camembert(self, text: str) -> Optional[Dict]:
        """
        Classification basée sur la similarité des embeddings CamemBERT
        
        Args:
            text: Texte à classifier
        
        Returns:
            Scores de classification ou None
        """
        if self.camembert_model is None:
            return None
        
        # Embeddings du texte
        text_embedding = self.get_camembert_embeddings(text)
        if text_embedding is None:
            return None
        
        # Embeddings de référence par catégorie (phrases types) - 3 classes
        reference_texts = {
            'releve_bancaire': "Relevé de compte bancaire opérations débit crédit solde RIB",
            'facture_electricite': "Facture électricité consommation kWh ONE RADEM compteur",
            'facture_eau': "Facture eau consommation m3 RADEM Lydec compteur assainissement"
        }
        
        scores = {}
        for category, ref_text in reference_texts.items():
            ref_embedding = self.get_camembert_embeddings(ref_text)
            if ref_embedding is not None:
                # Similarité cosinus
                similarity = np.dot(text_embedding, ref_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(ref_embedding)
                )
                # Normaliser entre 0 et 1
                scores[category] = float((similarity + 1) / 2)
        
        best_category = max(scores, key=scores.get)
        
        return {
            'scores': scores,
            'best_category': best_category,
            'confidence': scores[best_category]
        }
    
    def analyze_document_structure(self, text: str) -> Dict:
        """
        Analyse de la structure du document via le texte
        
        Args:
            text: Texte complet du document
        
        Returns:
            Métriques structurelles
        """
        lines = text.split('\n')
        
        # Métriques de base
        total_chars = len(text)
        total_words = len(text.split())
        total_lines = len([l for l in lines if l.strip()])
        
        # Densité numérique (indicateur de tableaux/relevés)
        numbers = re.findall(r'\d+', text)
        numeric_density = len(numbers) / max(total_words, 1)
        
        # Présence de tableaux (détection de colonnes alignées)
        has_table_structure = self._detect_table_in_text(lines)
        
        # Longueur moyenne des lignes
        avg_line_length = np.mean([len(l) for l in lines if l.strip()]) if lines else 0
        
        # Sections (détection de titres/sections)
        sections = self._detect_sections(lines)
        
        return {
            'total_characters': total_chars,
            'total_words': total_words,
            'total_lines': total_lines,
            'numeric_density': float(numeric_density),
            'has_table_structure': has_table_structure,
            'avg_line_length': float(avg_line_length),
            'section_count': len(sections),
            'sections': sections[:10]
        }
    
    def _detect_table_in_text(self, lines: List[str]) -> bool:
        """Détecte la présence de structures tabulaires dans le texte"""
        # Chercher des lignes avec plusieurs nombres alignés
        table_lines = 0
        for line in lines:
            numbers = re.findall(r'\d+[.,]?\d*', line)
            if len(numbers) >= 3:  # Au moins 3 nombres sur la ligne
                table_lines += 1
        
        return table_lines >= 3  # Au moins 3 lignes avec plusieurs nombres
    
    def _detect_sections(self, lines: List[str]) -> List[str]:
        """Détecte les sections/titres dans le texte"""
        sections = []
        for line in lines:
            line = line.strip()
            # Ligne courte en majuscules ou se terminant par ':'
            if (line.isupper() and len(line.split()) <= 5) or line.endswith(':'):
                sections.append(line)
        return sections
    
    def calculate_text_quality(self, text: str) -> Dict:
        """
        Calcule des métriques de qualité du texte
        
        Args:
            text: Texte à analyser
        
        Returns:
            Scores de qualité
        """
        words = text.split()
        
        # 1. Longueur suffisante
        length_score = min(len(words) / 100, 1.0)
        
        # 2. Ratio lettres/caractères (peu de symboles parasites)
        letters = sum(c.isalpha() for c in text)
        letter_ratio = letters / max(len(text), 1)
        
        # 3. Diversité lexicale
        unique_words = len(set(words))
        diversity = unique_words / max(len(words), 1)
        
        # 4. Présence de mots français courants
        french_words = ['le', 'la', 'de', 'du', 'et', 'à', 'pour', 'dans', 'sur']
        french_presence = sum(1 for w in french_words if w in text.lower()) / len(french_words)
        
        overall_quality = (length_score + letter_ratio + diversity + french_presence) / 4
        
        return {
            'length_score': float(length_score),
            'letter_ratio': float(letter_ratio),
            'diversity': float(diversity),
            'french_presence': float(french_presence),
            'overall_quality': float(overall_quality)
        }
    
    def classify_text(self, text: str, entities: Dict) -> Dict:
        """
        Classification NLP complète combinant toutes les méthodes
        
        Args:
            text: Texte à classifier
            entities: Entités extraites
        
        Returns:
            Résultat de classification complet
        """
        logger.info("Classification NLP en cours...")
        
        # 1. Analyse par mots-clés
        keyword_analysis = self.analyze_keywords(text)
        
        # 2. Classification CamemBERT (si disponible)
        camembert_analysis = self.classify_with_camembert(text)
        
        # 3. Analyse structurelle
        structure_analysis = self.analyze_document_structure(text)
        
        # 4. Qualité du texte
        quality_metrics = self.calculate_text_quality(text)
        
        # 5. Bonus basés sur les entités
        entity_bonus = self._calculate_entity_bonus(keyword_analysis['best_category'], entities)
        
        # Fusion des scores
        final_scores = {}
        for category in self.keywords.keys():
            score = keyword_analysis['scores'][category]
            
            # Ajouter le score CamemBERT si disponible
            if camembert_analysis:
                score = 0.6 * score + 0.4 * camembert_analysis['scores'][category]
            
            # Ajouter le bonus d'entités
            score += entity_bonus.get(category, 0.0)
            
            # Pénalité si qualité de texte faible
            score *= (0.5 + 0.5 * quality_metrics['overall_quality'])
            
            final_scores[category] = min(score, 1.0)
        
        best_category = max(final_scores, key=final_scores.get)
        
        return {
            'class': best_category,
            'confidence': final_scores[best_category],
            'all_scores': final_scores,
            'matched_keywords': keyword_analysis['matched_keywords'],
            'entities': entities,
            'structure': structure_analysis,
            'quality': quality_metrics,
            'camembert_available': camembert_analysis is not None
        }
    
    def _calculate_entity_bonus(self, category: str, entities: Dict) -> Dict:
        """Calcule des bonus basés sur les entités détectées (3 classes)"""
        bonus = {cat: 0.0 for cat in self.keywords.keys()}
        
        # Bonus pour relevé bancaire
        if 'numero_compte' in entities or 'rib' in entities:
            bonus['releve_bancaire'] += 0.15
        
        # Bonus pour factures électricité
        if 'kwh' in entities:
            bonus['facture_electricite'] += 0.15
        
        # Bonus pour factures eau
        if 'm3' in entities:
            bonus['facture_eau'] += 0.15
        
        return bonus