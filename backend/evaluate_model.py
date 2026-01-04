"""
Evaluate trained hybrid model on test/validation set.
Generates confusion matrix, per-class metrics, and comprehensive report.

Usage:
    python backend/evaluate_model.py --model-path backend/models/cv/hybrid_resnet50.pth --data-dir data
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from training script
from backend.train_hybrid import (
    DocumentDataset, GabaritsExtractor, HybridResNet50,
    CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
)


def load_model(model_path, device):
    """Load trained model."""
    model = HybridResNet50(num_classes=len(CLASSES)).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            gabarits = batch['gabarit'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, gabarits)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description='Evaluate hybrid model')
    parser.add_argument('--model-path', default='backend/models/cv/hybrid_resnet50.pth',
                       help='Path to trained model')
    parser.add_argument('--data-dir', default='data', help='Path to dataset')
    parser.add_argument('--output', default='backend/evaluation_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    
    # Load validation dataset
    gabarit_extractor = GabaritsExtractor()
    val_dataset = DocumentDataset(args.data_dir, split='val', gabarit_extractor=gabarit_extractor)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty. Skipping evaluation.")
        return
    
    # Determine classes present in the validation split
    unique_ids = sorted(set(val_dataset.labels))
    if not unique_ids:
        logger.warning("No labels found in validation set. Skipping evaluation.")
        return
    target_names = [IDX_TO_CLASS[i] for i in unique_ids]

    # Load model
    model = load_model(args.model_path, device)
    logger.info(f"Model loaded from {args.model_path}")
    
    # Evaluate
    predictions, labels = evaluate(model, val_loader, device)
    
    # Compute metrics
    cm = confusion_matrix(labels, predictions, labels=unique_ids)
    report = classification_report(labels, predictions, labels=unique_ids, target_names=target_names, output_dict=True)
    acc = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    
    logger.info(f"\nAccuracy: {acc:.4f}")
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(labels, predictions, labels=unique_ids, target_names=target_names)}")
    
    # Prepare output
    results = {
        'accuracy': float(acc),
        'balanced_accuracy': float(balanced_acc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'per_class_metrics': {}
    }

    for cls_name in target_names:
        cls_report = report.get(cls_name, {})
        results['per_class_metrics'][cls_name] = {
            'precision': float(cls_report.get('precision', 0)),
            'recall': float(cls_report.get('recall', 0)),
            'f1-score': float(cls_report.get('f1-score', 0)),
            'support': int(cls_report.get('support', 0))
        }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Optionally save confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        ax.set_xticks(range(len(target_names)))
        ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names, rotation=45)
        ax.set_yticklabels(target_names)
        
        # Add text annotations
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                text = ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        cm_path = output_path.parent / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150)
        logger.info(f"Confusion matrix saved to {cm_path}")
    except Exception as e:
        logger.warning(f"Could not save confusion matrix plot: {e}")


if __name__ == '__main__':
    main()
