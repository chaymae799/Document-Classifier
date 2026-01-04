"""
One-shot inference script for the hybrid CV+gabarits model.
- Handles image files directly.
- For PDFs, renders the first page to an image (requires PyMuPDF/fitz).
- Outputs JSON with class scores and top prediction.

Usage:
    python backend/final_inference.py --input uploads/sample.pdf \
        --model-path backend/models/cv/hybrid_resnet50.pth \
        --output results/example_output.json --device cpu
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from backend.train_hybrid import (
    GabaritsExtractor,
    HybridResNet50,
    CLASSES,
)


def load_model(model_path: str, device: torch.device) -> nn.Module:
    model = HybridResNet50(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def render_pdf_first_page(pdf_path: Path) -> Image.Image:
    if fitz is None:
        raise RuntimeError("PyMuPDF/fitz not available to render PDFs")
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def load_image(input_path: Path) -> Image.Image:
    if input_path.suffix.lower() == ".pdf":
        return render_pdf_first_page(input_path)
    return Image.open(input_path).convert("RGB")


def build_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def infer(model: nn.Module, extractor: GabaritsExtractor, img: Image.Image, device: torch.device, gabarit_source: Path):
    tform = build_transforms()
    img_tensor = tform(img).unsqueeze(0).to(device)
    gab = extractor.extract(gabarit_source)
    gab_tensor = torch.from_numpy(gab).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor, gab_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        best_idx = int(probs.argmax())
    return probs, best_idx


def main():
    parser = argparse.ArgumentParser(description="Hybrid model inference")
    parser.add_argument("--input", required=True, help="Path to image or PDF")
    parser.add_argument("--model-path", default="backend/models/cv/hybrid_resnet50.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Optional path to save JSON result")
    args = parser.parse_args()

    device = torch.device(args.device)
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Load
    model = load_model(args.model_path, device)
    extractor = GabaritsExtractor()

    # Image + gabarits source path
    temp_path = None
    if input_path.suffix.lower() == ".pdf":
        # Render first page and save to temp PNG for gabarit extraction
        img = load_image(input_path)
        temp_path = Path("results/_tmp_render.png")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(temp_path)
        gabarit_source = temp_path
    else:
        img = load_image(input_path)
        gabarit_source = input_path

    probs, best_idx = infer(model, extractor, img, device, gabarit_source)
    scores = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
    result = {
        "input": str(input_path),
        "prediction": CLASSES[best_idx],
        "confidence": float(probs[best_idx]),
        "scores": scores,
        "device": args.device,
    }

    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    # Cleanup temp file if created
    if temp_path and temp_path.exists():
        try:
            temp_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
