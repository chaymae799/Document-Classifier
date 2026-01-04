"""
Export OCR text (.txt) and hOCR for all processed images.
Usage:
    python backend/export_ocr.py
Outputs:
    data/ocr_text/<basename>.txt
    data/ocr_text/<basename>.hocr
"""
from pathlib import Path
from PIL import Image
import pytesseract
import argparse

OCR_DIR = Path(__file__).parent.parent / 'data' / 'ocr_text'
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed_images'

OCR_LANG = 'fra'


def ensure_dirs():
    OCR_DIR.mkdir(parents=True, exist_ok=True)


def process_image(img_path: Path):
    try:
        pil = Image.open(img_path).convert('RGB')
        # txt
        text = pytesseract.image_to_string(pil, lang=OCR_LANG)
        txt_path = OCR_DIR / (img_path.stem + '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        # hOCR
        hocr = pytesseract.image_to_pdf_or_hocr(pil, extension='hocr', lang=OCR_LANG)
        hocr_path = OCR_DIR / (img_path.stem + '.hocr')
        with open(hocr_path, 'wb') as f:
            f.write(hocr)
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Export OCR text for processed images')
    parser.add_argument('--lang', default='fra', help='Tesseract language (default: fra)')
    args = parser.parse_args()

    global OCR_LANG
    OCR_LANG = args.lang

    ensure_dirs()

    images = list(PROCESSED_DIR.rglob('*.jpg'))
    images += list(PROCESSED_DIR.rglob('*.png'))

    print(f"Found {len(images)} images to OCR")

    failures = []
    for img in images:
        ok, err = process_image(img)
        if not ok:
            failures.append((img.name, err))
            print(f"✗ {img.name}: {err}")
        else:
            print(f"✓ {img.name}")

    print(f"\nDone. Success: {len(images)-len(failures)} / {len(images)}")
    if failures:
        print("Failures:")
        for name, err in failures:
            print(f" - {name}: {err}")


if __name__ == '__main__':
    main()
