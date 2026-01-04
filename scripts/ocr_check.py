import shutil
import subprocess
from pathlib import Path
from PIL import Image
import pytesseract

print('cwd:', Path('.').resolve())
print('pytesseract version:', getattr(pytesseract, '__version__', '?'))

tess_path = shutil.which('tesseract')
print('tesseract binary:', tess_path)

try:
    if tess_path:
        p = subprocess.run([tess_path, '--version'], capture_output=True, text=True, check=False)
        print('tesseract --version:\n', p.stdout.splitlines()[0] if p.stdout else p.stderr)
except Exception as e:
    print('error running tesseract --version:', e)

# list installed languages via pytesseract if available
try:
    langs = pytesseract.get_languages(config='')
    print('pytesseract.get_languages():', langs)
except Exception as e:
    print('get_languages error:', e)

# try running OCR on the first jpg or png in data/raw_images
p = Path('data/raw_images')
imgs = list(p.glob('*.jpg')) + list(p.glob('*.jpeg')) + list(p.glob('*.png'))
if not imgs:
    print('no images found in', p)
else:
    img = imgs[0]
    print('testing image:', img)
    try:
        im = Image.open(img)
        text = pytesseract.image_to_string(im, lang='fra+eng')
        print('ocr length:', len(text))
        print('ocr sample (first 400 chars):')
        print(repr(text[:400]))
    except Exception as e:
        print('ocr error:', e)
