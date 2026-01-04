Frontend for Phase 1 - Document Classifier

Quick start

1. Start the backend (from `backend/`):

```powershell
cd backend
python app.py
```

2. Serve the frontend (from project root or `frontend/`):

```powershell
cd frontend
python -m http.server 8000
# then open http://localhost:8000 in your browser
```

Notes

- The frontend sends a multipart POST to `/api/classify` (field name `file`). Ensure the Flask backend is reachable at the same host/port or adjust the fetch URL in `app.js`.
- If OCR returns empty text, install Tesseract on your OS and ensure it's in `PATH`, or set `pytesseract.pytesseract.tesseract_cmd` to the tesseract executable path.
- Phase 1 uses pretrained models; you do not need labeled data to run it. However, to improve accuracy for your documents you should collect labeled examples and perform Phase 2 (fine-tuning).

Recommended Tesseract installer (Windows): https://github.com/tesseract-ocr/tesseract#windows
