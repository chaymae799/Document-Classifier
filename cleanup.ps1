# Script de nettoyage du projet document-classifier
# Garde uniquement les fichiers fonctionnels essentiels

Write-Host "🧹 NETTOYAGE DU PROJET" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan

# Suppression des caches Python
Write-Host "`n📂 Suppression des caches Python..." -ForegroundColor Yellow
Remove-Item -Recurse -Force backend\__pycache__ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force backend\modules\__pycache__ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force backend\utils\__pycache__ -ErrorAction SilentlyContinue
Write-Host "   ✅ Caches supprimés" -ForegroundColor Green

# Suppression des scripts de training (déjà terminé)
Write-Host "`n📂 Suppression des scripts de training..." -ForegroundColor Yellow
Remove-Item backend\train_fast.py -ErrorAction SilentlyContinue
Remove-Item backend\train_hybrid.py -ErrorAction SilentlyContinue
Remove-Item backend\prepare_dataset.py -ErrorAction SilentlyContinue
Remove-Item backend\preprocess_pipeline.py -ErrorAction SilentlyContinue
Write-Host "   ✅ Scripts de training supprimés" -ForegroundColor Green

# Suppression des scripts d'évaluation dupliqués
Write-Host "`n📂 Suppression des évaluations dupliquées..." -ForegroundColor Yellow
Remove-Item backend\evaluate_model.py -ErrorAction SilentlyContinue
Remove-Item backend\evaluate_nlp.py -ErrorAction SilentlyContinue
Remove-Item backend\evaluate_gabarits.py -ErrorAction SilentlyContinue
Write-Host "   ✅ Évaluations dupliquées supprimées" -ForegroundColor Green

# Suppression des scripts optionnels
Write-Host "`n📂 Suppression des scripts optionnels..." -ForegroundColor Yellow
Remove-Item backend\download_models.py -ErrorAction SilentlyContinue
Remove-Item backend\export_ocr.py -ErrorAction SilentlyContinue
Write-Host "   ✅ Scripts optionnels supprimés" -ForegroundColor Green

# Suppression des vieux checkpoints
Write-Host "`n📂 Nettoyage des vieux checkpoints..." -ForegroundColor Yellow
Remove-Item -Recurse -Force backend\models\cv\checkpoints -ErrorAction SilentlyContinue
Write-Host "   ✅ Vieux checkpoints supprimés" -ForegroundColor Green

# Nettoyage des dossiers temporaires
Write-Host "`n📂 Nettoyage des dossiers temporaires..." -ForegroundColor Yellow
Remove-Item -Recurse -Force logs -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force results -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force models -ErrorAction SilentlyContinue
Write-Host "   ✅ Dossiers temporaires supprimés" -ForegroundColor Green

# Nettoyage des uploads
Write-Host "`n📂 Nettoyage des uploads..." -ForegroundColor Yellow
Get-ChildItem -Path uploads -File -ErrorAction SilentlyContinue | Remove-Item -Force
Write-Host "   ✅ Uploads nettoyés" -ForegroundColor Green

Write-Host "`n✅ NETTOYAGE TERMINÉ!" -ForegroundColor Green
Write-Host "`n📋 STRUCTURE FINALE:" -ForegroundColor Cyan
Write-Host "   ✅ frontend/ (app.js, index_new.html, style.css)" -ForegroundColor White
Write-Host "   ✅ backend/app.py (serveur Flask)" -ForegroundColor White
Write-Host "   ✅ backend/modules/ (NLP, CV, Gabarits, Fusion)" -ForegroundColor White
Write-Host "   ✅ backend/models/cv/ (modèles entraînés)" -ForegroundColor White
Write-Host "   ✅ backend/utils/ (outils)" -ForegroundColor White
Write-Host "   ✅ backend/config.py, requirements.txt" -ForegroundColor White
Write-Host "   ✅ backend/calculate_metrics.py" -ForegroundColor White
Write-Host "   ✅ backend/evaluate_all_modules.py" -ForegroundColor White
Write-Host "   ✅ backend/evaluate_fusion.py" -ForegroundColor White
Write-Host "   ✅ backend/final_inference.py" -ForegroundColor White
Write-Host "   ✅ data/, data_augmented/, Dataset/" -ForegroundColor White
Write-Host "   ✅ scripts/" -ForegroundColor White
Write-Host "   ✅ uploads/ (vide)" -ForegroundColor White
Write-Host "   ✅ start_servers.py, README.md" -ForegroundColor White
