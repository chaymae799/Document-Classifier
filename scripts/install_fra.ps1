Write-Host "Downloading fra.traineddata into Tesseract tessdata folder..."
$temp = "$env:TEMP\fra.traineddata"
$url = 'https://github.com/tesseract-ocr/tessdata_fast/raw/main/fra.traineddata'
Invoke-WebRequest -Uri $url -OutFile $temp
$tessdata = 'C:\Program Files\Tesseract-OCR\tessdata'
if (!(Test-Path $tessdata)) {
    Write-Host "Tessdata folder not found at $tessdata. Please verify Tesseract installation." -ForegroundColor Yellow
    exit 1
}
Copy-Item $temp -Destination (Join-Path $tessdata 'fra.traineddata') -Force
Write-Host "Installed fra.traineddata to $tessdata"
Write-Host "You may need to restart any shells or services using Tesseract."
