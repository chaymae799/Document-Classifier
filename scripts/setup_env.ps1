Write-Host "Setting up Python virtual environment and installing requirements..."
$venv = "$PWD\\.venv"
if (!(Test-Path $venv)) {
    python -m venv $venv
}
Write-Host "Activating venv and upgrading pip..."
& "$venv\\Scripts\\python.exe" -m pip install --upgrade pip setuptools wheel

Write-Host "Installing requirements (this may take time, PyTorch needs special handling)..."
Try {
    & "$venv\\Scripts\\python.exe" -m pip install -r backend/requirements.txt
} Catch {
    Write-Host "Failed to install some packages. For PyTorch, please follow the instructions at https://pytorch.org/get-started/locally/" -ForegroundColor Yellow
}

Write-Host "Environment setup complete. Activate with: .\\.venv\\Scripts\\Activate.ps1"
