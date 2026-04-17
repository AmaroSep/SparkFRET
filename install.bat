@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   SparkFRET Installer
echo   Baylor College of Medicine - Bhatt Lab
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado.
    echo Instala Python 3.10-3.13 desde https://www.python.org/downloads/
    echo Asegurate de marcar "Add Python to PATH" durante la instalacion.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Python encontrado: %PYVER%

:: Create venv
echo.
echo [1/4] Creando entorno virtual (venv)...
python -m venv venv
if errorlevel 1 (
    echo ERROR: No se pudo crear el entorno virtual.
    pause
    exit /b 1
)
echo OK

:: Upgrade pip
echo.
echo [2/4] Actualizando pip...
venv\Scripts\python -m pip install --upgrade pip --quiet

:: PyTorch - detect CUDA
echo.
echo [3/4] Instalando PyTorch...
echo Detectando GPU...

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No se detecto GPU NVIDIA. Instalando PyTorch CPU...
    venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
) else (
    for /f "tokens=3" %%c in ('nvidia-smi --query-gpu^=driver_version --format^=csv,noheader 2^>nul') do set DRIVER=%%c
    echo GPU NVIDIA detectada. Instalando PyTorch CUDA 12.1...
    venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
)
echo OK

:: Install requirements
echo.
echo [4/4] Instalando dependencias (puede tardar 2-5 minutos)...
venv\Scripts\pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Fallo la instalacion de dependencias.
    echo Revisa requirements.txt e intenta manualmente:
    echo   venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
echo OK

:: Create launcher
echo.
echo Creando lanzador (launch_hub.bat)...
(
echo @echo off
echo cd /d "%~dp0"
echo echo Iniciando SparkFRET Hub...
echo venv\Scripts\streamlit run sparkfret_hub.py --server.maxUploadSize 500
) > launch_hub.bat
echo OK

echo.
echo ============================================================
echo   Instalacion completada!
echo.
echo   Para iniciar el hub:
echo     launch_hub.bat
echo   o:
echo     venv\Scripts\streamlit run sparkfret_hub.py
echo ============================================================
echo.
pause
