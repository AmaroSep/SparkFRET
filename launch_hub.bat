@echo off
cd /d "%~dp0"
echo Iniciando SparkFRET Hub...
venv\Scripts\streamlit run sparkfret_hub.py --server.maxUploadSize 500
pause
