@echo off
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\download_data.py
python src\experiment.py
