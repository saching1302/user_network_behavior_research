# User and Network Behavior Analytics for Cybersecurity Risk Detection

## Purpose
This project is a reproducible prototype for the proposed research direction:
**User and Network Behavior Analytics for Cybersecurity Risk Detection**.

The repository includes a small **synthetic/illustrative dataset** so the complete
pipeline can be executed locally without obtaining proprietary organizational data.

## Important research-use note
The included data are synthetic and intentionally contain a small number of anomalous
behavioral observations. Results produced from this sample are **illustrative only**.
They must not be reported in the final paper as empirical evidence from a real
organization or as results from the CERT dataset.

## Pipeline
Data -> behavioral features -> standardization -> Isolation Forest ->
relative risk score -> Low/Medium/High risk -> decision recommendation.

## Features
- login_count
- file_access
- web_requests
- email_count
- device_events
- after_hours

## Outputs
The script generates five publication-ready PNG figures and a CSV risk table in:
`results/figures/` and `results/tables/`.

## Run locally

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\run_analysis.py
```

### Jupyter
```powershell
jupyter notebook
```
Then open `notebooks/user_network_behavior_analysis.py` or copy its cells into a notebook.

## Figures
1. Distribution of file access activity
2. Login activity vs. file access
3. Illustrative user risk levels
4. Top 10 illustrative risk scores
5. File access by access-time category

## Research paper integration
Use the generated figures as **methodology/pipeline illustrations and pilot outputs**.
If the final paper is intended to report empirical findings, replace the synthetic
CSV with a documented real dataset and rerun the same pipeline.

## License
For academic prototyping and educational use.
