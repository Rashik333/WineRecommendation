import os
import subprocess

dependencies = [
    "streamlit",
    "pandas",
    "numpy",
    "scikit-learn",
    "plotly",
    "lime",
    "shap"
]

for package in dependencies:
    subprocess.call(["pip", "install", package])

