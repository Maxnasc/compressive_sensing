"""
Module: analysis/covert_report.py

Tools for parsing and consolidating classification reports.

This module provides utilities to parse scikit-learn's classification report output
and consolidate results from multiple JSON files into a single CSV file for
easy comparison and analysis across different models and techniques.

Functions:
- parse_classification_report: Convert sklearn report text to DataFrame
- process_reports: Extract reports from JSON files and combine into CSV

Author: Maxnasc7
License: MIT
"""

import os
import json
import pandas as pd
from sklearn.metrics import classification_report
from io import StringIO

def parse_classification_report(report_text):
    """
    Convert raw classification_report text output to DataFrame.
    
    Parses the text output from sklearn's classification_report function
    and converts it into a structured DataFrame with precision, recall, F1-score,
    and support metrics for each class.
    
    Parameters
    ----------
    report_text : str
        Raw text output from sklearn.metrics.classification_report
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Class, Precision, Recall, F1-score, Support
    """
    lines = report_text.strip().split("\n")
    lines = [l for l in lines if l.strip() != ""]  # remove linhas vazias

    # Apenas linhas com números ou 'accuracy'
    useful_lines = []
    for line in lines:
        if line.strip()[0].isdigit() or line.strip().startswith("accuracy") or "avg" in line:
            useful_lines.append(line)

    # Converter para CSV intermediário
    text_for_df = "Class Precision Recall F1-score Support\n"
    for line in useful_lines:
        line = " ".join(line.split())  # normaliza espaços
        parts = line.split(" ")
        if len(parts) == 5:
            text_for_df += " ".join(parts) + "\n"
        elif parts[0] == "accuracy":
            # accuracy tem formato especial
            text_for_df += f"accuracy {parts[-1]} 0 0 {parts[-1]}\n"

    df = pd.read_csv(StringIO(text_for_df), sep=" ")
    return df


def process_reports(directory, output_csv="combined_reports.csv"):
    """
    Extract classification reports from JSON files and combine into CSV.
    
    Scans a directory for JSON files containing classification reports,
    parses them using parse_classification_report, and consolidates all
    results into a single CSV file for analysis.
    
    Parameters
    ----------
    directory : str
        Path to directory containing JSON files with classification reports
    output_csv : str, default='combined_reports.csv'
        Output CSV filename where consolidated reports will be saved
    
    Returns
    -------
    None
        Writes consolidated reports to output_csv file
    
    Notes
    -----
    - Only processes files ending with '.json' that contain 'report' in filename
    - Each JSON should contain a classification report as a string
    """
    all_rows = []

    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        
        if 'report' not in filename:
            continue
        
        filepath = os.path.join(directory, filename)

        with open(filepath, "r") as f:
            try:
                report_text = json.load(f)
            except:
                print(f"Erro ao ler {filename}, ignorando.")
                continue

        df = parse_classification_report(report_text)
        df["source_file"] = filename  # adiciona nome do arquivo
        all_rows.append(df)

    # Combina tudo
    final_df = pd.concat(all_rows, ignore_index=True)

    # Salva como csv
    final_df.to_csv(output_csv, index=False)
    print(f"Arquivo salvo: {output_csv}")

    return final_df


# =============================
# USO:
# =============================

# Basta colocar seus arquivos no mesmo diretório do script
process_reports(directory="compressed_data_classification/models/best_models/svc_rbf", output_csv="compressed_data_classification/models/best_models/svc_rbf/resultado_final_mlp.csv")
