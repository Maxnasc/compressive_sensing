import os
import json
import pandas as pd
from sklearn.metrics import classification_report
from io import StringIO

def parse_classification_report(report_text):
    """
    Converte texto bruto de classification_report em DataFrame
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
    Lê todos os JSONs do diretório, extrai os relatórios e salva em um CSV único.
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
