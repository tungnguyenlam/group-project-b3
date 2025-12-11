import pandas as pd
import os
from typing import List, Dict
import json
from tabulate import tabulate

def save_output_to_json(source_texts: List[str], expected: List[str], predicted: List[str], file_dir: str):
    file_path = os.path.join(file_dir, 'output.json')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    results_dict = {}
    results_list = []
    for source_text, expected, predicted in zip(source_texts, expected, predicted):
        results_dict = {
            'source_text': source_text,
            'expected': expected,
            'predicted': predicted
        }
        results_list.append(results_dict)
    with open(file_path, 'w') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)


def save_metrics_to_json(metrics: Dict[str, float], file_dir: str):
    file_path = os.path.join(file_dir, 'metrics.json')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(file_path, 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

def print_tabulate_results(metrics: Dict[str, float]):
    metrics_data = [
        ["Model Name", metrics['model_name']],
        ["Character Error Rate (CER)", f"{metrics['cer']:.4f}"],
        ["Word Error Rate (WER)", f"{metrics['wer']:.4f}"],
        ["BLEU Score", f"{metrics['bleu']:.4f}"],
        ["SacreBLEU Score", f"{metrics['sacrebleu']:.4f}"],
        ["chrF Score", f"{metrics['chrf']:.4f}"],
        ["chrF++ Score", f"{metrics['chrf_pp']:.4f}"],
    ]
    print(tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="heavy_outline"))