import os
import json
import pandas as pd
from tabulate import tabulate

# Constants
METRICS_DIR = os.path.join('outputs', 'metrics')

def load_json(filename):
    path = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Please run the training and evaluation scripts first.")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def generate_comparison():
    # 1. Load all metrics
    print("Loading metrics for final comparison...")
    ann_meta = load_json('ann_meta.json')
    cnn_meta = load_json('cnn_meta.json')
    eval_metrics = load_json('eval_metrics.json')

    if not all([ann_meta, cnn_meta, eval_metrics]):
        return

    # 2. Build final summary dict
    final_summary = {
        "ann": {
            "model": "ANN (3 Hidden Layers)",
            "params": ann_meta["params"],
            "test_accuracy": eval_metrics["ann"]["accuracy"],
            "training_time": ann_meta["training_time"],
            "precision": eval_metrics["ann"]["precision"],
            "recall": eval_metrics["ann"]["recall"],
            "f1": eval_metrics["ann"]["f1"]
        },
        "cnn": {
            "model": "CNN (3x3, 5x5, 7x7 Kernels)",
            "params": cnn_meta["params"],
            "test_accuracy": eval_metrics["cnn"]["accuracy"],
            "training_time": cnn_meta["training_time"],
            "precision": eval_metrics["cnn"]["precision"],
            "recall": eval_metrics["cnn"]["recall"],
            "f1": eval_metrics["cnn"]["f1"]
        }
    }

    # 3. Save as outputs/metrics/final_comparison.json
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, 'final_comparison.json'), 'w') as f:
        json.dump(final_summary, f, indent=4)
    print(f"Final comparison saved to {os.path.join(METRICS_DIR, 'final_comparison.json')}")

    # 4. Print formatted comparison table using tabulate
    table_data = []
    headers = ["Metric", "ANN (3 Hidden Layers)", "CNN (Multiscale)"]
    
    metrics_to_show = [
        ("Total Parameters", "params", "{:,}"),
        ("Test Accuracy", "test_accuracy", "{:.4f}"),
        ("Training Time (s)", "training_time", "{:.2f}"),
        ("Precision (Macro)", "precision", "{:.4f}"),
        ("Recall (Macro)", "recall", "{:.4f}"),
        ("F1-Score (Macro)", "f1", "{:.4f}")
    ]

    for label, key, fmt in metrics_to_show:
        ann_val = final_summary["ann"][key]
        cnn_val = final_summary["cnn"][key]
        table_data.append([label, fmt.format(ann_val), fmt.format(cnn_val)])

    print("\n" + "="*60)
    print("           CIFAR-10 MODEL COMPARISON SUMMARY")
    print("="*60)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*60)

if __name__ == "__main__":
    generate_comparison()
