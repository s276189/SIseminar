import json
import os

def load_pr_data(data_path):
    """PRデータを読み込む"""
    pr_data = []
    for file in os.listdir(data_path):
        if file.endswith(".json"):
            with open(os.path.join(data_path, file), 'r') as f:
                pr_data.append(json.load(f))
    return pr_data
