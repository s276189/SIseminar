from sklearn.metrics import f1_score

def evaluate_model(model, predictions, output_dir):
    """モデルの評価と結果保存"""
    # 仮のF1スコア計算
    f1 = f1_score([1] * len(predictions), predictions)

    with open(f"{output_dir}/report.txt", "w") as f:
        f.write(f"F1 Score: {f1}\n")
