import argparse
import os
#from sklearn import datasets
from utils.data_loader import load_pr_data
from utils.feature_extractor import extract_features
from utils.model_trainer import train_model
from utils.evaluator import evaluate_model

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="PR Review Prediction Tool")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--data_path", required=True, help="Path to input PR data")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    args = parser.parse_args()

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. データ読み込み
    print("Loading PR data...")
    pr_data = load_pr_data(args.data_path)
    
    # 2. 特徴量抽出
    print("Extracting features...")
    metrics_list, objective_list = extract_features(pr_data, args.start_date, args.end_date)
    print("object_list:", objective_list)


    # 3. モデル学習と予測
    print("Training model...")
    model, predictions = train_model(metrics_list, objective_list, args.output_dir)

    # 4. 評価と結果保存
    print("Evaluating model...")
    evaluate_model(model, predictions, args.output_dir)

    print("Process completed. Results saved to:", args.output_dir)

if __name__ == "__main__":
    main()
