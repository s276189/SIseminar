from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np
import joblib
import os

def train_model(metrics_list, objective_list, output_dir):
    """
    Balanced Random Forestを使用してモデルを学習し、予測を行う
    :param metrics_list: 特徴量のリスト (X)
    :param objective_list: 目的変数のリスト (y)
    :param output_dir: モデルおよび結果を保存するディレクトリ
    :return: 学習済みモデルと予測結果
    """
    # データの確認
    if not metrics_list or not objective_list:
        raise ValueError("Error: Input data (metrics_list or objective_list) is empty.")

    # 特徴量と目的変数をNumpy配列に変換
    X = np.array(metrics_list)
    y = np.array(objective_list)

    # データ形状の確認
    print(f"Training data shape: X={X.shape}, y={y.shape}")

    # Balanced Random Forestを初期化
    print("Training Balanced Random Forest Classifier...")
    model = BalancedRandomForestClassifier(random_state=0)

    # モデルを学習
    model.fit(X, y)

    # 学習データに対する予測
    predictions = model.predict(X)

    # モデルの性能を評価 (F1スコア)
    f1 = f1_score(y, predictions, average="weighted")
    print(f"Training F1 Score: {f1:.4f}")

    # モデルを指定ディレクトリに保存
    model_path = os.path.join(output_dir, "balanced_random_forest_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    return model, predictions
