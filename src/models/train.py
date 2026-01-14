%%writefile src/models/train.py
import os
import json
import argparse
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve

import matplotlib.pyplot as plt

from src.models.pipeline import create_pipeline


def pick_features(df):
    # Я беру базовые + то, что добавили FE
    numeric_features = [
        "LIMIT_BAL", "AGE",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        "PAY_STATUS_MEAN", "BILL_TOTAL", "PAY_TOTAL",
    ]
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE", "AGE_BIN", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    # чуть "подчищаем": оставляем только те, что реально есть
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    return numeric_features, categorical_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--model_out", type=str, required=True)
    parser.add_argument("--metrics_out", type=str, default="metrics.json")
    parser.add_argument("--experiment", type=str, default="Credit_Default_Prediction")
    args = parser.parse_args()

    target = "default.payment.next.month"

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    if target not in train_df.columns:
        raise ValueError("No target column in train")

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].astype(int)

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target].astype(int)

    num_feats, cat_feats = pick_features(train_df)

    pipe = create_pipeline(num_feats, cat_feats)

    # простенький grid, чтобы реально было что "подбирать"
    param_grid = {
        "classifier__C": [0.3, 1.0, 3.0],
        "classifier__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )

    # MLflow (пусть будет локально)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics = {
            "test_auc": float(auc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1": float(f1),
        }

        # логируем в mlflow
        mlflow.log_params(search.best_params_)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # ROC curve plot
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC curve")
        roc_path = "roc_curve.png"
        plt.savefig(roc_path, dpi=140)
        plt.close()

        mlflow.log_artifact(roc_path)

        # сохраняем модель
        os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
        joblib.dump(best_model, args.model_out)

        # логируем модель в mlflow тоже
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # metrics.json для DVC
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(args.metrics_out)

        print("best params:", search.best_params_)
        print("metrics:", metrics)
        print("saved model:", args.model_out)


if __name__ == "__main__":
    main()
