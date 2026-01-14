%%writefile src/data/make_dataset.py
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.validation import validate_df
from src.features.build_features import build_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="path to raw csv")
    parser.add_argument("--output_dir", type=str, required=True, help="processed dir")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    df = validate_df(df)
    df = build_features(df)

    # если target нет — ошибка 
    target = "default.payment.next.month"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=df[target]
    )

    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("saved:", train_path, test_path)


if __name__ == "__main__":
    main()
