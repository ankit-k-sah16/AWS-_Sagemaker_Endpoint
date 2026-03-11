

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sklearn
import joblib
import os
import pandas as pd
import argparse


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    parser.add_argument("--train-files", type=str, default="train-V-1.csv")
    parser.add_argument("--test-files", type=str, default="test-V-1.csv")

    args, _ = parser.parse_known_args()

    print("Sklearn Version:", sklearn.__version__)
    print("Joblib Version:", joblib.__version__)

    train_df = pd.read_csv(os.path.join(args.train, args.train_files))
    test_df = pd.read_csv(os.path.join(args.test, args.test_files))

    label = "price_range"
    features = list(train_df.columns)
    features.remove(label)

    X_train = train_df[features]
    y_train = train_df[label]

    X_test = test_df[features]
    y_test = test_df[label]

    print("Training Random Forest Model...")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=1
    )

    model.fit(X_train, y_train)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print("Model saved at:", model_path)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
