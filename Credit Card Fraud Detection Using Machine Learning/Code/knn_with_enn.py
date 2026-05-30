import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.under_sampling import EditedNearestNeighbours


def main(csv_path: Path, test_size: float = 0.2, random_state: int = 42, n_neighbors: int = 5):
    df = pd.read_csv(csv_path)
    if 'Class' not in df.columns:
        raise ValueError("Expected column 'Class' as target in the CSV")

    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    enn = EditedNearestNeighbours()
    X_res, y_res = enn.fit_resample(X_train_scaled, y_train)

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    clf.fit(X_res, y_res)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = float('nan')

    print(f"Test size: {test_size}, Random state: {random_state}")
    print(f"Samples before ENN (train): {len(X_train)}")
    print(f"Samples after ENN (train): {len(X_res)}")
    print(f"Accuracy: {acc:.6f}")
    print(f"ROC AUC: {roc_auc:.6f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train KNN after ENN cleaning on training set.')
    parser.add_argument('--csv', type=str, default='creditcard.csv', help='Path to creditcard.csv')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--n-neighbors', type=int, default=5, help='K for KNN')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        # assume script is in same folder as CSV
        csv_path = Path(__file__).parent / args.csv
    main(csv_path, test_size=args.test_size, random_state=args.random_state, n_neighbors=args.n_neighbors)
