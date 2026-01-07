"""
Trainer + model utilities.

CSV format expected (header):
timestamp,F1,F2,F3,F4,F5,AX,AY,AZ,GX,GY,GZ,label
"""
import os
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "model.joblib"
SCALE_PATH = "scaler.joblib"

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Drop rows without label
    df = df.dropna(subset=["label"])
    # Feature columns in consistent order
    feat_cols = ["F1","F2","F3","F4","F5","AX","AY","AZ","GX","GY","GZ"]
    X = df[feat_cols].astype(float).values
    y = df["label"].astype(str).values
    return X, y

def train_knn(csv_path, k=3):
    X, y = load_data(csv_path)
    # simple scaler: mean/std
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0

    Xs = (X - mu) / sigma

    try:
        # Prefer scikit-learn KNeighborsClassifier if present
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(Xs, y)
        model = {"type": "sklearn_knn", "model": clf, "mu": mu, "sigma": sigma}
        joblib.dump(model, MODEL_PATH)
        print(f"Trained sklearn KNN and saved to {MODEL_PATH}")
    except Exception as e:
        # fallback: store training data to do nearest neighbor with numpy at predict-time
        model = {"type": "numpy_knn", "X": X, "y": y, "mu": mu, "sigma": sigma}
        joblib.dump(model, MODEL_PATH)
        print("scikit-learn not available or failed; saved numpy fallback model.")

    # Save scaler separately for easy reuse (optional)
    joblib.dump({"mu": mu, "sigma": sigma}, SCALE_PATH)
    return model

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def predict(sample_dict):
    """
    sample_dict: dict with keys F1..F5,AX..GZ numeric
    returns: predicted label and distance/confidence
    """
    model = load_model()
    if model is None:
        raise ValueError("No model found. Train first with train_knn(csv_path).")

    feat_cols = ["F1","F2","F3","F4","F5","AX","AY","AZ","GX","GY","GZ"]
    x = np.array([float(sample_dict[c]) for c in feat_cols], dtype=float)
    mu = model.get("mu")
    sigma = model.get("sigma")
    xs = (x - mu) / sigma

    if model["type"] == "sklearn_knn":
        clf = model["model"]
        pred = clf.predict(xs.reshape(1, -1))[0]
        # get distance to neighbors as a crude confidence
        try:
            dists, idx = clf.kneighbors(xs.reshape(1, -1), n_neighbors=clf.n_neighbors)
            conf = 1.0 / (1.0 + float(dists.mean()))
        except Exception:
            conf = 0.5
        return {"label": pred, "confidence": conf}
    else:
        # numpy fallback: nearest neighbor on scaled features
        Xtrain = (model["X"] - mu) / sigma
        diffs = Xtrain - xs
        dists = np.linalg.norm(diffs, axis=1)
        best = np.argmin(dists)
        pred = model["y"][best]
        conf = 1.0 / (1.0 + float(dists[best]))
        return {"label": pred, "confidence": conf, "distance": float(dists[best])}

if __name__ == "__main__":
    # quick CLI: train from csv
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Labeled CSV to train from")
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()
    train_knn(args.csv, k=args.k)
