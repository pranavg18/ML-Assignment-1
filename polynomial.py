"""Polynomial models, fold-local preprocessing, and portable JSON inference"""
from itertools import combinations_with_replacement
import json
from pathlib import Path
import numpy as np
import pandas as pd


def powers(n_features, degree):
    terms = []
    for d in range(1, degree + 1):
        for indices in combinations_with_replacement(range(n_features), d):
            terms.append(np.bincount(indices, minlength=n_features))
    return np.asarray(terms, dtype=int)


def design(x, exponents):
    z = np.ones((len(x), len(exponents)))
    for j in range(x.shape[1]):
        z *= x[:, j:j+1] ** exponents[:, j]
    return z


def load_data(path, expected, target=False):
    frame = pd.read_csv(path)
    columns = expected + (["y"] if target else [])
    if list(frame.columns) != columns:
        raise ValueError(f"{path}: expected {columns}; got {list(frame.columns)}")
    if len(frame) == 0 or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: empty or nonfinite data")
    return frame


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def predict(model, frame):
    x = frame[model["features"]].to_numpy(dtype=float)
    z = design(x, np.asarray(model["powers"]))
    pred = ((z - model["term_mean"]) / model["term_scale"]) @ np.asarray(model["coef"]) + model["intercept"]
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions")
    return pred
