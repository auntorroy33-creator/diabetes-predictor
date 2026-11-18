"""Train and compare 3 models on the Pima Indians Diabetes dataset, save the best one.

Dataset: put a file named  data/diabetes.csv  (Kaggle 'Pima Indians Diabetes Database')
in the data folder, OR just run this script and it will try to download it automatically.
"""
import json
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import FEATURES, TARGET, ZERO_AS_MISSING, zero_to_nan

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "diabetes.csv"
MODEL_FILE = ROOT / "model.joblib"
METRICS_FILE = ROOT / "metrics.json"
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        print("data/diabetes.csv not found, trying to download...")
        DATA_FILE.parent.mkdir(exist_ok=True)
        try:
            urllib.request.urlretrieve(URL, DATA_FILE)
        except Exception as e:
            raise SystemExit(
                f"\nCould not download the dataset ({e}).\n"
                "Please download 'diabetes.csv' manually from Kaggle:\n"
                "  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
                f"and save it as: {DATA_FILE}\n"
            )
    with open(DATA_FILE) as f:
        has_header = f.readline().lower().startswith("pregnancies")
    df = pd.read_csv(DATA_FILE) if has_header else pd.read_csv(DATA_FILE, header=None, names=FEATURES + [TARGET])
    return df[FEATURES + [TARGET]].dropna()


def make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("zero_to_nan", FunctionTransformer(zero_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=42),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
}

if __name__ == "__main__":
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    print(f"Loaded {len(df)} patients | diabetic: {y.sum()} ({y.mean():.0%}) | healthy: {(y == 0).sum()}\n")

    # 1) Compare models with 5-fold cross-validation
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scoring = ["accuracy", "recall", "precision", "f1", "roc_auc"]
    rows = []
    for name, make in MODELS.items():
        res = cross_validate(make_pipeline(make()), X, y, cv=cv, scoring=scoring)
        rows.append({"Model": name, **{m.upper(): round(res[f"test_{m}"].mean(), 3) for m in scoring}})
    table = pd.DataFrame(rows).set_index("Model")
    print("=== 5-fold cross-validation ===")
    print(table, "\n")
    best_name = table["ROC_AUC"].idxmax()
    print(f"Best model by ROC-AUC: {best_name}\n")

    # 2) Hold-out test for a detailed report + feature importance
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = make_pipeline(MODELS[best_name]()).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    print("=== Hold-out test (20%) ===")
    print(classification_report(y_te, pred, target_names=["healthy", "diabetic"]))
    print("Confusion matrix [rows=true, cols=pred]:\n", confusion_matrix(y_te, pred))
    print(f"ROC-AUC: {roc_auc_score(y_te, proba):.3f}")

    imp = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=20, random_state=42)
    importance = dict(sorted(zip(FEATURES, imp.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))

    # 3) Save final model trained on ALL data
    final = make_pipeline(MODELS[best_name]()).fit(X, y)
    joblib.dump(final, MODEL_FILE)

    clean = df.copy()
    clean[ZERO_AS_MISSING] = clean[ZERO_AS_MISSING].replace(0, float("nan"))
    json.dump({
        "best_model": best_name,
        "cv_table": table.reset_index().to_dict(orient="records"),
        "holdout_auc": round(float(roc_auc_score(y_te, proba)), 3),
        "holdout_report": classification_report(y_te, pred, output_dict=True, target_names=["healthy", "diabetic"]),
        "importance": importance,
        "medians": clean[FEATURES].median().to_dict(),
        "ranges": {c: [float(clean[c].min()), float(clean[c].max())] for c in FEATURES},
    }, open(METRICS_FILE, "w"), indent=2)
    print(f"\nSaved {MODEL_FILE.name} and {METRICS_FILE.name}")
