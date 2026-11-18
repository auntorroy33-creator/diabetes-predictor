"""Shared code used by both train.py and app.py (kept here so the saved model can be loaded)."""
import numpy as np
import pandas as pd

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"

# In the Pima dataset a value of 0 for these columns really means "not measured"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def zero_to_nan(X):
    X = pd.DataFrame(X, columns=FEATURES).copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    return X
