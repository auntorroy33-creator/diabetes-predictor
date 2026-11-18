import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from common import FEATURES

ROOT = Path(__file__).parent
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.warning("Educational project only. This is NOT a medical diagnosis. "
           "Please consult a doctor for any health concern.")

if not (ROOT / "model.joblib").exists():
    st.error("Model not trained yet. In the terminal run:  `python train.py`  then refresh this page.")
    st.stop()


@st.cache_resource
def load():
    return joblib.load(ROOT / "model.joblib"), json.load(open(ROOT / "metrics.json"))


model, info = load()
med = info["medians"]

# ------------------------------------------------------------------ inputs
st.sidebar.header("Patient details")
vals = {
    "Pregnancies": st.sidebar.number_input("Pregnancies", 0, 20, 1),
    "Glucose": st.sidebar.number_input("Glucose (mg/dL)", 40, 300, 120),
    "BloodPressure": st.sidebar.number_input("Blood pressure, diastolic (mm Hg)", 30, 140, 70),
    "SkinThickness": st.sidebar.number_input("Skin thickness (mm)", 5, 100, 20),
    "Insulin": st.sidebar.number_input("Insulin (mu U/ml)", 10, 900, 80),
    "BMI": st.sidebar.number_input("BMI", 10.0, 70.0, 28.0, step=0.1),
    "DiabetesPedigreeFunction": st.sidebar.number_input(
        "Diabetes pedigree function (family history score)", 0.05, 2.5, 0.40, step=0.01),
    "Age": st.sidebar.number_input("Age", 15, 100, 35),
}
row = pd.DataFrame([vals])[FEATURES]

tab1, tab2 = st.tabs(["Prediction", "Model performance"])

# ------------------------------------------------------------------ prediction
with tab1:
    if st.button("Predict risk", type="primary"):
        p = float(model.predict_proba(row)[0, 1])
        pct = int(round(p * 100))
        if p < 0.30:
            st.success(f"🟢 LOW risk: {pct}% estimated probability")
        elif p < 0.60:
            st.warning(f"🟡 MODERATE risk: {pct}% estimated probability")
        else:
            st.error(f"🔴 HIGH risk: {pct}% estimated probability")
        st.progress(p)

        # What-if explanation: how much does each value move the risk vs. a typical patient?
        st.subheader("What is driving this result?")
        effects = {}
        for f in FEATURES:
            alt = row.copy()
            alt[f] = med[f]
            effects[f] = p - float(model.predict_proba(alt)[0, 1])
        eff = pd.Series(effects).sort_values()
        st.bar_chart(eff.rename("Change in risk vs. typical value"), horizontal=True)
        st.caption("Positive bars push the risk up compared with a typical patient; negative bars lower it.")

        top = eff.sort_values(ascending=False)
        up = [f"**{f}**" for f, v in top.items() if v > 0.02][:3]
        if up:
            st.markdown("Main risk-raising factors: " + ", ".join(up) + ".")

        st.subheader("General healthy-habit tips")
        st.markdown(
            "- Keep a healthy weight and stay physically active (about 150 min/week)\n"
            "- Reduce sugary drinks and refined carbohydrates\n"
            "- Get regular blood sugar and blood pressure checks\n"
            "- If your risk is moderate or high, see a doctor for a proper test (e.g. HbA1c, fasting glucose)"
        )
    else:
        st.info("Fill in the patient details in the sidebar and click **Predict risk**.")

# ------------------------------------------------------------------ performance
with tab2:
    st.markdown(f"**Selected model:** {info['best_model']}  |  **Hold-out ROC-AUC:** {info['holdout_auc']}")
    st.markdown("**Model comparison (5-fold cross-validation)**")
    st.dataframe(pd.DataFrame(info["cv_table"]).set_index("Model"))
    st.markdown("**Feature importance** (drop in ROC-AUC when a feature is shuffled)")
    st.bar_chart(pd.Series(info["importance"]), horizontal=True)
    rep = info["holdout_report"]["diabetic"]
    st.caption(f"Diabetic class on hold-out data: precision {rep['precision']:.2f}, "
               f"recall {rep['recall']:.2f}, F1 {rep['f1-score']:.2f}. "
               "Dataset: Pima Indians Diabetes (768 women, 21+ years old), so results may not "
               "generalize to other populations.")
