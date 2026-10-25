from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
CLF_PATH = MODEL_DIR / "best_classification_model.joblib"
REG_PATH = MODEL_DIR / "best_regression_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Smart Outcome Predictor",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 1.5rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px;
       border: 1px solid rgba(128,128,128,.25);
       background: linear-gradient(120deg, rgba(65,105,225,.15), rgba(20,184,166,.10));}
.hero h1 {margin: 0 0 .35rem 0; font-size: 2.1rem;}
.hero p {margin: 0; opacity: .8;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    clf = joblib.load(CLF_PATH) if CLF_PATH.exists() else None
    reg = joblib.load(REG_PATH) if REG_PATH.exists() else None
    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return clf, reg, meta

def get_preprocessor(model):
    if hasattr(model, "named_steps"):
        for key in ("prep", "preprocessor", "preprocess"):
            if key in model.named_steps:
                return model.named_steps[key]
    return None

def get_input_columns(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    prep = get_preprocessor(model)
    if prep is not None and hasattr(prep, "feature_names_in_"):
        return list(prep.feature_names_in_)
    if prep is not None and hasattr(prep, "transformers_"):
        cols = []
        for _, _, selected in prep.transformers_:
            if isinstance(selected, (list, tuple, np.ndarray, pd.Index)):
                cols.extend(list(selected))
        if cols:
            return list(dict.fromkeys(cols))
    return []

def get_feature_types(model, columns):
    numeric, categorical = set(), set()
    prep = get_preprocessor(model)
    if prep is not None and hasattr(prep, "transformers_"):
        for name, _, selected in prep.transformers_:
            if isinstance(selected, (list, tuple, np.ndarray, pd.Index)):
                selected = [c for c in selected if c in columns]
                if str(name).lower() in ("num", "numeric", "numerical"):
                    numeric.update(selected)
                elif str(name).lower() in ("cat", "categorical"):
                    categorical.update(selected)
    numeric -= categorical
    categorical.update(c for c in columns if c not in numeric and c not in categorical)
    return numeric, categorical

def input_form(model, key, button_text):
    columns = get_input_columns(model)
    if not columns:
        st.error(
            "Model ke original feature names nahi mil rahe. Original pandas DataFrame "
            "par fitted complete preprocessing Pipeline export karo."
        )
        return None

    numeric, categorical = get_feature_types(model, columns)
    values = {}
    with st.form(key):
        st.caption("Training dataset ke original input features enter karo.")
        col_a, col_b = st.columns(2)
        for i, feature in enumerate(columns):
            with (col_a if i % 2 == 0 else col_b):
                label = str(feature).replace("_", " ").title()
                if feature in numeric:
                    values[feature] = st.number_input(
                        label, value=0.0, step=1.0, key=f"{key}_{feature}"
                    )
                else:
                    values[feature] = st.text_input(
                        label, value="", key=f"{key}_{feature}",
                        help="Training data ke category spelling/case ko match karo."
                    )
        submitted = st.form_submit_button(button_text, use_container_width=True)

    if not submitted:
        return None
    row = pd.DataFrame([[values[c] for c in columns]], columns=columns)
    for col in numeric:
        row[col] = pd.to_numeric(row[col], errors="coerce")
    for col in categorical:
        row[col] = row[col].replace("", np.nan)
    return row

def completion_probability(model, row):
    """Return probability for class 1 if the model supports predict_proba."""
    if not hasattr(model, "predict_proba"):
        return None

    probs = model.predict_proba(row)
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        final_model = list(model.named_steps.values())[-1]
        classes = getattr(final_model, "classes_", None)

    if classes is not None and 1 in list(classes):
        idx = list(classes).index(1)
    else:
        idx = 1 if probs.shape[1] > 1 else 0

    return float(probs[0][idx])


st.markdown("""
<div class="hero">
  <h1>🎓 Smart Outcome Predictor</h1>
  <p>Predict student course completion and estimate final scores.</p>
</div>
""", unsafe_allow_html=True)
st.write("")

try:
    classifier, regressor, metadata = load_artifacts()
except Exception as error:
    st.error("Model load nahi ho paya. Python/scikit-learn/XGBoost versions check karo.")
    st.exception(error)
    st.stop()

if classifier is None or regressor is None:
    st.warning("Trained model files nahi mile.")
    st.markdown("Notebook mein Parts B–F aur model-export cell run karo.")
    st.code(
        "models/best_classification_model.joblib\n"
        "models/best_regression_model.joblib\n"
        "models/model_metadata.json",
        language="text",
    )
    st.info("app.py aur models/ folder same project directory mein hone chahiye.")
    st.stop()

with st.sidebar:
    st.header("📌 Model Details")
    st.write("**Classification:**", metadata.get("classification_model", "Loaded model"))
    if metadata.get("classification_f1_score") is not None:
        st.caption(f"Selection F1 Score: {metadata['classification_f1_score']:.3f}")
    st.divider()
    st.write("**Regression:**", metadata.get("regression_model", "Loaded model"))
    if metadata.get("regression_rmse") is not None:
        st.caption(f"Selection RMSE: {metadata['regression_rmse']:.3f}")
    st.caption("Predictions are estimates, not guaranteed outcomes.")

tab1, tab2, tab3 = st.tabs(
    ["🎯 Completion Predictor", "📈 Final Score Predictor", "ℹ️ About"]
)

with tab1:
    st.subheader("Course Completion Prediction")
    row = input_form(classifier, "clf_form", "✨ Predict Completion")
    if row is not None:
        try:
            prediction = classifier.predict(row)[0]
            is_completed = prediction == 1 or str(prediction) == "1"
            status = "Completed" if is_completed else "Not Completed"
            probability = completion_probability(classifier, row)
            a, b = st.columns(2)
            a.metric("Predicted Status", status)
            if probability is not None:
                b.metric("Estimated Completion Probability", f"{probability:.1%}")
                st.progress(float(np.clip(probability, 0, 1)))
            else:
                b.metric("Predicted Class", str(prediction))
            if is_completed:
                st.success("Model completion predict karta hai. Progress monitor karte rahein.")
            else:
                st.warning("Model non-completion predict karta hai. Academic support consider karein.")
        except Exception as error:
            st.error("Prediction fail hui. Input schema aur saved model pipeline verify karo.")
            st.exception(error)

with tab2:
    st.subheader("Final Score Estimation")
    row = input_form(regressor, "reg_form", "✨ Estimate Final Score")
    if row is not None:
        try:
            raw_score = float(regressor.predict(row)[0])
            score = float(np.clip(raw_score, 0, 100))
            a, b = st.columns([1, 2])
            a.metric("Estimated Final Score", f"{score:.1f} / 100")
            b.progress(score / 100)
            b.caption(f"Raw model output: {raw_score:.2f}")
            if score < 40:
                st.warning("Estimated score lower range mein hai; extra practice helpful ho sakti hai.")
            elif score < 70:
                st.info("Focused revision aur practice helpful ho sakti hai.")
            else:
                st.success("Estimated score higher range mein hai. Consistent preparation maintain karein.")
            st.caption("Score ranges are general guidance, not official grading thresholds.")
        except Exception as error:
            st.error("Score prediction fail hui. Input schema aur saved model pipeline verify karo.")
            st.exception(error)

with tab3:
    st.subheader("About this project")
    st.markdown("""
    - **Classification target:** `completion_status` (0 = Not Completed, 1 = Completed)
    - **Regression target:** `final_score` (0–100)
    - Saved pipelines should include preprocessing such as imputation, scaling and encoding.
    - Validate models on representative new data and protect student privacy.
    - Predictions should support educators, not be the sole basis for consequential decisions.
    """)

st.divider()
st.caption("Smart Outcome Predictor • Machine-learning decision-support demo")
