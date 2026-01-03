DOC_NOTES = """
RC Shear Wall Bilinear Force–Displacement Curve GUI
- Predict 4 outputs: Dy (mm), Fy (kN), Du (mm), Fu (kN)
- Theta removed from inputs
- Supports: RF multi-output, ANN multi-output, CatBoost/XGBoost/LightGBM per-output models
- Uses EXACT filenames as uploaded to your GitHub repo
"""

# ============================================================
# 🚀 STEP 1: CORE IMPORTS & ENVIRONMENT SETUP
# ============================================================

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ============================================================
# 🚀 SUB STEP 1.1: OPTIONAL ML LIBRARIES (SAFE IMPORTS)
# ============================================================

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None


# ============================================================
# 🚀 STEP 2: STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(page_title="RC Wall Bilinear Curve GUI", layout="wide")


# ============================================================
# 🚀 STEP 3: FILE SEARCH UTILITIES (ROBUST FOR STREAMLIT CLOUD)
# ============================================================

def pfind(fname: str) -> Path | None:
    """
    Find file in common locations:
    - repo root
    - models/ model/ assets/ data/
    - fallback recursive search
    """
    roots = [
        Path("."),
        Path("./models"),
        Path("./model"),
        Path("./Model"),
        Path("./assets"),
        Path("./data"),
    ]
    for r in roots:
        p = r / fname
        if p.exists():
            return p.resolve()

    # fallback: recursive search
    for p in Path(".").rglob(fname):
        return p.resolve()

    return None


# ============================================================
# 🚀 STEP 4: INPUT FEATURES (THETA REMOVED)
# ============================================================

# IMPORTANT:
# Keep the SAME feature list/order used in your training.
# If your bilinear model uses different columns, edit FEATURES + TRAIN_NAME_MAP below.

FEATURES = [
    "bw", "L", "h", "fc", "fy",
    "ρt", "ρsh", "ρl", "ρbl",
    "a/d", "P"
]

# ============================================================
# 🚀 SUB STEP 4.1: UI RANGES (OPTIONAL; ADJUST IF NEEDED)
# ============================================================

R = {
    "bw":  (100.0, 500.0),
    "L":   (300.0, 3000.0),
    "h":   (200.0, 1200.0),
    "fc":  (15.0, 80.0),
    "fy":  (200.0, 800.0),
    "ρt":  (0.10, 8.0),
    "ρsh": (0.10, 8.0),
    "ρl":  (0.10, 8.0),
    "ρbl": (0.10, 8.0),
    "a/d": (0.50, 6.00),
    "P":   (0.00, 0.60),
}

# ============================================================
# 🚀 SUB STEP 4.2: COLUMN NAME MAPPING (GUI -> TRAINING)
# ============================================================

# If your training column names differ, map them here.
# If training uses same names, set TRAIN_NAME_MAP = {}.
TRAIN_NAME_MAP = {
    "ρt": "pt",
    "ρsh": "psh",
    "ρl": "pl",
    "ρbl": "pbl",
    # Uncomment ONLY if your training used these names:
    # "a/d": "a_over_d",
    # "P": "axial_ratio",
}

def to_train_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if TRAIN_NAME_MAP:
        df = df.rename(columns=TRAIN_NAME_MAP)
    return df


# ============================================================
# 🚀 STEP 5: OUTPUT DEFINITIONS (4 OUTPUTS)
# ============================================================

OUTPUTS = ["Dy (mm)", "Fy (kN)", "Du (mm)", "Fu (kN)"]

# These tags match your uploaded filenames:
# Best_*_Δymm  -> Dy
# Best_*_FykN  -> Fy
# Best_*_Δmmm  -> Du
# Best_*_FmkN  -> Fu
TAG = {
    "Dy (mm)": "Δymm",
    "Fy (kN)": "FykN",
    "Du (mm)": "Δmmm",
    "Fu (kN)": "FmkN",
}


# ============================================================
# 🚀 STEP 6: MODEL FILENAMES (EXACT MATCH TO YOUR REPO)
# ============================================================

RF_FILE = "Best_RF_Model.pkl"

ANN_MLP_FILE = "ANN_MLP_Model.keras"
ANN_MLP_SX   = "ANN_MLP_Scaler_X.save"
ANN_MLP_SY   = "ANN_MLP_Scaler_y.save"

ANN_PS_FILE  = "ANN_PS_Model.keras"
ANN_PS_SX    = "ANN_PS_Scaler_X.save"
ANN_PS_SY    = "ANN_PS_Scaler_y.save"

CAT_PREFIX = "Best_CatBoost_"
XGB_PREFIX = "Best_XGBoost_"
LGB_PREFIX = "Best_LightGBM_"


# ============================================================
# 🚀 STEP 7: SCALER SHIM (FOR .save FILES)
# ============================================================

class _ScalerShim:
    def __init__(self, scaler):
        self.scaler = scaler
    def transform(self, X):
        return self.scaler.transform(X)
    def inverse_transform(self, Y):
        return self.scaler.inverse_transform(Y)


# ============================================================
# 🚀 STEP 8: LOAD ALL MODELS (CACHED)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_all_models():
    models = {}
    health = {"loaded": [], "missing": [], "notes": []}

    # ------------------------------------------------------------
    # 🚀 SUB STEP 8.1: RF (MULTI-OUTPUT)
    # ------------------------------------------------------------
    p = pfind(RF_FILE)
    if p:
        try:
            models["RF"] = joblib.load(p)
            health["loaded"].append(str(p))
        except Exception as e:
            health["missing"].append(f"{RF_FILE} (error: {e})")
    else:
        health["missing"].append(RF_FILE)

    # ------------------------------------------------------------
    # 🚀 SUB STEP 8.2: ANN MLP (MULTI-OUTPUT)
    # ------------------------------------------------------------
    if load_model is not None:
        pm = pfind(ANN_MLP_FILE)
        psx = pfind(ANN_MLP_SX)
        psy = pfind(ANN_MLP_SY)
        if pm and psx and psy:
            try:
                m = load_model(pm)
                sx = _ScalerShim(joblib.load(psx))
                sy = _ScalerShim(joblib.load(psy))
                models["ANN_MLP"] = {"model": m, "sx": sx, "sy": sy}
                health["loaded"] += [str(pm), str(psx), str(psy)]
            except Exception as e:
                health["missing"].append(f"{ANN_MLP_FILE} (error: {e})")
        else:
            health["missing"].append(f"{ANN_MLP_FILE} / scalers not found")
    else:
        health["notes"].append("TensorFlow not available -> ANN MLP disabled.")

    # ------------------------------------------------------------
    # 🚀 SUB STEP 8.3: ANN PS (MULTI-OUTPUT)
    # ------------------------------------------------------------
    if load_model is not None:
        pm = pfind(ANN_PS_FILE)
        psx = pfind(ANN_PS_SX)
        psy = pfind(ANN_PS_SY)
        if pm and psx and psy:
            try:
                m = load_model(pm)
                sx = _ScalerShim(joblib.load(psx))
                sy = _ScalerShim(joblib.load(psy))
                models["ANN_PS"] = {"model": m, "sx": sx, "sy": sy}
                health["loaded"] += [str(pm), str(psx), str(psy)]
            except Exception as e:
                health["missing"].append(f"{ANN_PS_FILE} (error: {e})")
        else:
            health["missing"].append(f"{ANN_PS_FILE} / scalers not found")
    else:
        health["notes"].append("TensorFlow not available -> ANN PS disabled.")

    # ------------------------------------------------------------
    # 🚀 SUB STEP 8.4: CATBOOST (4 SEPARATE MODELS)
    # ------------------------------------------------------------
    if CatBoostRegressor is not None:
        cat = {}
        ok = True
        for out in OUTPUTS:
            fname = f"{CAT_PREFIX}{TAG[out]}.cbm"
            p = pfind(fname)
            if not p:
                ok = False
                health["missing"].append(fname)
                break
            try:
                mm = CatBoostRegressor()
                mm.load_model(str(p))
                cat[out] = mm
                health["loaded"].append(str(p))
            except Exception as e:
                ok = False
                health["missing"].append(f"{fname} (error: {e})")
                break
        if ok and len(cat) == 4:
            models["CatBoost"] = cat
    else:
        health["notes"].append("CatBoost not available -> CatBoost disabled.")

    # ------------------------------------------------------------
    # 🚀 SUB STEP 8.5: XGBOOST (4 SEPARATE MODELS)
    # ------------------------------------------------------------
    if xgb is not None:
        xb = {}
        ok = True
        for out in OUTPUTS:
            fname = f"{XGB_PREFIX}{TAG[out]}.json"
            p = pfind(fname)
            if not p:
                ok = False
                health["missing"].append(fname)
                break
            try:
                booster = xgb.Booster()
                booster.load_model(str(p))
                xb[out] = booster
                health["loaded"].append(str(p))
            except Exception as e:
                ok = False
                health["missing"].append(f"{fname} (error: {e})")
                break
        if ok and len(xb) == 4:
            models["XGBoost"] = xb
    else:
        health["notes"].append("XGBoost not available -> XGBoost disabled.")

    # ------------------------------------------------------------
    # 🚀 SUB STEP 8.6: LIGHTGBM (4 SEPARATE MODELS)
    # ------------------------------------------------------------
    if lgb is not None:
        lg = {}
        ok = True
        for out in OUTPUTS:
            fname = f"{LGB_PREFIX}{TAG[out]}.txt"   # EXACT in your repo
            p = pfind(fname)
            if not p:
                ok = False
                health["missing"].append(fname)
                break
            try:
                booster = lgb.Booster(model_file=str(p))
                lg[out] = booster
                health["loaded"].append(str(p))
            except Exception as e:
                ok = False
                health["missing"].append(f"{fname} (error: {e})")
                break
        if ok and len(lg) == 4:
            models["LightGBM"] = lg
    else:
        health["notes"].append("LightGBM not available -> LightGBM disabled.")

    return models, health


# ============================================================
# 🚀 STEP 9: PREDICTION FUNCTION (RETURN 4 OUTPUTS)
# ============================================================

def predict_outputs(model_key: str, X_df: pd.DataFrame, models: dict) -> dict:
    X_train = to_train_columns(X_df)
    X_np = X_train.values.astype(float)

    if model_key == "RF":
        y = np.array(models["RF"].predict(X_train)).reshape(1, -1)
        return {OUTPUTS[i]: float(y[0, i]) for i in range(4)}

    if model_key in ("ANN_MLP", "ANN_PS"):
        pack = models[model_key]
        Xs = pack["sx"].transform(X_np)
        yp = np.array(pack["model"].predict(Xs, verbose=0)).reshape(1, -1)
        y = pack["sy"].inverse_transform(yp)
        return {OUTPUTS[i]: float(y[0, i]) for i in range(4)}

    if model_key == "CatBoost":
        return {out: float(models["CatBoost"][out].predict(X_np)[0]) for out in OUTPUTS}

    if model_key == "XGBoost":
        dm = xgb.DMatrix(X_np)
        return {out: float(models["XGBoost"][out].predict(dm)[0]) for out in OUTPUTS}

    if model_key == "LightGBM":
        return {out: float(models["LightGBM"][out].predict(X_np)[0]) for out in OUTPUTS}

    raise ValueError("Unknown model key")


# ============================================================
# 🚀 STEP 10: BILINEAR CURVE PLOTTER
# ============================================================

def plot_bilinear(Dy, Fy, Du, Fu):
    """
    Bilinear curve: (0,0) -> (Dy,Fy) -> (Du,Fu)
    """
    x = [0.0, float(Dy), float(Du)]
    y = [0.0, float(Fy), float(Fu)]

    fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=200)
    ax.plot(x, y, marker="o", linewidth=2)

    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (kN)")
    ax.grid(True, alpha=0.3)

    ax.annotate("Yield (Dy,Fy)", (float(Dy), float(Fy)), textcoords="offset points", xytext=(8, 8))
    ax.annotate("Ultimate (Du,Fu)", (float(Du), float(Fu)), textcoords="offset points", xytext=(8, -12))
    return fig


# ============================================================
# 🚀 STEP 11: LOAD MODELS ONCE
# ============================================================

MODELS, HEALTH = load_all_models()


# ============================================================
# 🚀 STEP 12: HEADER UI (LOGO + SCHEMATIC)
# ============================================================

left, mid, right = st.columns([1, 2, 1], gap="large")

with left:
    logo = pfind("TJU logo.png")
    if logo:
        st.image(str(logo), use_container_width=True)

with mid:
    st.markdown(
        "<h2 style='text-align:center; margin-bottom: 0px;'>RC Shear Wall Bilinear Force–Displacement Curve</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; margin-top: 6px;'>Predict Dy, Fy, Du, Fu using trained ML models</p>",
        unsafe_allow_html=True
    )

with right:
    sch = pfind("RC shear wall schematic2.png")
    if sch:
        st.image(str(sch), use_container_width=True)


# ============================================================
# 🚀 STEP 13: MODEL LOAD STATUS BOX
# ============================================================

with st.expander("Model load status", expanded=False):
    st.write("✅ Loaded files:")
    st.write(HEALTH["loaded"] if HEALTH["loaded"] else "None")
    st.write("⚠️ Missing / errors:")
    st.write(HEALTH["missing"] if HEALTH["missing"] else "None")
    if HEALTH["notes"]:
        st.write("ℹ️ Notes:")
        st.write(HEALTH["notes"])

available = [k for k in ["RF", "ANN_MLP", "ANN_PS", "CatBoost", "XGBoost", "LightGBM"] if k in MODELS]
if not available:
    st.error("No models loaded. Check requirements.txt and file names in repo.")
    st.stop()


# ============================================================
# 🚀 STEP 14: SESSION STATE (HISTORY)
# ============================================================

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["Time", "Model"] + FEATURES + OUTPUTS)


# ============================================================
# 🚀 STEP 15: MAIN LAYOUT (INPUTS | OUTPUTS)
# ============================================================

colA, colB = st.columns([1.15, 1.0], gap="large")

with colA:
    st.subheader("Input parameters (theta removed)")
    model_choice = st.selectbox("Select ML model", available, index=0)

    user_vals = {}
    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]

    for i, f in enumerate(FEATURES):
        mn, mx = R.get(f, (0.0, 1.0))

        # keep step logic similar but simple
        span = float(mx - mn)
        step = 0.1 if span <= 20 else 1.0

        default = float(mn + 0.30 * span)

        with cols[i % 3]:
            user_vals[f] = st.number_input(
                f"{f}",
                min_value=float(mn),
                max_value=float(mx),
                value=float(default),
                step=float(step),
            )

    b1, b2 = st.columns([1, 1])
    calc = b1.button("Calculate", use_container_width=True)
    clear = b2.button("Clear history", use_container_width=True)


with colB:
    st.subheader("Predicted outputs + Bilinear curve")

    if clear:
        st.session_state.results_df = pd.DataFrame(columns=["Time", "Model"] + FEATURES + OUTPUTS)
        st.success("History cleared.")

    if calc:
        X_df = pd.DataFrame([user_vals], columns=FEATURES)
        try:
            yhat = predict_outputs(model_choice, X_df, MODELS)

            # show outputs
            out_table = pd.DataFrame(
                {"Output": OUTPUTS, "Predicted": [yhat[o] for o in OUTPUTS]}
            )
            st.dataframe(out_table, use_container_width=True, hide_index=True)

            # plot
            fig = plot_bilinear(
                Dy=yhat["Dy (mm)"],
                Fy=yhat["Fy (kN)"],
                Du=yhat["Du (mm)"],
                Fu=yhat["Fu (kN)"],
            )
            st.pyplot(fig, use_container_width=True)

            # save history
            row = {"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Model": model_choice}
            row.update(user_vals)
            row.update(yhat)

            st.session_state.results_df = pd.concat(
                [st.session_state.results_df, pd.DataFrame([row])],
                ignore_index=True
            )

            st.success("Prediction saved to history.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.markdown("---")
    st.subheader("History")
    st.dataframe(st.session_state.results_df, use_container_width=True)

    if len(st.session_state.results_df) > 0:
        csv_bytes = st.session_state.results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download history as CSV",
            data=csv_bytes,
            file_name="bilinear_predictions_history.csv",
            mime="text/csv",
            use_container_width=True
        )
