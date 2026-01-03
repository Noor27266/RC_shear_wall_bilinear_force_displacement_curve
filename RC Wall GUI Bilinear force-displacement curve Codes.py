DOC_NOTES = """
RC Shear Wall Bilinear Force–Displacement Curve Estimator — same logic/UI as your DI app
- Theta removed
- 4 outputs: Dy (mm), Fy (kN), Du (mm), Fu (kN)
"""


# =============================================================================
# 🚀 STEP 1: CORE IMPORTS & TENSORFLOW BACKEND SETUP
# =============================================================================

# =============================================================================
# 🚀 SUB STEP 1.1: ENVIRONMENT CONFIGURATION
# =============================================================================
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# =============================================================================
# 🚀 SUB STEP 1.2: CORE LIBRARY IMPORTS
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path
from glob import glob

# =============================================================================
# 🚀 SUB STEP 1.3: MACHINE LEARNING LIBRARY IMPORTS
# =============================================================================
import xgboost as xgb
import joblib
import catboost
import lightgbm as lgb

# =============================================================================
# 🚀 SUB STEP 1.4: KERAS COMPATIBILITY LOADER SETUP
# =============================================================================
try:
    from tensorflow.keras.models import load_model as _tf_load_model
except Exception:
    _tf_load_model = None
try:
    from keras.models import load_model as _k3_load_model   # works when keras==3 is present
except Exception:
    _k3_load_model = None


def _load_keras_model(path):
    """Try tf.keras first, then keras (Keras 3)."""
    errs = []
    if _tf_load_model is not None:
        try:
            return _tf_load_model(path)
        except Exception as e:
            errs.append(f"tf.keras: {e}")
    if _k3_load_model is not None:
        try:
            return _k3_load_model(path)
        except Exception as e:
            errs.append(f"keras: {e}")
    raise RuntimeError(" / ".join(errs) if errs else "No Keras loader available")


# =============================================================================
# 🚀 SUB STEP 1.5: SESSION STATE INITIALIZATION
# =============================================================================
st.session_state.setdefault("results_df", pd.DataFrame())


# =============================================================================
# 🔧 STEP 2: UTILITY FUNCTIONS & HELPER TOOLS
# =============================================================================

css = lambda s: st.markdown(s, unsafe_allow_html=True)


def b64(path: Path) -> str:
    # safe: return "" if not found
    try:
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        pass
    return ""


def dv(R, key, proposed):
    lo, hi = R[key]
    return float(max(lo, min(proposed, hi)))


# ---------- path helper ----------
BASE_DIR = Path(__file__).resolve().parent


def pfind(candidates, must_exist=True):
    """
    Find first existing file from candidates.
    must_exist=True  -> raise FileNotFoundError (use for models)
    must_exist=False -> return None (use for images)
    """
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    roots = [BASE_DIR, Path.cwd(), Path("/mnt/data")]
    for root in roots:
        if not root.exists():
            continue
        for c in candidates:
            p = root / c
            if p.exists():
                return p
    for root in [BASE_DIR, Path("/mnt/data")]:
        if not root.exists():
            continue
        for sub in root.iterdir():
            if sub.is_dir():
                for c in candidates:
                    p = sub / c
                    if p.exists():
                        return p
    pats = []
    for c in candidates:
        for root in [BASE_DIR, Path.cwd(), Path("/mnt/data")]:
            if root.exists():
                pats.append(str(root / "**" / c))
    for pat in pats:
        matches = glob(pat, recursive=True)
        if matches:
            return Path(matches[0])
    if must_exist:
        raise FileNotFoundError(f"None of these files were found: {candidates}")
    return None


# =============================================================================
# 🎨 STEP 3: STREAMLIT PAGE CONFIGURATION & UI STYLING
# =============================================================================

st.set_page_config(
    page_title="RC Shear Wall Bilinear Curve Estimator", layout="wide", page_icon="🧱"
)

st.markdown(
    """
<style>
html, body{
    margin:0 !important;
    padding:0 !important;
    overflow:hidden !important;
}
header[data-testid="stHeader"]{ height:0 !important; padding:0 !important; background:transparent !important; }
header[data-testid="stHeader"] *{ display:none !important; }

section.main > div.block-container{
    padding-top:0 !important;
    margin-top:-2.5rem !important;
}

.vega-embed, .vega-embed .chart-wrapper{
    max-width:100% !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# 🎨 SUB STEP 3.1: FONT SIZE SCALING CONFIGURATION
# =============================================================================
SCALE_UI = 0.36
s = lambda v: int(round(v * SCALE_UI))

FS_TITLE = s(20)
FS_SECTION = s(60)
FS_LABEL = s(50)
FS_UNITS = s(30)
FS_INPUT = s(30)
FS_SELECT = s(35)
FS_BUTTON = s(20)
FS_BADGE = s(30)
FS_RECENT = s(20)
INPUT_H = max(32, int(FS_INPUT * 2.0))

# =============================================================================
# 🎨 SUB STEP 3.2: COLOR SCHEME DEFINITION
# =============================================================================
DEFAULT_LOGO_H = 45
PRIMARY = "#8E44AD"
SECONDARY = "#f9f9f9"
INPUT_BG = "#ffffff"
INPUT_BORDER = "#e6e9f2"
LEFT_BG = "#e0e4ec"

# =============================================================================
# 🎨 STEP 3.3: COMPREHENSIVE CSS STYLING & THEME SETUP
# =============================================================================
css(
    f"""
<style>
  .block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 0.5rem !important;
    max-height: none !important;
    overflow: visible !important;
}}

  h1 {{
      font-size:{FS_TITLE}px !important;
      margin:0 rem 0 !important;
  }}

  .section-header {{
      font-size:{FS_SECTION}px !important;
      font-weight:700;
      margin:.35rem 0;
  }}

  .stNumberInput label,
  .stSelectbox label {{
      font-size:{FS_LABEL}px !important;
      font-weight:700;
  }}

  .stNumberInput label .katex,
  .stSelectbox label .katex {{
      font-size:{FS_LABEL}px !important;
      line-height:1.2 !important;
  }}

  .stNumberInput label .katex .fontsize-ensurer,
  .stSelectbox label .katex .fontsize-ensurer {{
      font-size:1em !important;
  }}

  .stNumberInput label .katex .mathrm,
  .stSelectbox label .katex .mathrm {{
      font-size:{FS_UNITS}px !important;
  }}

  div[data-testid="stNumberInput"] input[type="number"],
  div[data-testid="stNumberInput"] input[type="text"] {{
      font-size:{FS_INPUT}px !important;
      height:{INPUT_H}px !important;
      line-height:{INPUT_H - 8}px !important;
      font-weight:600 !important;
      padding:10px 12px !important;
  }}

  div[data-testid="stNumberInput"] [data-baseweb*="input"] {{
      background:{INPUT_BG} !important;
      border:1px solid {INPUT_BORDER} !important;
      border-radius:12px !important;
      box-shadow:0 1px 2px rgba(16,24,40,.06) !important;
      transition:border-color .15s ease, box-shadow .15s ease !important;
  }}

  div[data-testid="stNumberInput"] button {{
      background:#ffffff !important;
      border:1px solid {INPUT_BORDER} !important;
      border-radius:10px !important;
      box-shadow:0 1px 1px rgba(16,24,40,.05) !important;
  }}

  .stSelectbox [role="combobox"],
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:first-child,
  div[data-testid="stSelectbox"] div[role="listbox"],
  div[data-testid="stSelectbox"] div[role="option"] {{
      font-size:{FS_SELECT}px !important;
  }}

  div.stButton > button {{
      font-size:{FS_BUTTON}px !important;
      height:{max(42, int(round(FS_BUTTON*1.45)))}px !important;
      line-height:{max(36, int(round(FS_BUTTON*1.15)))}px !important;
      color:#fff !important;
      font-weight:700;
      border:none !important;
      border-radius:8px !important;
      background:#4CAF50 !important;
  }}

  button[key="reset_btn"] {{
      background:#2196F3 !important;
  }}

  button[key="clear_btn"] {{
      background:#f44336 !important;
  }}

  #compact-form {{
      max-width:900px;
      margin:0 auto;
  }}

  html, body, #root, .stApp, section.main, .block-container, [data-testid="stAppViewContainer"] {{
      background: linear-gradient(90deg, #e0e4ec 60%, transparent 60%) !important;
      min-height: 100vh !important;
      height: auto !important;
      overflow:hidden !important;
  }}

  [data-testid="column"]:first-child {{
      min-height: 100vh !important;
      background: #e0e4ec !important;
      margin-top: 0px !important;
      padding-top: 0px !important;
  }}
</style>
"""
)

# =============================================================================
# 🏷️ LOGO LOADING FOR LEFT PANEL  (UNCHANGED STYLE)
# =============================================================================
# keep same behavior but make it safe: try TJU logo, else logo2-01
_logo_file = pfind(["TJU logo.png", "logo2-01.png"], must_exist=False)
_b64 = b64(_logo_file) if _logo_file else ""


# =============================================================================
# 🤖 STEP 5: MACHINE LEARNING MODEL LOADING & HEALTH CHECKING
# =============================================================================

def record_health(name, ok, msg=""):
    health.append((name, ok, msg, "ok" if ok else "err"))

health = []

# 4-output definition + tags (match your filenames)
OUTPUTS = ["Dy (mm)", "Fy (kN)", "Du (mm)", "Fu (kN)"]
TAG = {
    "Dy (mm)": "Δymm",
    "Fy (kN)": "FykN",
    "Du (mm)": "Δmmm",
    "Fu (kN)": "FmkN",
}

class _ScalerShim:
    """Wrapper to keep X / y scalers together for ANN models (4 outputs)."""
    def __init__(self, X_scaler, Y_scaler):
        self.Xs = X_scaler
        self.Ys = Y_scaler

    def transform_X(self, X):
        return self.Xs.transform(X)

    def inverse_transform_y(self, y):
        y = np.array(y)
        y = y.reshape(1, -1) if y.ndim == 1 else y
        return self.Ys.inverse_transform(y)


# ---------------------------- PS (ANN) MULTI-OUTPUT --------------------------
ann_ps_model = None
ann_ps_proc  = None
try:
    ps_model_path = pfind(["ANN_PS_Model.keras", "ANN_PS_Model.h5"])
    ann_ps_model  = _load_keras_model(ps_model_path)

    sx = joblib.load(pfind(["ANN_PS_Scaler_X.save", "ANN_PS_Scaler_X.pkl", "ANN_PS_Scaler_X.joblib"]))
    sy = joblib.load(pfind(["ANN_PS_Scaler_y.save", "ANN_PS_Scaler_y.pkl", "ANN_PS_Scaler_y.joblib"]))

    ann_ps_proc = _ScalerShim(sx, sy)
    record_health("PS (ANN)", True, f"loaded from {ps_model_path}")
except Exception as e:
    record_health("PS (ANN)", False, f"{e}")

# ---------------------------- MLP (ANN) MULTI-OUTPUT -------------------------
ann_mlp_model = None
ann_mlp_proc  = None
try:
    mlp_model_path = pfind(["ANN_MLP_Model.keras", "ANN_MLP_Model.h5"])
    ann_mlp_model  = _load_keras_model(mlp_model_path)

    sx = joblib.load(pfind(["ANN_MLP_Scaler_X.save", "ANN_MLP_Scaler_X.pkl", "ANN_MLP_Scaler_X.joblib"]))
    sy = joblib.load(pfind(["ANN_MLP_Scaler_y.save", "ANN_MLP_Scaler_y.pkl", "ANN_MLP_Scaler_y.joblib"]))

    ann_mlp_proc = _ScalerShim(sx, sy)
    record_health("MLP (ANN)", True, f"loaded from {mlp_model_path}")
except Exception as e:
    record_health("MLP (ANN)", False, f"{e}")

# ---------------------------- Random Forest (MULTI-OUTPUT) -------------------
rf_model = None
try:
    rf_path = pfind([
        "Best_RF_Model.pkl",
        "random_forest_model.pkl",
        "random_forest_model.joblib",
        "rf_model.pkl",
        "RF_model.pkl",
    ])
    rf_model = joblib.load(rf_path)
    record_health("Random Forest", True, f"loaded with joblib from {rf_path}")
except Exception as e:
    record_health("Random Forest", False, str(e))

# ---------------------------- XGBoost (4 boosters) ---------------------------
xgb_models = None
try:
    xgb_models = {}
    for out in OUTPUTS:
        fn = f"Best_XGBoost_{TAG[out]}.json"
        p = pfind([fn])
        booster = xgb.Booster()
        booster.load_model(str(p))
        xgb_models[out] = booster
    record_health("XGBoost", True, "loaded 4 output models")
except Exception as e:
    xgb_models = None
    record_health("XGBoost", False, str(e))

# ---------------------------- CatBoost (4 models) ----------------------------
cat_models = None
try:
    cat_models = {}
    for out in OUTPUTS:
        fn = f"Best_CatBoost_{TAG[out]}.cbm"
        p = pfind([fn])
        m = catboost.CatBoostRegressor()
        m.load_model(str(p))
        cat_models[out] = m
    record_health("CatBoost", True, "loaded 4 output models")
except Exception as e:
    cat_models = None
    record_health("CatBoost", False, str(e))

# ---------------------------- LightGBM (4 boosters) --------------------------
lgb_models = None
try:
    lgb_models = {}
    for out in OUTPUTS:
        fn = f"Best_LightGBM_{TAG[out]}.txt"
        p = pfind([fn])
        booster = lgb.Booster(model_file=str(p))
        lgb_models[out] = booster
    record_health("LightGBM", True, "loaded 4 output models")
except Exception as e:
    lgb_models = None
    record_health("LightGBM", False, str(e))

# ---------------------------- Registry ---------------------------------------
model_registry = {}
if cat_models is not None: model_registry["CatBoost"] = cat_models
if xgb_models is not None: model_registry["XGBoost"] = xgb_models
if lgb_models is not None: model_registry["LightGBM"] = lgb_models
if ann_mlp_model is not None and ann_mlp_proc is not None: model_registry["MLP"] = ann_mlp_model
if ann_ps_model is not None and ann_ps_proc is not None: model_registry["PS"] = ann_ps_model
if rf_model is not None: model_registry["Random Forest"] = rf_model

MODEL_ORDER = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
LABEL_TO_KEY = {"RF": "Random Forest"}


# =============================================================================
# 📊 STEP 6: INPUT PARAMETERS & DATA RANGES DEFINITION  (THETA REMOVED ONLY)
# =============================================================================
R = {
    "lw": (400.0, 3500.0),
    "hw": (495.0, 5486.4),
    "tw": (26.0, 305.0),
    "fc": (13.38, 93.6),
    "fyt": (0.0, 1187.0),
    "fysh": (0.0, 1375.0),
    "fyl": (160.0, 1000.0),
    "fybl": (0.0, 900.0),
    "rt": (0.000545, 0.025139),
    "rsh": (0.0, 0.041888),
    "rl": (0.0, 0.029089),
    "rbl": (0.0, 0.031438),
    "axial": (0.0, 0.86),
    "b0": (45.0, 3045.0),
    "db": (0.0, 500.0),
    "s_db": (0.0, 47.65625),
    "AR": (0.388889, 5.833333),
    "M_Vlw": (0.388889, 4.1),
}

U = lambda s: rf"\;(\mathrm{{{s}}})"

GEOM = [
    (rf"$l_w{U('mm')}$", "lw", 1000.0, 1.0, None, "Length"),
    (rf"$h_w{U('mm')}$", "hw", 495.0, 1.0, None, "Height"),
    (rf"$t_w{U('mm')}$", "tw", 200.0, 1.0, None, "Thickness"),
    (rf"$b_0{U('mm')}$", "b0", 200.0, 1.0, None, "Boundary element width"),
    (rf"$d_b{U('mm')}$", "db", 400.0, 1.0, None, "Boundary element length"),
    (r"$AR$", "AR", 2.0, 0.01, None, "Aspect ratio"),
    
]

MATS = [
    (rf"$f'_c{U('MPa')}$", "fc", 40.0, 0.1, None, "Concrete strength"),
    (rf"$f_{{yt}}{U('MPa')}$", "fyt", 400.0, 1.0, None, "Transverse web yield strength"),
    (rf"$f_{{ysh}}{U('MPa')}$", "fysh", 400.0, 1.0, None, "Transverse boundary yield strength"),
    (rf"$f_{{yl}}{U('MPa')}$", "fyl", 400.0, 1.0, None, "Vertical web yield strength"),
    (rf"$f_{{ybl}}{U('MPa')}$", "fybl", 400.0, 1.0, None, "Vertical boundary yield strength"),
    (r"$M/(V_{l_w})$", "M_Vlw", 2.0, 0.01, None, "Shear span ratio"),
]

# theta row REMOVED only
REINF = [
    (r"$\rho_t\;(\%)$", "rt", 0.25, 0.0001, "%.6f", "Transverse web ratio"),
    (r"$\rho_{sh}\;(\%)$", "rsh", 0.25, 0.0001, "%.6f", "Transverse boundary ratio"),
    (r"$\rho_l\;(\%)$", "rl", 0.25, 0.0001, "%.6f", "Vertical web ratio"),
    (r"$\rho_{bl}\;(\%)$", "rbl", 0.25, 0.0001, "%.6f", "Vertical boundary ratio"),
    (r"$s/d_b$", "s_db", 0.25, 0.01, None, "Hoop spacing ratio"),
    (r"$P/(A_g f'_c)$", "axial", 0.10, 0.001, None, "Axial Load Ratio"),
]

def num(label, key, default, step, fmt, help_):
    return st.number_input(
        label,
        value=dv(R, key, default),
        step=step,
        min_value=R[key][0],
        max_value=R[key][1],
        format=fmt if fmt else None,
        help=help_,
    )

# Hide +/- buttons
css(
    """
<style>
div[data-testid="stNumberInput"] button {
    display: none !important;
}
</style>
"""
)

# =============================================================================
# 📊 SUB STEP 6.1: LAYOUT COLUMNS SETUP
# =============================================================================
left, right = st.columns([1.5, 1], gap="large")

# =============================================================================
# 📊 SUB STEP 6.2: LEFT PANEL CONTENT IMPLEMENTATION (UNCHANGED UI)
# =============================================================================
with left:
    if _b64:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2px;">
            <img src="data:image/png;base64,{_b64}" 
                 style="height: 90px; width: auto;" 
                 alt="Logo" />
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0px; margin: 0; padding: 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:transparent; border-radius:12px; padding:0px; margin:0 0 3px 0; box-shadow:none;">
        <div style="text-align:center; font-size:25px; font-weight:600; color:#333; margin:0; padding:0;">
            RC Shear Wall Bilinear Force–Displacement Curve
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin: 0 0 0 0; padding: 0;">
        <div class='form-banner'>Inputs Features</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1], gap="small")

    with c1:
        st.markdown("<div class='section-header'>Geometry </div>", unsafe_allow_html=True)
        lw, hw, tw, b0, db, AR, M_Vlw = [num(*row) for row in GEOM]

    with c2:
        st.markdown("<div class='section-header'>Reinf. Ratios </div>", unsafe_allow_html=True)
        rt, rsh, rl, rbl, s_db, axial = [num(*row) for row in REINF]  # theta removed

    with c3:
        st.markdown("<div class='section-header'>Material Strengths</div>", unsafe_allow_html=True)
        fc, fyt, fysh = [num(*row) for row in MATS[:3]]
        fyl, fybl = [num(*row) for row in MATS[3:]]

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 🎮 STEP 7: RIGHT PANEL - CONTROLS & INTERACTION ELEMENTS (UNCHANGED UI)
# =============================================================================
SCHEM_BOX_H    = 300
SCHEM_IMG_H    = 480
SCHEM_OFFSET_X = 80
SCHEM_OFFSET_Y = -40

SCHEM2_IMG_H   = 480
SCHEM2_OFFSET_X = 500
SCHEM2_OFFSET_Y = -40

CHART_W = 350

with right:

    # --- TWO schematics side by side in fixed-height box ---
    # IMPORTANT FIX: use safe pfind and b64 so Streamlit Cloud won't crash
    img1 = pfind(["logo2-01.png", "TJU logo.png"], must_exist=False)
    img2 = pfind(["RC shear wall schematic2.png"], must_exist=False)

    img1_b64 = b64(img1) if img1 else ""
    img2_b64 = b64(img2) if img2 else ""

    st.markdown(
        f"""
        <div style="position:relative; height:{SCHEM_BOX_H}px; margin-bottom:0;">
            <!-- First schematic -->
            <img src="data:image/png;base64,{img1_b64}"
                 style="
                    position:absolute;
                    left:{SCHEM_OFFSET_X}px;
                    top:{SCHEM_OFFSET_Y}px;
                    height:{SCHEM_IMG_H}px;
                    width:auto;
                 " />
            <!-- Second schematic -->
            <img src="data:image/png;base64,{img2_b64}"
                 style="
                    position:absolute;
                    left:{SCHEM2_OFFSET_X}px;
                    top:{SCHEM2_OFFSET_Y}px;
                    height:{SCHEM2_IMG_H}px;
                    width:auto;
                 " />
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_plot, col_controls = st.columns([3, 1])

    # =============================================================================
    # ⭐ SUB-STEP 7.1 — OUTPUT AREA (LEFT SIDE) (SAME SLOT AS YOUR DI PLOT)
    # =============================================================================
    with col_plot:
        chart_slot = st.empty()

    # =============================================================================
    # ⭐ SUB-STEP 7.2 — MODEL SELECTION + BUTTONS (RIGHT SIDE) - UNCHANGED
    # =============================================================================
    with col_controls:

        available = set(model_registry.keys())
        ordered_keys = [m for m in MODEL_ORDER if m in available] or ["(no models loaded)"]
        display_labels = ["RF" if m == "Random Forest" else m for m in ordered_keys]

        model_choice_label = st.selectbox(
            "Model Selection",
            display_labels,
            key="model_select_compact",
        )
        model_choice = LABEL_TO_KEY.get(model_choice_label, model_choice_label)

        st.session_state["model_choice"] = model_choice

        if "do_calculation" not in st.session_state:
            st.session_state.do_calculation = False

        if st.button("Calculate", key="calc_btn", use_container_width=True):
            st.session_state.do_calculation = True
            st.rerun()

        if st.button("Reset", key="reset_btn", use_container_width=True):
            st.session_state.do_calculation = False
            st.rerun()

        if st.button("Clear All", key="clear_btn", use_container_width=True):
            st.session_state.results_df = pd.DataFrame()
            st.session_state.do_calculation = False
            st.success("All predictions cleared!")
            st.rerun()

        if not st.session_state.results_df.empty:
            csv = st.session_state.results_df.to_csv(index=False)
            st.download_button(
                "📂 Download as CSV",
                data=csv,
                file_name="bilinear_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_csv",
            )

css("""
<style>
div[data-testid="stSelectbox"],
div.stButton,
div[data-testid="stDownloadButton"] {
    position: relative !important;
    top: 165px !important;
    left: 30px !important;
    margin-bottom: 10px !important;
    height: auto !important;
    margin-right: 20px !important;
}

div[data-testid="column"]:nth-child(2) {
    margin-top: -30px !important;
    padding-right: 20px !important;
}

div[data-testid="column"]:nth-child(2) > div:nth-child(2) {
    padding-top: 10px !important;
    padding-right: 20px !important;
}
</style>
""")


# =============================================================================
# ⚡ STEP 8: BILINEAR PREDICTION + OUTPUT DISPLAY (THETA REMOVED, 4 OUTPUTS)
# =============================================================================

# --- training name map (same style as your DI code, but θ removed)
_TRAIN_NAME_MAP = {
    "l_w": "lw",
    "h_w": "hw",
    "t_w": "tw",
    "f′c": "fc",
    "fyt": "fyt",
    "fysh": "fysh",
    "fyl": "fyl",
    "fybl": "fybl",
    "ρt": "pt",
    "ρsh": "psh",
    "ρl": "pl",
    "ρbl": "pbl",
    "P/(Agf′c)": "P/(Agfc)",
    "b0": "b0",
    "db": "db",
    "s/db": "s/db",
    "AR": "AR",
    "M/Vlw": "M/Vlw",
}

_TRAIN_COL_ORDER = [
    "lw","hw","tw","fc","fyt","fysh","fyl","fybl",
    "pt","psh","pl","pbl","P/(Agfc)","b0","db","s/db",
    "AR","M/Vlw",
]

def _df_in_train_order(df):
    return df.rename(columns=_TRAIN_NAME_MAP).reindex(columns=_TRAIN_COL_ORDER)

def _make_input_df(lw,hw,tw,fc,fyt,fysh,fyl,fybl,rt,rsh,rl,rbl,axial,b0,db,s_db,AR,M_Vlw):
    cols = ["l_w","h_w","t_w","f′c","fyt","fysh","fyl","fybl","ρt","ρsh","ρl","ρbl",
            "P/(Agf′c)","b0","db","s/db","AR","M/Vlw"]
    vals = [lw,hw,tw,fc,fyt,fysh,fyl,fybl,rt,rsh,rl,rbl,axial,b0,db,s_db,AR,M_Vlw]
    return pd.DataFrame([vals], columns=cols)

def predict_4(choice, input_df):
    df_trees = _df_in_train_order(input_df).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    X = df_trees.values.astype(np.float32)

    # ----- 4-output dict -----
    if choice == "LightGBM":
        return {out: float(model_registry["LightGBM"][out].predict(X)[0]) for out in OUTPUTS}

    if choice == "XGBoost":
        dm = xgb.DMatrix(X)
        return {out: float(model_registry["XGBoost"][out].predict(dm)[0]) for out in OUTPUTS}

    if choice == "CatBoost":
        return {out: float(model_registry["CatBoost"][out].predict(X)[0]) for out in OUTPUTS}

    if choice == "Random Forest":
        pred = np.array(model_registry["Random Forest"].predict(df_trees))
        pred = pred.reshape(1, -1)
        return {OUTPUTS[i]: float(pred[0, i]) for i in range(4)}

    if choice == "PS":
        Xn = ann_ps_proc.transform_X(X)
        try:
            yhat = model_registry["PS"].predict(Xn, verbose=0)
        except Exception:
            model_registry["PS"].compile(optimizer="adam", loss="mse")
            yhat = model_registry["PS"].predict(Xn, verbose=0)
        y = ann_ps_proc.inverse_transform_y(yhat)[0]
        return {OUTPUTS[i]: float(y[i]) for i in range(4)}

    if choice == "MLP":
        Xn = ann_mlp_proc.transform_X(X)
        try:
            yhat = model_registry["MLP"].predict(Xn, verbose=0)
        except Exception:
            model_registry["MLP"].compile(optimizer="adam", loss="mse")
            yhat = model_registry["MLP"].predict(Xn, verbose=0)
        y = ann_mlp_proc.inverse_transform_y(yhat)[0]
        return {OUTPUTS[i]: float(y[i]) for i in range(4)}

    return {out: 0.0 for out in OUTPUTS}

def plot_bilinear(Dy, Fy, Du, Fu):
    import matplotlib.pyplot as plt
    x = [0.0, float(Dy), float(Du)]
    y = [0.0, float(Fy), float(Fu)]
    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=200)
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (kN)")
    ax.grid(True, alpha=0.3)
    return fig

# =============================================================================
# MAIN PREDICTION LOGIC (SAME PATTERN AS YOUR DI)
# =============================================================================

model_choice = st.session_state.get("model_choice", None)
if not model_choice:
    for m in MODEL_ORDER:
        if m in model_registry:
            model_choice = m
            break

if st.session_state.get("do_calculation", False) and model_choice and model_choice in model_registry:
    xdf = _make_input_df(
        lw, hw, tw, fc, fyt, fysh, fyl, fybl,
        rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw
    )

    try:
        pred4 = predict_4(model_choice, xdf)

        row = xdf.copy()
        for k in OUTPUTS:
            row[k] = pred4[k]

        st.session_state.results_df = pd.concat([st.session_state.results_df, row], ignore_index=True)
        st.session_state.do_calculation = False

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.session_state.do_calculation = False

# Always display outputs if we have results (use same chart_slot area)
if not st.session_state.results_df.empty:
    last = st.session_state.results_df.iloc[-1]

    Dy = float(last["Dy (mm)"])
    Fy = float(last["Fy (kN)"])
    Du = float(last["Du (mm)"])
    Fu = float(last["Fu (kN)"])

    with chart_slot.container():
        st.markdown("<div style='margin-top:150px;'>", unsafe_allow_html=True)

        out_df = pd.DataFrame({
            "Output": OUTPUTS,
            "Predicted": [Dy, Fy, Du, Fu]
        })
        st.dataframe(out_df, use_container_width=True, hide_index=True)

        fig = plot_bilinear(Dy, Fy, Du, Fu)
        st.pyplot(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 🎨 STEP 9: FINAL UI POLISH & BANNER STYLING (UNCHANGED)
# =============================================================================
st.markdown(
    """
<style>
.form-banner{
  background: linear-gradient(90deg, #0E9F6E, #84CC16) !important;
  color: #fff !important;
  text-align: center !important;
  border-radius: 10px !important;
  padding: .25rem .55rem !important;
  margin-top: 2px !important;
  margin-bottom: 10px !important;
  transform: translateY(0) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

