import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st
import numpy as np
import json
from PIL import Image
import time

try:
    import keras
except ImportError:
    import tensorflow as tf
    keras = tf.keras

import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { background: #06090a; color: #e2f0e2; font-family: 'DM Sans', sans-serif; }
h1,h2,h3,h4 { font-family: 'Syne', sans-serif; }

/* ════════════ SIDEBAR ════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050c05 0%, #040804 50%, #060b08 100%) !important;
    border-right: 1px solid #0f1f0f;
    width: 270px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

/* kill default streamlit padding inside sidebar */
section[data-testid="stSidebar"] .block-container { padding: 0 !important; }

/* sidebar radio group → styled nav list */
section[data-testid="stSidebar"] .stRadio { padding: 0 14px !important; }

/* hide the widget's own (visually-collapsed) label completely */
section[data-testid="stSidebar"] .stRadio > label { display: none !important; }

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
    gap: 5px !important;
    flex-direction: column !important;
}

/* each nav row */
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #6b9e6b !important;
    padding: 11px 16px !important;
    margin: 0 !important;
    border-radius: 10px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.18s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 11px !important;
    cursor: pointer !important;
    width: 100% !important;
}

/* completely remove the native radio dot/circle marker */
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

/* the text portion of the label */
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p {
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    margin: 0 !important;
}

/* hover state — unselected items only */
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
    background: rgba(74,222,74,0.08) !important;
    border-color: rgba(74,222,74,0.22) !important;
    color: #bbf7d0 !important;
}

/* selected nav item — strong, unmistakable highlight */
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(135deg, rgba(74,222,74,0.16), rgba(74,222,74,0.06)) !important;
    border-color: #4ade80 !important;
    color: #eafff0 !important;
    box-shadow: inset 0 0 0 1px rgba(74,222,74,0.15), 0 4px 14px rgba(74,222,74,0.12) !important;
}
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) p {
    color: #eafff0 !important;
    font-weight: 700 !important;
}

/* "Navigation" eyebrow + section labels we author ourselves keep their own colors */
section[data-testid="stSidebar"] { color: #6b9e6b; }

/* slider */
section[data-testid="stSidebar"] .stSlider { padding: 0 18px !important; }

/* ════════════ MAIN CONTENT ════════════ */
.main .block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1200px; }

/* ── Hero ── */
.hero {
    border-radius: 24px; padding: 64px 48px; text-align: center;
    position: relative; overflow: hidden; margin-bottom: 2.5rem;
    background: linear-gradient(135deg, #081408 0%, #0f2e14 50%, #091f12 100%);
    border: 1px solid #1a3a1a;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 50% at 50% 0%, rgba(74,222,74,0.09) 0%, transparent 70%);
    animation: glow 5s ease-in-out infinite;
}
@keyframes glow { 0%,100%{opacity:.6} 50%{opacity:1} }
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(74,222,74,0.1); border: 1px solid rgba(74,222,74,0.25);
    border-radius: 50px; padding: 6px 16px; font-size: 0.72rem;
    color: #4ade80; letter-spacing: 3px; text-transform: uppercase;
    font-weight: 600; margin-bottom: 24px;
}
.hero-title {
    font-size: 3.8rem; font-weight: 800; line-height: 1.05;
    background: linear-gradient(135deg, #4ade80 0%, #86efac 50%, #bbf7d0 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 16px;
}
.hero-sub { font-size: 1.1rem; color: #6b9e6b; font-weight: 300; line-height: 1.6; max-width: 560px; margin: 0 auto; }

/* ── Stat bar ── */
.stat-bar { display: flex; gap: 12px; margin: 2rem 0; }
.stat-chip {
    flex: 1; background: #0c1a0c; border: 1px solid #182c18;
    border-radius: 14px; padding: 20px 14px; text-align: center; transition: 0.25s;
}
.stat-chip:hover { border-color: #4ade80; transform: translateY(-2px); }
.stat-val { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #4ade80; line-height: 1; }
.stat-lbl { font-size: 0.75rem; color: #4a7a4a; margin-top: 6px; }

/* ── Choice cards (the 2 big options on Analyse page) ── */
.choice-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 2rem 0 0; }
.choice-card {
    border-radius: 22px 22px 0 0; padding: 44px 32px 28px; text-align: center;
    transition: all 0.3s; position: relative; overflow: hidden;
    border-bottom: none;
}
.choice-card::before {
    content: ''; position: absolute; inset: 0; opacity: 0;
    transition: opacity 0.3s;
}
.choice-card:hover::before { opacity: 1; }

.choice-disease {
    background: linear-gradient(145deg, #091a09, #0f2e14);
    border: 2px solid #1a3a1a;
}
.choice-disease::before { background: radial-gradient(ellipse at top, rgba(74,222,74,0.08), transparent 70%); }
.choice-disease:hover { border-color: #4ade80; }

.choice-pest {
    background: linear-gradient(145deg, #160901, #2a1003);
    border: 2px solid #3a1a06;
}
.choice-pest::before { background: radial-gradient(ellipse at top, rgba(251,146,60,0.1), transparent 70%); }
.choice-pest:hover { border-color: #fb923c; }

.choice-icon { font-size: 4rem; margin-bottom: 20px; display: block; }
.choice-title-d { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; color: #4ade80; margin-bottom: 10px; }
.choice-title-p { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; color: #fb923c; margin-bottom: 10px; }
.choice-desc { font-size: 0.88rem; color: #9fc99f; line-height: 1.6; margin-bottom: 20px; }
.choice-pest .choice-desc { color: #d9b48f; }
.choice-pill-d {
    display: inline-block; background: rgba(74,222,74,0.12);
    border: 1px solid rgba(74,222,74,0.3); border-radius: 50px;
    padding: 5px 14px; font-size: 0.75rem; color: #86efac; font-weight: 600;
}
.choice-pill-p {
    display: inline-block; background: rgba(251,146,60,0.12);
    border: 1px solid rgba(251,146,60,0.3); border-radius: 50px;
    padding: 5px 14px; font-size: 0.75rem; color: #fed7aa; font-weight: 600;
}
.choice-arrow { font-size: 0.78rem; margin-top: 16px; display: block; color: #4a7a4a; letter-spacing: 0.3px; }
.choice-pest .choice-arrow { color: #8a5a2a; }

/* dock the action button flush under its card, same width, rounded bottom only */
.btn-disease, .btn-pest { margin-top: -1px; }
.btn-disease .stButton > button, .btn-pest .stButton > button {
    border-radius: 0 0 22px 22px !important;
    padding: 16px 22px !important;
}
.btn-disease .stButton > button { border-top: 2px solid #1a3a1a !important; }
.btn-pest .stButton > button { border-top: 2px solid #3a1a06 !important; }

/* ── Back button header ── */
.mode-header {
    display: flex; align-items: center; gap: 14px; margin-bottom: 2rem;
    padding: 16px 22px; border-radius: 14px;
}
.mode-header-d { background: #091a09; border: 1px solid #1e3d1e; }
.mode-header-p { background: #160901; border: 1px solid #3a1a06; }
.mode-header-icon { font-size: 2rem; }
.mode-header-label { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; }
.mode-header-label-d { color: #4ade80; }
.mode-header-label-p { color: #fb923c; }
.mode-header-sub { font-size: 0.8rem; color: #6b8a6b; margin-top: 2px; }

/* ── Upload panel ── */
.panel-d { background: #091a09; border: 1px solid #1e3d1e; border-radius: 18px; padding: 24px; }
.panel-p { background: #160901; border: 1px solid #3a1a06; border-radius: 18px; padding: 24px; }
.panel-label { font-family: 'Syne', sans-serif; font-size: 0.85rem; font-weight: 700; margin-bottom: 10px; }
.panel-label-d { color: #86efac; }
.panel-label-p { color: #fbbf24; }

/* ── File tag ── */
.ftag { display: inline-block; border-radius: 8px; padding: 5px 12px; font-size: 0.8rem; margin: 3px; }
.ftag-d { background: rgba(74,222,74,0.08); border: 1px solid rgba(74,222,74,0.2); color: #86efac; }
.ftag-p { background: rgba(251,146,60,0.08); border: 1px solid rgba(251,146,60,0.2); color: #fbbf24; }

/* ── Tips ── */
.tips { border-radius: 12px; padding: 16px 18px; margin-top: 18px; font-size: 0.84rem; line-height: 1.9; }
.tips-d { background: #0c1a0c; border: 1px solid #182c18; color: #6b9e6b; }
.tips-p { background: #160a01; border: 1px solid #2c1803; color: #9e7a3a; }
.tips-title { font-family: 'Syne', sans-serif; font-size: 0.82rem; font-weight: 700; margin-bottom: 8px; }
.tips-title-d { color: #4ade80; }
.tips-title-p { color: #fb923c; }

/* ── Result card ── */
.rcard {
    border-radius: 20px; padding: 32px; text-align: center;
    margin-top: 1.5rem; position: relative; overflow: hidden;
}
.rcard-d { background: linear-gradient(145deg, #091a09, #0c2e0e); border: 1px solid #4ade80; }
.rcard-p { background: linear-gradient(145deg, #160901, #2a1003); border: 1px solid #fb923c; }
.rcard::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background-size: 200%; animation: shimmer 2.5s linear infinite;
}
.rcard-d::after { background: linear-gradient(90deg, #4ade80, #86efac, #4ade80); }
.rcard-p::after { background: linear-gradient(90deg, #fb923c, #fbbf24, #fb923c); }
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
.rcard-eyebrow { font-size: 0.72rem; letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 10px; }
.rcard-eyebrow-d { color: #4a7a4a; }
.rcard-eyebrow-p { color: #7a5020; }
.rcard-plant { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; }
.rcard-plant-d { color: #4ade80; }
.rcard-plant-p { color: #fb923c; }
.rcard-cond { font-size: 1.1rem; font-weight: 500; margin-top: 6px; }
.rcard-cond-d { color: #bbf7d0; }
.rcard-cond-p { color: #fed7aa; }

/* ── Confidence bar ── */
.cbar-wrap { border-radius: 50px; height: 10px; margin: 14px 0 6px; overflow: hidden; }
.cbar-wrap-d { background: #1a3a1a; }
.cbar-wrap-p { background: #3a1a05; }
.cbar-fill { height: 100%; border-radius: 50px; }
.cbar-fill-d { background: linear-gradient(90deg, #4ade80, #86efac); }
.cbar-fill-p { background: linear-gradient(90deg, #fb923c, #fbbf24); }

/* ── Alerts ── */
.alert { border-radius: 12px; padding: 14px 18px; font-size: 0.92rem; margin-top: 14px; }
.alert-ok  { background: rgba(74,222,74,0.07);  border: 1px solid rgba(74,222,74,0.25);  color: #86efac; }
.alert-bad { background: rgba(251,146,60,0.07); border: 1px solid rgba(251,146,60,0.25); color: #fed7aa; }

/* ── Rank bar row ── */
.rank-row { margin-bottom: 14px; }
.rank-meta { display: flex; justify-content: space-between; font-size: 0.83rem; margin-bottom: 4px; }

/* ── Section title ── */
.stitle { font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700;
          letter-spacing: 1.5px; text-transform: uppercase; margin: 1.8rem 0 1rem;
          padding-bottom: 8px; }
.stitle-d { color: #4ade80; border-bottom: 1px solid #1a3a1a; }
.stitle-p { color: #fb923c; border-bottom: 1px solid #3a1a06; }

/* ── Placeholder ── */
.placeholder {
    height: 380px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; border-radius: 18px;
    border: 2px dashed;
}
.placeholder-d { background: #091a09; border-color: #1a3a1a; }
.placeholder-p { background: #160901; border-color: #3a1a06; }

/* ── Feature grid (Home) ── */
.fgrid { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin-top: 1.2rem; }
.fcard { border-radius: 14px; padding: 22px 18px; transition: 0.2s; }
.fcard-d { background: #0c1a0c; border: 1px solid #182c18; }
.fcard-d:hover { border-color: #4ade80; transform: translateY(-3px); }
.fcard-p { background: #160a01; border: 1px solid #2c1803; }
.fcard-p:hover { border-color: #fb923c; transform: translateY(-3px); }
.fcard-icon { font-size: 1.8rem; margin-bottom: 10px; }
.fcard-title { font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700; margin-bottom: 6px; }
.fcard-title-d { color: #bbf7d0; }
.fcard-title-p { color: #fed7aa; }
.fcard-desc { font-size: 0.82rem; line-height: 1.55; }
.fcard-desc-d { color: #4a7a4a; }
.fcard-desc-p { color: #7a5020; }

/* ── Step row ── */
.steps { display: flex; gap: 10px; margin: 1.2rem 0; }
.step { flex: 1; text-align: center; }
.step-num { width: 36px; height: 36px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            margin: 0 auto 8px; font-family: 'Syne', sans-serif;
            font-weight: 800; font-size: 0.88rem; color: #06090a; }
.step-num-d { background: linear-gradient(135deg, #4ade80, #16a34a); }
.step-num-p { background: linear-gradient(135deg, #fb923c, #ea580c); }
.step-txt { font-size: 0.78rem; }
.step-txt-d { color: #6b9e6b; }
.step-txt-p { color: #9e7a3a; }

/* ── Info cards (About) ── */
.icard { border-radius: 14px; padding: 20px; margin: 10px 0; }
.icard-d { background: #0c1a0c; border: 1px solid #182c18; border-left: 4px solid #4ade80; }
.icard-p { background: #160a01; border: 1px solid #2c1803; border-left: 4px solid #fb923c; }
.icard h4 { font-family: 'Syne', sans-serif; font-size: 0.95rem; margin-bottom: 8px; }
.icard-d h4 { color: #4ade80; }
.icard-p h4 { color: #fb923c; }
.icard p { font-size: 0.87rem; line-height: 1.6; }
.icard-d p { color: #86efac; }
.icard-p p { color: #fbbf24; }

/* ── Buttons override ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    border-radius: 12px !important; border: none !important;
    padding: 12px 22px !important; width: 100% !important;
    font-size: 0.95rem !important; transition: 0.2s !important;
    background: linear-gradient(135deg, #1e3a1e, #16281a) !important;
    color: #d7f5d7 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
}
.stButton > button p { color: inherit !important; font-weight: 700 !important; }

/* Plant Disease choice button — solid green, dark text for contrast */
.stButton:has(button[kind]) { }
div[data-testid="stVerticalBlock"] button:focus-visible { outline: 2px solid #4ade80 !important; outline-offset: 2px; }

.btn-disease .stButton > button {
    background: linear-gradient(135deg, #4ade80, #22c55e) !important;
    color: #07210c !important;
    box-shadow: 0 4px 18px rgba(74,222,74,0.25) !important;
}
.btn-disease .stButton > button:hover {
    background: linear-gradient(135deg, #5eead4, #4ade80) !important;
    box-shadow: 0 10px 28px rgba(74,222,74,0.35) !important;
}

.btn-pest .stButton > button {
    background: linear-gradient(135deg, #fb923c, #ea580c) !important;
    color: #2a0f02 !important;
    box-shadow: 0 4px 18px rgba(251,146,60,0.25) !important;
}
.btn-pest .stButton > button:hover {
    background: linear-gradient(135deg, #fbbf24, #fb923c) !important;
    box-shadow: 0 10px 28px rgba(251,146,60,0.35) !important;
}

.btn-go-d .stButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: #eafff0 !important;
}
.btn-go-d .stButton > button:disabled {
    background: #16281a !important; color: #3a5c3a !important; opacity: 1 !important;
}
.btn-go-p .stButton > button {
    background: linear-gradient(135deg, #ea580c, #c2410c) !important;
    color: #fff7ed !important;
}
.btn-go-p .stButton > button:disabled {
    background: #2a1607 !important; color: #6a4a2a !important; opacity: 1 !important;
}

.btn-back .stButton > button {
    background: #14241a !important;
    color: #bbf7d0 !important;
    border: 1px solid #2a4a2a !important;
}
.btn-back .stButton > button:hover {
    background: #1c331f !important;
    border-color: #4ade80 !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #06090a; }
::-webkit-scrollbar-thumb { background: #1a3a1a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_CLASS_NAMES = [
    "Cherry - Powdery Mildew","Peach - Healthy","Apple - Cedar Apple Rust",
    "Cherry - Healthy","Potato - Early Blight","Strawberry - Healthy",
    "Potato - Late Blight","Blueberry - Healthy","Tomato - Yellow Leaf Curl Virus",
    "Tomato - Spider Mites (Two-spotted)","Orange - Haunglongbing (Citrus Greening)",
    "Grape - Leaf Blight (Isariopsis)","Tomato - Bacterial Spot",
    "Pepper Bell - Bacterial Spot","Apple - Healthy","Grape - Healthy",
    "Tomato - Septoria Leaf Spot","Tomato - Late Blight","Tomato - Target Spot",
    "Pepper Bell - Healthy","Apple - Black Rot","Tomato - Healthy",
    "Corn - Cercospora / Gray Leaf Spot","Potato - Healthy","Corn - Northern Leaf Blight",
    "Squash - Powdery Mildew","Corn - Common Rust","Tomato - Early Blight",
    "Grape - Esca (Black Measles)","Strawberry - Leaf Scorch","Corn - Healthy",
    "Tomato - Leaf Mold","Apple - Apple Scab","Peach - Bacterial Spot",
    "Raspberry - Healthy","Tomato - Mosaic Virus","Soybean - Healthy","Grape - Black Rot",
]

PEST_FOLDER_ORDER = [
    "62","17","80","78","37","100","12","11","0","66","27","101",
    "47","79","87","25","48","52","86","13","26","21","3","7",
    "97","92","14","33","6","89","60","16","34","64","40","82",
    "81","59","41","75","31","76","1","56","83","23","36","39",
    "54","58","55","28","94","85","93","20","96","5","88","68",
    "50","98","29","77","74","67","84","22","2","69","61","8",
    "18","32","91","49","70","45","42","30","38","65","53","10",
    "43","63","95","35","9","73","99","90","4","15","57","51",
    "72","24","19","44","71","46",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_pest_mapping():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_mapping.json")
    with open(p) as f: return json.load(f)

@st.cache_data
def pest_labels(mapping):
    return [mapping[fid] for fid in PEST_FOLDER_ORDER]

@st.cache_resource
def load_disease_model():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.keras")
    try:
        import keras as _k; return _k.models.load_model(p, compile=False)
    except Exception: pass
    return tf.keras.models.load_model(p, compile=False)

@st.cache_resource
def load_pest_model():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.keras")
    for loader in [
        lambda: __import__('keras').saving.load_model(p),
        lambda: tf.keras.models.load_model(p),
        lambda: __import__('tf_keras').models.load_model(p),
    ]:
        try: return loader()
        except Exception: pass
    raise RuntimeError("Cannot load best_model.keras — install keras>=3.0 or tensorflow>=2.16")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def run_disease(file):
    model = load_disease_model()
    file.seek(0)
    img = Image.open(file).convert("RGB").resize((256,256))
    arr = np.expand_dims(np.array(img, dtype=np.float32)/255.0, 0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return idx, float(np.max(preds))*100, preds

def run_pest(img: Image.Image, labels, top_k=5):
    model = load_pest_model()
    arr = np.expand_dims(np.array(img.convert("RGB").resize((224,224), Image.LANCZOS), dtype=np.float32), 0)
    preds = model.predict(arr, verbose=0)[0]
    top = np.argsort(preds)[::-1][:top_k]
    return [{"class": labels[i], "conf": float(preds[i])} for i in top], preds

def fmt(raw):
    p = raw.split(" - ")
    return (p[0].strip(), p[1].strip()) if len(p)==2 else (raw, "Unknown")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
# "analyse_mode" is ONLY set when user clicks a choice card on the Analyse page.
# It is reset when they go back. This keeps it from bleeding into other pages.
if "analyse_mode" not in st.session_state:
    st.session_state.analyse_mode = None   # None | "disease" | "pest"

# ─────────────────────────────────────────────────────────────────────────────
# ══ SIDEBAR ══
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo + branding block
    st.markdown("""
    <div style='padding: 28px 20px 16px; text-align: center; position: relative;'>
        <div style='font-size: 3rem; line-height: 1; margin-bottom: 6px;'>🌾</div>
        <div style='font-family: Syne, sans-serif; font-size: 1.35rem; font-weight: 800;
                    background: linear-gradient(135deg, #4ade80, #86efac 60%, #fb923c);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; letter-spacing: -0.3px;'>
            CropGuard AI
        </div>
        <div style='font-size: 0.7rem; color: #3a5c3a; letter-spacing: 2.5px;
                    text-transform: uppercase; margin-top: 4px;'>
            Smart Crop Protection
        </div>
    </div>

    <div style='height: 1px; background: linear-gradient(90deg, transparent, #1a3a1a, transparent); margin: 0 16px 20px;'></div>

    <div style='padding: 0 14px; margin-bottom: 6px;'>
        <div style='font-size: 0.65rem; color: #2d4a2d; letter-spacing: 2px;
                    text-transform: uppercase; font-weight: 600; padding: 0 4px; margin-bottom: 8px;'>
            Navigation
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav", ["🏠  Home", "🔬  Analyse", "📚  Reference Library", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style='height: 1px; background: linear-gradient(90deg, transparent, #1a3a1a, transparent); margin: 20px 16px;'></div>
    """, unsafe_allow_html=True)

    # Dynamic status panel
    am = st.session_state.analyse_mode
    if am == "disease":
        st.markdown("""
        <div style='margin: 0 14px; background: linear-gradient(135deg, #091a09, #0c2010);
                    border: 1px solid #1e3d1e; border-radius: 14px; padding: 16px;'>
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 12px;'>
                <div style='width: 8px; height: 8px; border-radius: 50%; background: #4ade80;
                            box-shadow: 0 0 6px #4ade80; flex-shrink: 0;'></div>
                <div style='font-family: Syne, sans-serif; font-size: 0.82rem; font-weight: 700; color: #4ade80;'>
                    Disease Mode Active
                </div>
            </div>
            <div style='display: flex; flex-direction: column; gap: 7px;'>
                <div style='display: flex; justify-content: space-between; font-size: 0.76rem;'>
                    <span style='color: #3a6a3a;'>Model</span>
                    <span style='color: #6b9e6b; font-weight: 500;'>CNN (5-block)</span>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.76rem;'>
                    <span style='color: #3a6a3a;'>Classes</span>
                    <span style='color: #6b9e6b; font-weight: 500;'>38 diseases</span>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.76rem;'>
                    <span style='color: #3a6a3a;'>Species</span>
                    <span style='color: #6b9e6b; font-weight: 500;'>14 plants</span>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.76rem;'>
                    <span style='color: #3a6a3a;'>Input</span>
                    <span style='color: #6b9e6b; font-weight: 500;'>256 × 256 px</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif am == "pest":
        top_k = st.slider("Top-N predictions", 1, 10, 5,
                          help="Number of ranked pest predictions to show")
        st.markdown("""
        <div style='margin: 14px 14px 0; background: linear-gradient(135deg, #160901, #240e02);
                    border: 1px solid #3a1a06; border-radius: 14px; padding: 16px;'>
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 12px;'>
                <div style='width: 8px; height: 8px; border-radius: 50%; background: #fb923c;
                            box-shadow: 0 0 6px #fb923c; flex-shrink: 0;'></div>
                <div style='font-family: Syne, sans-serif; font-size: 0.82rem; font-weight: 700; color: #fb923c;'>
                    Pest Mode Active
                </div>
            </div>
            <div style='display: flex; flex-direction: column; gap: 7px;'>
                <div style='display: flex; justify-content: space-between; font-size: 0.76rem;'>
                    <span style='color: #6a3a1a;'>Model</span>
                    <span style='color: #9e7a3a; font-weight: 500;'>EfficientNetB0</span>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.76rem;'>
                    <span style='color: #6a3a1a;'>Classes</span>
                    <span style='color: #9e7a3a; font-weight: 500;'>102 pests</span>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.76rem;'>
                    <span style='color: #6a3a1a;'>Crops</span>
                    <span style='color: #9e7a3a; font-weight: 500;'>Rice, Wheat, Maize…</span>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.76rem;'>
                    <span style='color: #6a3a1a;'>Input</span>
                    <span style='color: #9e7a3a; font-weight: 500;'>224 × 224 px</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        top_k = 5
        st.markdown("""
        <div style='margin: 0 14px; background: #0d1f0f; border: 1px solid #1e3d1e;
                    border-radius: 14px; padding: 16px;'>
            <div style='font-family: Syne, sans-serif; font-size: 0.78rem; font-weight: 700;
                        color: #7fd99f; margin-bottom: 12px; letter-spacing: 0.5px;
                        text-transform: uppercase;'>
                Two Models Available
            </div>
            <div style='display: flex; flex-direction: column; gap: 10px;'>
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <span style='font-size: 1.2rem;'>🌿</span>
                    <div>
                        <div style='font-size: 0.78rem; color: #86efac; font-weight: 700;'>Plant Disease</div>
                        <div style='font-size: 0.72rem; color: #6b9e6b;'>38 classes · CNN</div>
                    </div>
                </div>
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <span style='font-size: 1.2rem;'>🐛</span>
                    <div>
                        <div style='font-size: 0.78rem; color: #fdba74; font-weight: 700;'>Crop Pest</div>
                        <div style='font-size: 0.72rem; color: #b08552;'>102 classes · EfficientNetB0</div>
                    </div>
                </div>
            </div>
            <div style='margin-top: 14px; padding-top: 12px; border-top: 1px solid #1e3d1e;
                        font-size: 0.72rem; color: #5a8a5a; text-align: center;'>
                Go to Analyse to get started
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='height: 1px; background: linear-gradient(90deg, transparent, #1a3a1a, transparent); margin: 20px 16px;'></div>
    <div style='padding: 0 16px 28px; text-align: center;'>
        <div style='font-size: 0.68rem; color: #4a7a4a; line-height: 1.8;'>
            Powered by TensorFlow · Keras 3<br>
            Built for agricultural assistance
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══ HOME ══
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠  Home":

    st.markdown("""
    <div class='hero'>
        <div class='hero-eyebrow'>🤖 AI-Powered Agriculture</div>
        <div class='hero-title'>CropGuard AI</div>
        <div class='hero-sub'>
            One platform. Two AI models. Detect plant diseases and crop pests
            instantly — protect your harvest before it's too late.
        </div>
    </div>
    <div class='stat-bar'>
        <div class='stat-chip'><div class='stat-val'>38</div><div class='stat-lbl'>Disease Classes</div></div>
        <div class='stat-chip'><div class='stat-val'>102</div><div class='stat-lbl'>Pest Classes</div></div>
        <div class='stat-chip'><div class='stat-val'>70K+</div><div class='stat-lbl'>Training Images</div></div>
        <div class='stat-chip'><div class='stat-val'>14</div><div class='stat-lbl'>Plant Species</div></div>
        <div class='stat-chip'><div class='stat-val'>&lt;2s</div><div class='stat-lbl'>Inference Time</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='stitle stitle-d'>⚙️ How It Works</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [("Choose", "Pick Disease or Pest detection mode"),
             ("Upload", "Take a clear photo of the plant or pest"),
             ("Analyse", "AI processes the image in under 2 seconds"),
             ("Act", "Get diagnosis and take field action")]
    for col, (i,(t,d)) in zip(cols, enumerate(steps, 1)):
        with col:
            st.markdown(f"""
            <div class='step'>
                <div class='step-num step-num-d'>{i}</div>
                <div style='font-family:Syne,sans-serif;font-size:0.88rem;font-weight:700;color:#bbf7d0;margin-bottom:4px'>{t}</div>
                <div class='step-txt step-txt-d'>{d}</div>
            </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='stitle stitle-d'>🌿 Plant Disease Detection</div>", unsafe_allow_html=True)
        for icon,t,d in [("⚡","Instant","Results in under 2s using optimised CNN."),
                          ("🎯","38 Classes","14 plant species, healthy + diseased states."),
                          ("🔬","High Accuracy","Trained on 70,000+ PlantVillage images.")]:
            st.markdown(f"<div class='fcard fcard-d'><div class='fcard-icon'>{icon}</div><div class='fcard-title fcard-title-d'>{t}</div><div class='fcard-desc fcard-desc-d'>{d}</div></div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='stitle stitle-p'>🐛 Crop Pest Detection</div>", unsafe_allow_html=True)
        for icon,t,d in [("⚡","Instant","EfficientNetB0 delivers results in under 2s."),
                          ("🎯","102 Types","Comprehensive pest coverage for major crops."),
                          ("🌾","Multi-crop","Rice, wheat, maize, vegetables and fruits.")]:
            st.markdown(f"<div class='fcard fcard-p'><div class='fcard-icon'>{icon}</div><div class='fcard-title fcard-title-p'>{t}</div><div class='fcard-desc fcard-desc-p'>{d}</div></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══ ANALYSE ══
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬  Analyse":

    am = st.session_state.analyse_mode

    # ── STEP 1: No mode chosen yet → show the 2 big choice cards ──
    if am is None:
        st.markdown("""
        <div style='text-align:center; margin-bottom: 0.5rem;'>
            <div style='font-family: Syne, sans-serif; font-size: 2.2rem; font-weight: 800;
                        background: linear-gradient(135deg, #4ade80, #fb923c);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        background-clip: text;'>What would you like to detect?</div>
            <div style='color: #3a6a3a; font-size: 0.95rem; margin-top: 8px;'>
                Choose a detection mode to get started
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='choice-row'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("""
            <div class='choice-card choice-disease'>
                <span class='choice-icon'>🌿</span>
                <div class='choice-title-d'>Plant Disease Detection</div>
                <div class='choice-desc'>
                    Identify diseases on plant leaves across 14 species.
                    Upload a leaf photo and get an instant AI diagnosis
                    with confidence score and health status.
                </div>
                <span class='choice-pill-d'>38 classes · CNN · 256px</span>
                <span class='choice-arrow'>↓ Click below to select</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='btn-disease'>", unsafe_allow_html=True)
            if st.button("🌿  Detect Plant Disease", key="pick_disease"):
                st.session_state.analyse_mode = "disease"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class='choice-card choice-pest'>
                <span class='choice-icon'>🐛</span>
                <div class='choice-title-p'>Crop Pest Detection</div>
                <div class='choice-desc'>
                    Identify 102 pest and insect types on rice, wheat,
                    maize, vegetables and fruits. Upload a field image
                    and get ranked predictions instantly.
                </div>
                <span class='choice-pill-p'>102 classes · EfficientNetB0 · 224px</span>
                <span class='choice-arrow'>↓ Click below to select</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='btn-pest'>", unsafe_allow_html=True)
            if st.button("🐛  Detect Crop Pest", key="pick_pest"):
                st.session_state.analyse_mode = "pest"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── STEP 2: Mode chosen → show upload + predict UI ──
    else:
        D = (am == "disease")
        s = "d" if D else "p"

        # Header row with back button
        hcol1, hcol2 = st.columns([5, 1])
        with hcol1:
            icon = "🌿" if D else "🐛"
            title = "Plant Disease Diagnosis" if D else "Crop Pest Identification"
            sub   = "38 disease classes across 14 plant species" if D else "102 pest & insect types across major crops"
            label_color = "#4ade80" if D else "#fb923c"
            st.markdown(f"""
            <div class='mode-header mode-header-{s}'>
                <div class='mode-header-icon'>{icon}</div>
                <div>
                    <div class='mode-header-label mode-header-label-{s}'>{title}</div>
                    <div class='mode-header-sub'>{sub}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        with hcol2:
            st.markdown("<div style='height:14px'></div><div class='btn-back'>", unsafe_allow_html=True)
            if st.button("← Back", key="back_btn"):
                st.session_state.analyse_mode = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1,1], gap="large")

        with col1:
            st.markdown(f"<div class='panel-label panel-label-{s}'>📤 Upload Image</div>", unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "img", type=["jpg","jpeg","png","webp"],
                label_visibility="collapsed", key=f"up_{am}"
            )
            if uploaded:
                st.image(uploaded, caption="Uploaded image", use_container_width=True)
                st.markdown(f"""
                <div style='margin-top:8px'>
                    <span class='ftag ftag-{s}'>📁 {uploaded.name}</span>
                    <span class='ftag ftag-{s}'>📦 {round(uploaded.size/1024,1)} KB</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
            btn_label = "🚀  Analyse Plant" if D else "🚀  Identify Pest"
            btn_class = "btn-go-d" if D else "btn-go-p"
            st.markdown(f"<div class='{btn_class}'>", unsafe_allow_html=True)
            go = st.button(btn_label, disabled=(uploaded is None), key="go_btn")
            st.markdown("</div>", unsafe_allow_html=True)

            tips_body = (
                "✓ Clear, well-lit leaf photo<br>✓ Single leaf, full surface visible<br>✓ Avoid shadows and blur<br>✓ JPG, PNG or WEBP"
                if D else
                "✓ Close-up of the pest<br>✓ Good lighting, minimal blur<br>✓ Whole insect if possible<br>✓ Include affected plant area"
            )
            accent = "#4ade80" if D else "#fb923c"
            st.markdown(f"""
            <div class='tips tips-{s}'>
                <div class='tips-title tips-title-{s}'>📸 Tips for Best Results</div>
                {tips_body}
            </div>""", unsafe_allow_html=True)

        with col2:
            if go and uploaded:
                prog = st.progress(0)
                stat = st.empty()

                stat.markdown(f"<p style='color:{accent}'>🔬 Preprocessing image…</p>", unsafe_allow_html=True)
                for i in range(40): time.sleep(0.006); prog.progress(i)

                stat.markdown(f"<p style='color:{accent}'>🧠 Running neural network…</p>", unsafe_allow_html=True)

                if D:
                    try:
                        idx, conf, all_p = run_disease(uploaded)
                        if idx < 0 or idx >= len(DISEASE_CLASS_NAMES):
                            st.error(f"Unexpected index {idx}"); st.stop()
                        plant, cond = fmt(DISEASE_CLASS_NAMES[idx])
                        healthy = "healthy" in DISEASE_CLASS_NAMES[idx].lower()
                    except Exception as e:
                        st.error(f"❌ {e}"); st.stop()

                    for i in range(40,100): time.sleep(0.003); prog.progress(i)
                    stat.markdown("<p style='color:#4ade80'>✅ Analysis complete!</p>", unsafe_allow_html=True)
                    time.sleep(0.25); prog.empty(); stat.empty()

                    st.markdown(f"""
                    <div class='rcard rcard-d'>
                        <div class='rcard-eyebrow rcard-eyebrow-d'>Diagnosis Result</div>
                        <div class='rcard-plant rcard-plant-d'>{plant}</div>
                        <div class='rcard-cond rcard-cond-d'>{cond}</div>
                        <div style='margin-top:22px'>
                            <div style='display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:5px'>
                                <span style='color:#3a6a3a'>Confidence</span>
                                <span style='color:#4ade80;font-weight:700'>{conf:.1f}%</span>
                            </div>
                            <div class='cbar-wrap cbar-wrap-d'>
                                <div class='cbar-fill cbar-fill-d' style='width:{conf}%'></div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    if healthy:
                        st.markdown("<div class='alert alert-ok'>✅ <strong>Great news!</strong> Your plant appears healthy. Maintain regular care and monitoring.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='alert alert-bad'>⚠️ <strong>Action Required!</strong> <b>{cond}</b> detected on <b>{plant}</b>. Take immediate steps to prevent spread.</div>", unsafe_allow_html=True)

                    st.markdown("<div class='stitle stitle-d'>📊 Top 3 Predictions</div>", unsafe_allow_html=True)
                    for ri in np.argsort(all_p)[-3:][::-1]:
                        p2, c2 = fmt(DISEASE_CLASS_NAMES[ri])
                        pct = float(all_p[ri])*100
                        st.markdown(f"""
                        <div class='rank-row'>
                            <div class='rank-meta'>
                                <span style='color:#86efac'>{p2} — {c2}</span>
                                <span style='color:#4ade80;font-weight:700'>{pct:.1f}%</span>
                            </div>
                            <div class='cbar-wrap cbar-wrap-d' style='height:6px;margin:0'>
                                <div class='cbar-fill cbar-fill-d' style='width:{pct}%'></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                else:  # pest
                    try:
                        uploaded.seek(0)
                        img = Image.open(uploaded)
                        mapping = load_pest_mapping()
                        labels  = pest_labels(mapping)
                        results, all_p = run_pest(img, labels, top_k)
                    except Exception as e:
                        st.error(f"❌ {e}"); st.stop()

                    for i in range(40,100): time.sleep(0.003); prog.progress(i)
                    stat.markdown("<p style='color:#fb923c'>✅ Analysis complete!</p>", unsafe_allow_html=True)
                    time.sleep(0.25); prog.empty(); stat.empty()

                    top = results[0]
                    conf_pct = top["conf"]*100
                    name = top["class"].title()

                    st.markdown(f"""
                    <div class='rcard rcard-p'>
                        <div class='rcard-eyebrow rcard-eyebrow-p'>Pest Identified</div>
                        <div class='rcard-plant rcard-plant-p'>🐛 {name}</div>
                        <div style='margin-top:22px'>
                            <div style='display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:5px'>
                                <span style='color:#6a3a1a'>Confidence</span>
                                <span style='color:#fb923c;font-weight:700'>{conf_pct:.1f}%</span>
                            </div>
                            <div class='cbar-wrap cbar-wrap-p'>
                                <div class='cbar-fill cbar-fill-p' style='width:{conf_pct}%'></div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    badge = "🟢" if conf_pct>=70 else ("🟡" if conf_pct>=40 else "🔴")
                    if conf_pct >= 40:
                        st.markdown(f"<div class='alert alert-bad'>{badge} <strong>Pest Detected:</strong> <b>{name}</b> — apply appropriate pest management measures.</div>", unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Low confidence. Try a clearer, close-up photo of the pest.")

                    st.markdown(f"<div class='stitle stitle-p'>📊 Top {len(results)} Predictions</div>", unsafe_allow_html=True)
                    for i,r in enumerate(results):
                        pct = r["conf"]*100
                        st.markdown(f"""
                        <div class='rank-row'>
                            <div class='rank-meta'>
                                <span style='color:#fbbf24'>{i+1}. {r["class"].title()}</span>
                                <span style='color:#fb923c;font-weight:700'>{pct:.1f}%</span>
                            </div>
                            <div class='cbar-wrap cbar-wrap-p' style='height:6px;margin:0'>
                                <div class='cbar-fill cbar-fill-p' style='width:{pct}%'></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

            elif not uploaded:
                icon2 = "🌿" if D else "🐛"
                lbl   = "Upload a leaf image to begin" if D else "Upload a pest image to begin"
                c1h   = "#2d4a2d" if D else "#4a2a10"
                c2h   = "#1a3a1a" if D else "#3a1a06"
                st.markdown(f"""
                <div class='placeholder placeholder-{s}'>
                    <div style='font-size:4rem'>{icon2}</div>
                    <div style='font-family:Syne,sans-serif;font-size:1rem;color:{c1h};margin-top:16px'>{lbl}</div>
                    <div style='font-size:0.82rem;color:{c2h};margin-top:6px'>JPG · PNG · WEBP</div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══ REFERENCE LIBRARY ══
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📚  Reference Library":

    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                background:linear-gradient(135deg,#4ade80,#fb923c);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;margin-bottom:0.5rem'>
        Reference Library
    </div>
    <div style='color:#3a6a3a;margin-bottom:1.5rem;font-size:0.9rem'>
        Browse all detectable disease and pest classes.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌿 Plant Diseases  (38)", "🐛 Crop Pests  (102)"])

    with tab1:
        plants = {}
        for cls in DISEASE_CLASS_NAMES:
            plant, cond = fmt(cls)
            plants.setdefault(plant, []).append(cond)
        for plant, conditions in plants.items():
            with st.expander(f"🌱 {plant}  ({len(conditions)} classes)"):
                cols = st.columns(2)
                for i, cond in enumerate(conditions):
                    with cols[i%2]:
                        color = "#4ade80" if "healthy" in cond.lower() else "#fb923c"
                        icon  = "✅" if "healthy" in cond.lower() else "⚠️"
                        st.markdown(f"""
                        <div style='background:#0c1a0c;border:1px solid #182c18;border-left:3px solid {color};
                                    border-radius:8px;padding:9px 13px;margin:4px 0;font-size:0.85rem;color:#86efac'>
                            {icon} {cond}
                        </div>""", unsafe_allow_html=True)

    with tab2:
        try:
            mapping = load_pest_mapping()
            labels  = pest_labels(mapping)
            search = st.text_input("🔍 Search pest", placeholder="e.g. aphid, whitefly, stem borer…")
            filtered = [(i+1, n.title()) for i,n in enumerate(labels) if search.lower() in n.lower()]
            cols = st.columns(3)
            for j,(num,name) in enumerate(filtered[:90]):
                with cols[j%3]:
                    st.markdown(f"""
                    <div style='background:#160a01;border:1px solid #2c1803;border-left:3px solid #fb923c;
                                border-radius:8px;padding:9px 13px;margin:4px 0;font-size:0.82rem;color:#fbbf24'>
                        🐛 {name}
                    </div>""", unsafe_allow_html=True)
            if len(filtered) > 90:
                st.info(f"Showing first 90 of {len(filtered)} results. Refine your search.")
        except FileNotFoundError:
            st.warning("⚠️ `class_mapping.json` not found. Place it alongside `main.py`.")


# ─────────────────────────────────────────────────────────────────────────────
# ══ ABOUT ══
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️  About":

    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#bbf7d0;margin-bottom:2rem'>
        ℹ️ About CropGuard AI
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='stitle stitle-d'>🌿 Plant Disease Model</div>", unsafe_allow_html=True)
        for t,b in [("🎯 Goal","Early detection of leaf diseases across 14 plant species, enabling faster field intervention."),
                    ("🧠 Architecture","5-block CNN: Conv2D (16→32→64×3) + BatchNorm + MaxPooling → GAP → Dense(128) → Dense(38, softmax)."),
                    ("⚙️ Training","Adam lr=0.0005 · CategoricalCrossentropy · EarlyStopping · ReduceLROnPlateau · 256×256 input"),
                    ("📦 Dataset","PlantVillage — 87,000+ images · 38 classes · 14 species · augmented")]:
            st.markdown(f"<div class='icard icard-d'><h4>{t}</h4><p>{b}</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='stitle stitle-p'>🐛 Crop Pest Model</div>", unsafe_allow_html=True)
        for t,b in [("🎯 Goal","Identify 102 pest and insect types across rice, wheat, maize, vegetables and fruits."),
                    ("🧠 Architecture","EfficientNetB0 (fine-tuned) — lightweight backbone with built-in Rescaling layers."),
                    ("⚙️ Training","Transfer learning · 224×224 RGB input · EfficientNet internal preprocessing pipeline."),
                    ("📦 Dataset","102-class crop pest dataset. Class order recovered from training folder traversal.")]:
            st.markdown(f"<div class='icard icard-p'><h4>{t}</h4><p>{b}</p></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:2rem;padding:28px;
                background:linear-gradient(135deg,#091a09,#0c2010);
                border-radius:16px;border:1px solid #1a3a1a;text-align:center'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;
                    background:linear-gradient(135deg,#4ade80,#fb923c);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text'>
            🌾 CropGuard AI
        </div>
        <div style='font-size:0.82rem;color:#3a6a3a;margin-top:10px;line-height:1.8'>
            Python · TensorFlow · Keras 3 · EfficientNetB0 · Streamlit · NumPy · Pillow<br>
            Built for agricultural education and field assistance
        </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("CropGuard AI · Plant Disease (CNN · 38 classes) + Crop Pest (EfficientNetB0 · 102 classes)")
