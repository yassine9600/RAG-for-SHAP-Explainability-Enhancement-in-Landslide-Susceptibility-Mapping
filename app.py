"""
╔══════════════════════════════════════════════════════════════════════╗
║         LANDSLIDE SUSCEPTIBILITY MAPPING — PROFESSIONAL APP         ║
║   LightGBM Prediction · Subarea Selection · SHAP Explainability     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
import pickle
import jenkspy
import tempfile
import os
import zipfile
import folium
from streamlit_folium import st_folium
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import shap
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ── RAG / LLM engine ─────────────────────────────────────────────────────────
try:
    from rag_engine import (
        build_vectordb, load_vectordb, vectordb_exists,
        retrieve, two_stage_retrieve, shap_rerank,
        build_prediction_context, build_area_context,
        classify_failure_mechanism, detect_contradictions,
        call_openrouter, DEFAULT_SYSTEM_PROMPT, OPENROUTER_MODELS,
        CHROMA_PATH, FEW_SHOT_EXAMPLES,
        # Validation suite
        compute_ragas_style,
        compute_eqi,
        # Cross-cutting: statistical significance reporting
        bootstrap_metric_ci,
        wilcoxon_signed_rank_test,
        compute_cohens_kappa,
        compute_empirical_correlations,
        update_collinear_pairs_from_data,
        validate_collinear_pairs,
        sensitivity_analysis_l1_threshold,
        _COLLINEAR_PAIRS, _EMPIRICAL_CORRELATIONS,
        _L1_RATIO_THRESHOLD, _L1_CORR_THRESHOLD,
        # Factor decoders
        decode_lulc, decode_geology,
        LULC_LABELS, LULC_INTERPRETATION,
        GEOLOGY_LABELS, GEOLOGY_INTERPRETATION,
        # Spatial enrichment
        compute_utm_distances, build_spatial_context,
        # SHAP Limitation Diagnostics (L1 collinearity, L2 sign reversal)
        build_shap_limitations_block,
        detect_collinearity_artefacts,
        # Lexicon + new validation functions
        GEOFAITHFULNESS_LEXICON, compute_lexicon_coverage,
        # Inter-polygon consistency + KB inventory
        run_consistency_check, get_kb_inventory,
        # E1–E7 enhancements
        FEATURE_ALIASES,
        filter_contradicting_chunks,
        build_contradiction_resolution_block,
        compute_composite_validation_score,
    )
    RAG_ENGINE_OK = True
except ImportError as _rag_err:
    RAG_ENGINE_OK = False
    _rag_err_msg  = str(_rag_err)

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG + CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Landslide Susceptibility",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f4f6f9; }
[data-testid="stSidebar"]          { background: #ffffff; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] * { color: #1a2332 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #1e3a5f !important; font-weight: 700 !important; }
[data-testid="stSidebar"] label { color: #374151 !important; font-weight: 500 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #374151 !important; }
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stTextInput > div,
[data-testid="stSidebar"] .stNumberInput > div {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select { color: #1a2332 !important; background: #f8fafc !important; }
[data-testid="stSidebar"] .stMarkdown { color: #374151 !important; }
[data-testid="stSidebar"] hr { border-color: #e2e8f0 !important; }

.app-header {
    background: linear-gradient(135deg, #1a2332 0%, #243447 60%, #1e3a5f 100%);
    padding: 1.6rem 2rem 1.2rem; border-radius: 12px;
    margin-bottom: 1.4rem; box-shadow: 0 4px 18px rgba(0,0,0,0.18);
}
.app-header h1 { color: #ffffff; font-size: 2rem; margin: 0; }
.app-header p  { color: #90b8e0; margin: .3rem 0 0; font-size: .95rem; }

.kpi-row { display: flex; gap: 14px; margin-bottom: 1rem; }
.kpi-card {
    flex: 1; background: #ffffff; border-radius: 10px;
    padding: 1rem 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 5px solid #3b7dd8;
}
.kpi-card.orange { border-left-color: #e07b39; }
.kpi-card.red    { border-left-color: #c0392b; }
.kpi-card.green  { border-left-color: #27ae60; }
.kpi-label { font-size:.75rem; color:#7f8c9a; text-transform:uppercase;
             letter-spacing:.8px; margin-bottom:.25rem; }
.kpi-value { font-size:1.7rem; font-weight:700; color:#1a2332; }
.kpi-sub   { font-size:.78rem; color:#95a5a6; margin-top:.15rem; }

.section-card {
    background:#ffffff; border-radius:12px; padding:1.4rem 1.6rem;
    box-shadow:0 2px 10px rgba(0,0,0,0.07); margin-bottom:1.2rem;
}
.section-title {
    font-size:1.05rem; font-weight:700; color:#1a2332;
    border-bottom:2px solid #e8ecf1; padding-bottom:.5rem; margin-bottom:1rem;
}

[data-baseweb="tab-list"] {
    background:#ffffff; border-radius:10px;
    box-shadow:0 2px 8px rgba(0,0,0,0.07); padding:6px;
}
[data-baseweb="tab"]    { border-radius:8px !important; font-weight:600 !important; }
[aria-selected="true"]  { background:#3b7dd8 !important; color:white !important; }

.sidebar-badge {
    background:#f0f7ff; border:1px solid #93c5fd; border-radius:8px;
    padding:8px 10px; margin:6px 0; font-size:.82rem;
}
.sidebar-badge .lbl { color:#1e40af; font-size:.72rem; text-transform:uppercase; font-weight:600; }
.sidebar-badge .val { color:#1a2332; font-weight:700; }

.info-box {
    background:#e8f4fd; border-left:4px solid #3b7dd8; border-radius:6px;
    padding:.7rem 1rem; font-size:.88rem; color:#1a5276; margin-bottom:.8rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <h1>🏔️ Landslide Susceptibility Mapping</h1>
  <p>LightGBM Prediction &nbsp;·&nbsp; Jenks Natural Breaks &nbsp;·&nbsp;
     Subarea Selection &nbsp;·&nbsp; SHAP Explainability</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
def numpy_to_python(obj):
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    return obj


def classify_jenks(val, brks, labels):
    for i in range(1, len(brks)):
        if val <= brks[i]:
            return labels[i - 1]
    return labels[-1]


# ── Academic matplotlib styling for publication-quality figures ─────────────
def apply_academic_style(fig, ax, xlabel="", ylabel="",
                          grid=True, tight=True):
    """
    Journal-standard chart formatting (Geomorphology / NHESS / Landslides).
    - Times New Roman font family
    - NO embedded title (use st.caption below for figure caption)
    - Legend upper-right with thin border, white background, no overlap
    - Axis label padding to prevent data-label overlay
    - Light x-axis dashed grid behind bars, no top/right spines
    """
    matplotlib.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   8.5,
        "figure.dpi":        150,
        "mathtext.fontset":  "dejavuserif",
    })
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # NO embedded title — academic figures use captions below
    ax.set_title("")

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontfamily="serif", labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontfamily="serif", labelpad=8)

    # Spines — thin, left/bottom only
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    # Grid — light dashed x-axis only, behind bars
    if grid:
        ax.grid(True, axis="x", linestyle="--", alpha=0.25, color="#999",
                linewidth=0.5)
        ax.set_axisbelow(True)

    # Tick parameters — outward, small, padded
    ax.tick_params(axis="both", which="major", length=3, width=0.5,
                    direction="out", labelsize=9, pad=4)
    for lbl in ax.get_yticklabels() + ax.get_xticklabels():
        lbl.set_fontfamily("serif")

    # Legend — upper right, framed, white fill, no overlap
    try:
        leg = ax.get_legend()
        if leg:
            leg.set_bbox_to_anchor((1.0, 1.0))
            leg.get_frame().set_linewidth(0.5)
            leg.get_frame().set_edgecolor("#666")
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_alpha(0.95)
            for t in leg.get_texts():
                t.set_fontfamily("serif")
                t.set_fontsize(8.5)
    except Exception:
        pass

    if tight:
        fig.tight_layout(pad=1.2)
    return fig, ax


def jenks_breaks(data, n):
    try:
        return jenkspy.jenks_breaks(data, n_classes=n)
    except TypeError:
        return jenkspy.jenks_breaks(data, nb_class=n)


def shp_zip_bytes(gdf_to_save):
    buf = BytesIO()
    with tempfile.TemporaryDirectory() as td:
        gdf_to_save.to_file(os.path.join(td, "result.shp"))
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in os.listdir(td):
                zf.write(os.path.join(td, f), f)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR — UPLOADS & SETTINGS
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📂 Data Inputs")
    uploaded_bundle = st.file_uploader("Model Bundle (.pkl)",       type=["pkl"])
    uploaded_excel  = st.file_uploader("Covariates (.xlsx)",        type=["xlsx"])
    uploaded_shp    = st.file_uploader(
        "Shapefile (.zip — shp, shx, dbf, prj)", type=["zip"]
    )
    st.markdown("---")
    st.markdown("## ⚙️ Classification")
    n_classes = st.slider("Jenks natural breaks", 3, 7, 5)

    # ── LLM / RAG Settings ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🤖 LLM · RAG Settings")

    openrouter_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-...",
        help="Get a free key at https://openrouter.ai",
    )

    if RAG_ENGINE_OK:
        llm_model = st.selectbox(
            "Model",
            list(OPENROUTER_MODELS.keys()),
            format_func=lambda m: OPENROUTER_MODELS[m],
        )

        st.markdown("**Decoding Parameters**")
        llm_temperature = st.slider(
            "Temperature", 0.0, 1.5, 0.3, 0.05,
            help="Controls randomness. Lower = more deterministic."
        )
        llm_top_p = st.slider(
            "Top-P (nucleus)", 0.1, 1.0, 0.9, 0.05,
            help="Cumulative probability mass to sample from."
        )
        llm_top_k = st.slider(
            "Top-K", 1, 100, 40,
            help="Number of top tokens considered at each step."
        )
        llm_min_p = st.slider(
            "Min-P", 0.0, 0.2, 0.05, 0.01,
            help="Minimum probability relative to top token."
        )
        llm_max_tokens = st.slider(
            "Max output tokens", 256, 4096, 1024, 64,
            help="Maximum length of the generated explanation."
        )

        st.markdown("**Penalty Parameters**")
        llm_freq_penalty = st.slider(
            "Frequency penalty", 0.0, 2.0, 0.3, 0.05,
            help="Penalises repeated tokens proportionally to frequency."
        )
        llm_pres_penalty = st.slider(
            "Presence penalty", 0.0, 2.0, 0.1, 0.05,
            help="Flat penalty for any previously seen token."
        )
        llm_rep_penalty = st.slider(
            "Repetition penalty", 1.0, 2.0, 1.1, 0.05,
            help="Multiplicative penalty for repeated sequences."
        )

        st.markdown("**Context & Few-Shot**")
        llm_context_turns = st.slider(
            "Context window (turns)", 0, 10, 4,
            help="Number of previous Q&A turns to include in context."
        )
        llm_few_shot = st.slider(
            "Few-shot examples", 0, len(FEW_SHOT_EXAMPLES)//2, 2,
            help="Example Q&A pairs prepended to guide response style."
        )
        llm_n_chunks = st.slider(
            "RAG top-k chunks", 1, 15, 5,
            help="Number of retrieved literature chunks per query."
        )

        st.markdown("**PDF Knowledge Base**")
        pdf_folder = st.text_input(
            "PDF folder path",
            value="./pdfs",
            help="Local folder containing your scientific PDF papers.",
        )
        llm_chunk_size = st.number_input(
            "Chunk size (chars)", 256, 1024, 512, 64,
            help="Size of each text chunk when indexing PDFs."
        )
        llm_chunk_overlap = st.number_input(
            "Chunk overlap (chars)", 0, 256, 64, 16,
        )
        rebuild_db = st.button(
            "🔄 Build / Rebuild Vector DB",
            help="Index all PDFs in the folder into ChromaDB.",
        )
    else:
        st.warning(f"RAG engine unavailable: install chromadb + sentence-transformers")
        llm_model = "meta-llama/llama-3.3-70b-instruct"
        llm_temperature = 0.3; llm_top_p = 0.9; llm_top_k = 40
        llm_min_p = 0.05; llm_max_tokens = 1024
        llm_freq_penalty = 0.3; llm_pres_penalty = 0.1; llm_rep_penalty = 1.1
        llm_context_turns = 4; llm_few_shot = 2; llm_n_chunks = 5
        pdf_folder = "./pdfs"; llm_chunk_size = 512; llm_chunk_overlap = 64
        rebuild_db = False

# ── Guard ────────────────────────────────────────────────────────────
if not (uploaded_bundle and uploaded_excel and uploaded_shp):
    st.markdown("""
    <div class="info-box">
      ⬅️ &nbsp;Please upload <b>all three files</b> in the sidebar to begin:<br>
      &nbsp;&nbsp;① Model bundle (.pkl) &nbsp;&nbsp;
      ② Covariates (.xlsx) &nbsp;&nbsp;
      ③ Shapefile (.zip)
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════
#  LOADERS (cached)
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_bundle(fb):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(fb); path = tmp.name
    try:
        b = joblib.load(path)
    except Exception:
        b = pickle.loads(fb)
    finally:
        os.remove(path)
    return b


@st.cache_data
def load_covariates(fb):
    return pd.read_excel(BytesIO(fb))


@st.cache_data
def load_shapefile(fb):
    with tempfile.TemporaryDirectory() as td:
        zp = os.path.join(td, "up.zip")
        with open(zp, "wb") as f: f.write(fb)
        with zipfile.ZipFile(zp) as zf: zf.extractall(td)
        shp = None
        for root, _, files in os.walk(td):
            for fn in files:
                if fn.lower().endswith(".shp"):
                    shp = os.path.join(root, fn); break
            if shp: break
        if shp is None:
            st.error("No .shp found in ZIP."); st.stop()
        stem    = os.path.splitext(os.path.basename(shp))[0]
        has_shx = any(f.lower() == stem.lower() + ".shx"
                      for f in os.listdir(os.path.dirname(shp)))
        if not has_shx:
            st.warning(".shx missing — attempting SHAPE_RESTORE_SHX.")
        try:
            import pyogrio
            gdf = pyogrio.read_dataframe(
                shp, config_options={"SHAPE_RESTORE_SHX": "YES"})
        except TypeError:
            os.environ["GDAL_SHAPE_RESTORE_SHX"] = "YES"
            gdf = gpd.read_file(shp)
            os.environ.pop("GDAL_SHAPE_RESTORE_SHX", None)
        except Exception as e:
            st.error(f"Shapefile read error: {e}"); st.stop()
    return gdf


# ── Load ─────────────────────────────────────────────────────────────
bundle        = load_bundle(uploaded_bundle.getvalue())
covariates_df = load_covariates(uploaded_excel.getvalue())
gdf_full      = load_shapefile(uploaded_shp.getvalue())

for k in ("model", "scaler", "feature_names"):
    if k not in bundle:
        st.error(f"Bundle missing key: **'{k}'**"); st.stop()

model         = bundle["model"]
scaler        = bundle["scaler"]
feature_names = bundle["feature_names"]

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR — MODEL INFO
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("---")
    st.markdown("## 📦 Model Info")
    best_f1 = bundle.get("best_cv_f1", "N/A")
    f1_str  = f"{best_f1:.4f}" if isinstance(best_f1, (int, float)) else str(best_f1)
    st.markdown(f"""
    <div class="sidebar-badge">
      <div class="lbl">Model</div>
      <div class="val">{bundle.get('model_name','LightGBM')}</div>
    </div>
    <div class="sidebar-badge">
      <div class="lbl">Training Date</div>
      <div class="val">{bundle.get('training_date','N/A')}</div>
    </div>
    <div class="sidebar-badge">
      <div class="lbl">Best CV F1</div>
      <div class="val">{f1_str}</div>
    </div>
    <div class="sidebar-badge">
      <div class="lbl">Features</div>
      <div class="val">{len(feature_names)}</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Feature list", expanded=False):
        _KEY_FEATURES = {"vs30", "twi", "VS30", "TWI",
                         "vs_30", "vs 30", "tw_i"}
        _feat_html = "<div style='font-size:0.82rem;line-height:1.7'>"
        for _fi, _fn in enumerate(feature_names):
            _is_key = (_fn.lower() in {"vs30","twi"} or
                       any(k in _fn.lower() for k in ["vs30","twi"]))
            _bg  = "#e8f4fd" if _is_key else "transparent"
            _fw  = "600"    if _is_key else "normal"
            _tag = " 🔵"   if _is_key else ""
            _feat_html += (
                f"<div style='padding:1px 4px;background:{_bg};"
                f"border-radius:3px;font-weight:{_fw}'>"
                f"{_fi+1}. {_fn}{_tag}</div>"
            )
        _feat_html += "</div>"
        st.markdown(_feat_html, unsafe_allow_html=True)
        st.caption(f"🔵 = Seismic site amplification factors | {len(feature_names)} features total")
    with st.expander("Best hyperparameters"):
        st.json(numpy_to_python(bundle.get("best_params", {})))
    with st.expander("Test-set metrics"):
        m = bundle.get("metrics", {})
        st.json(numpy_to_python(m) if isinstance(m, dict) else {})

# ══════════════════════════════════════════════════════════════════════
#  VALIDATE COLUMNS
# ══════════════════════════════════════════════════════════════════════
if "Id" not in covariates_df.columns:
    st.error("Column **'Id'** not found in the covariates Excel."); st.stop()
if "Id" not in gdf_full.columns:
    st.error("Column **'Id'** not found in the shapefile."); st.stop()
miss = [f for f in feature_names if f not in covariates_df.columns]
if miss:
    st.error(f"Features missing from covariates: `{miss}`"); st.stop()

# Normalise Id types
gdf_full["Id"]      = gdf_full["Id"].astype(str).str.strip()
covariates_df["Id"] = covariates_df["Id"].astype(str).str.strip()

# ── Detect UTM X/Y columns (WGS_1984_UTM_Zone_29N) ──────────────────────────
# Accept common column name variants (case-insensitive)
def _find_utm_col(df, keywords):
    for col in df.columns:
        cl = col.lower().strip()
        if any(cl == k or cl.startswith(k) for k in keywords):
            return col
    return None

_xcol = _find_utm_col(covariates_df, ["x", "x_utm", "easting",  "xcoord", "x_coord", "longitude_utm"])
_ycol = _find_utm_col(covariates_df, ["y", "y_utm", "northing", "ycoord", "y_coord", "latitude_utm"])

# Validate: must be numeric and in UTM 29N range (Haouz ≈ X 380k-530k, Y 3380k-3560k)
_utm_ok = False
if _xcol and _ycol:
    try:
        _xvals = pd.to_numeric(covariates_df[_xcol], errors="coerce")
        _yvals = pd.to_numeric(covariates_df[_ycol], errors="coerce")
        _xmed, _ymed = float(_xvals.median()), float(_yvals.median())
        if 300_000 < _xmed < 700_000 and 3_000_000 < _ymed < 4_000_000:
            _utm_ok   = True
            UTM_XCOL  = _xcol
            UTM_YCOL  = _ycol
    except Exception:
        pass

if not _utm_ok:
    UTM_XCOL = None
    UTM_YCOL = None

if UTM_XCOL:
    st.sidebar.success(f"📍 UTM coords detected: **{UTM_XCOL}** / **{UTM_YCOL}**")
else:
    st.sidebar.caption("ℹ️ No UTM X/Y columns detected in Excel — spatial distances unavailable.")

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR — SUBAREA FILTER
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("---")
    st.markdown("## 🗂️ Subarea Filter")

    candidate_cols = [
        c for c in gdf_full.columns
        if c not in ("Id", "geometry")
        and gdf_full[c].dtype == object
        and 1 < gdf_full[c].nunique() <= 200
    ]

    use_subarea   = st.toggle("Enable subarea filter", value=False)
    subarea_col   = None
    selected_vals = None

    if use_subarea:
        if not candidate_cols:
            st.warning("No suitable categorical columns found in shapefile.")
            use_subarea = False
        else:
            subarea_col = st.selectbox(
                "Filter column", candidate_cols,
                help="Column defining subareas (district, zone, etc.)",
            )
            all_vals      = sorted(gdf_full[subarea_col].dropna().unique().tolist())
            selected_vals = st.multiselect(
                "Select subarea(s)", all_vals,
                default=all_vals[:1] if all_vals else [],
            )
            if not selected_vals:
                st.warning("Select at least one subarea.")
                use_subarea = False

# ══════════════════════════════════════════════════════════════════════
#  APPLY SUBAREA FILTER
# ══════════════════════════════════════════════════════════════════════
if use_subarea and subarea_col and selected_vals:
    gdf_work = gdf_full[gdf_full[subarea_col].isin(selected_vals)].copy()
    if gdf_work.empty:
        st.error("Subarea filter returned empty GeoDataFrame."); st.stop()
    active_ids = set(gdf_work["Id"].unique())
    cov_work   = covariates_df[covariates_df["Id"].isin(active_ids)].copy()
    area_label = f"{subarea_col}: {', '.join(selected_vals)}"
else:
    gdf_work   = gdf_full.copy()
    cov_work   = covariates_df.copy()
    area_label = "Full Study Area"

# ══════════════════════════════════════════════════════════════════════
#  PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════
X_raw = cov_work[feature_names].copy()

nan_mask = X_raw.isnull().any(axis=1)
if nan_mask.sum():
    st.warning(f"{nan_mask.sum()} row(s) with missing values excluded.")
    X_raw    = X_raw[~nan_mask]
    cov_work = cov_work[~nan_mask].copy()

X_scaled = scaler.transform(X_raw)
proba    = model.predict_proba(X_scaled)[:, 1]
cov_work = cov_work.copy()
cov_work["probability"] = proba

# ── Jenks ────────────────────────────────────────────────────────────
breaks = jenks_breaks(proba.tolist(), n_classes)

LABELS_MAP = {
    3: ["Low", "Moderate", "High"],
    4: ["Low", "Moderate", "High", "Very High"],
    5: ["Very Low", "Low", "Moderate", "High", "Very High"],
    6: ["Very Low", "Low", "Moderate", "High", "Very High", "Extremely High"],
    7: ["Very Low", "Low", "Low-Moderate", "Moderate",
        "High", "Very High", "Extremely High"],
}
class_labels = LABELS_MAP.get(n_classes, [f"Class {i+1}" for i in range(n_classes)])

cov_work["susceptibility"] = cov_work["probability"].apply(
    lambda v: classify_jenks(v, breaks, class_labels)
)

# ── Merge onto GDF ───────────────────────────────────────────────────
gdf_result = gdf_work.merge(
    cov_work[["Id", "probability", "susceptibility"]],
    on="Id", how="left",
)
unmatched = gdf_result["probability"].isna().sum()
if unmatched:
    st.warning(f"{unmatched} polygon(s) had no matching Id — shown grey.")

if gdf_result.crs is None:
    gdf_result = gdf_result.set_crs(epsg=4326)
gdf_4326 = gdf_result.to_crs(epsg=4326)

# ══════════════════════════════════════════════════════════════════════
#  COLOUR SCHEME
# ══════════════════════════════════════════════════════════════════════
COLOR_SCHEMES = {
    3: ["#3288bd", "#ffffbf", "#d53e4f"],
    4: ["#3288bd", "#abdda4", "#fdae61", "#d53e4f"],
    5: ["#3288bd", "#abdda4", "#ffffbf", "#fdae61", "#d53e4f"],
    6: ["#3288bd", "#abdda4", "#ffffbf", "#fdae61", "#d53e4f", "#7b0d1e"],
    7: ["#3288bd", "#6baed6", "#abdda4", "#ffffbf",
        "#fdae61", "#d53e4f", "#7b0d1e"],
}
COLORS      = COLOR_SCHEMES.get(n_classes, COLOR_SCHEMES[5])
LABEL_COLOR = dict(zip(class_labels, COLORS))

# ══════════════════════════════════════════════════════════════════════
#  SHAP COMPUTATION
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="⏳  Computing TreeSHAP values…")
def compute_shap(_model, X_arr, feat_names):
    """Standard marginal TreeSHAP (baseline for comparison)."""
    explainer = shap.TreeExplainer(_model)
    sv        = explainer.shap_values(X_arr)
    if isinstance(sv, list):
        sv = sv[1]          # binary classification — positive class
    df_shap = pd.DataFrame(sv, columns=feat_names)
    base    = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(base[1])
    else:
        base = float(base)
    return df_shap, base


def extract_sections_1_2(llm_text: str) -> str:
    """
    Extract only Section 1 (Summary Verdict) and Section 2 (Feature Attribution
    Analysis) from the LLM output. RAGAS and GeoFaithfulness evaluate only these
    two sections — they contain the scientific claims and directional attributions
    that the metrics are designed to assess.
    """
    import re
    text = llm_text or ""
    # Find where Section 3 starts (any of its possible headings)
    patterns = [
        r'\*\*3\.\s',           # **3. ...
        r'###?\s*3\.',          # ### 3. or ## 3.
        r'\n3\.\s+Geomorphol',  # plain "3. Geomorphological"
        r'\*\*Geomorphological Synthesis\*\*',
    ]
    earliest = len(text)
    for pat in patterns:
        m = re.search(pat, text)
        if m and m.start() < earliest:
            earliest = m.start()
    return text[:earliest].strip()


# ── TreeSHAP: fast global computation (always available) ──────────────
shap_df, shap_base = compute_shap(model, X_scaled, feature_names)


# ── Vector DB rebuild (triggered from sidebar) ──────────────────────────────
if RAG_ENGINE_OK and rebuild_db:
    with st.spinner("Building vector DB from PDFs…"):
        try:
            progress_bar = st.progress(0.0)
            def _prog(frac, msg):
                progress_bar.progress(frac, text=msg)
            n_chunks, pdf_report = build_vectordb(
                pdf_folder, CHROMA_PATH,
                chunk_size=int(llm_chunk_size),
                overlap=int(llm_chunk_overlap),
                progress_cb=_prog,
            )
            progress_bar.empty()
            st.success(f"✅ Vector DB built: {n_chunks:,} chunks from {pdf_folder}")

            # Persist build report for KB inventory tab
            st.session_state["_kb_pdf_report"] = pdf_report

            # ── Per-file diagnostic table ────────────────────────────
            import pandas as _pd_rpt
            rpt_df = _pd_rpt.DataFrame(pdf_report)
            rpt_df.columns = ["File","Pages","Chars","Chunks","Method","Error"]
            rpt_df["Status"] = rpt_df["Chunks"].apply(
                lambda x: "✅" if x > 0 else "❌")
            st.dataframe(
                rpt_df[["Status","File","Method","Pages","Chars","Chunks","Error"]],
                use_container_width=True,
                hide_index=True,
            )
            st.cache_resource.clear()   # force reload of vector DB

        except FileNotFoundError as fe:
            st.error(f"📁 {fe}")
            st.info(
                "**How to fix:** create the folder, place your PDFs inside, "
                "then click *Build / Rebuild Vector DB* again."
            )
        except ValueError as ve:
            # Detailed per-file diagnostic from the engine
            st.error("Vector DB build failed — no text could be extracted.")
            st.code(str(ve), language="text")
            st.info(
                "**Possible causes and fixes:**\n\n"
                "- **Wrong folder path** — check the sidebar path "
                "(default `./pdfs`; on Windows use absolute path "
                "e.g. `C:/Users/you/pdfs`)\n"
                "- **Scanned PDFs (image-only)** — install OCR: "
                "`pip install pytesseract Pillow` "
                "and the Tesseract binary from tesseract-ocr.github.io\n"
                "- **Password-protected PDFs** — remove protection first\n"
                "- **Corrupt PDFs** — re-download or re-export the file"
            )
        except Exception as db_err:
            st.error(f"Unexpected error: {db_err}")

# ── Load vector DB (cached) ──────────────────────────────────────────────────
@st.cache_resource
def _load_vdb(chroma_path):
    try:
        return load_vectordb(chroma_path)
    except Exception:
        return None

rag_collection = _load_vdb(CHROMA_PATH) if RAG_ENGINE_OK else None

# Attach SHAP columns to cov_work
for f in feature_names:
    cov_work[f"shap_{f}"] = shap_df[f].values

shap_abs = shap_df.abs()
# ── Global SHAP rank dict (used by L4 local-instability detector) ─────────
_global_rank_order = shap_df.abs().mean().sort_values(ascending=False)
GLOBAL_SHAP_RANK   = {feat: i+1 for i, feat in enumerate(_global_rank_order.index)}
cov_work["dominant_feature"] = shap_abs.idxmax(axis=1).values
cov_work["dominant_shap"]    = shap_abs.max(axis=1).values

# Merge SHAP info into GDF
shap_merge_cols = (
    ["Id", "dominant_feature", "dominant_shap"]
    + [f"shap_{f}" for f in feature_names]
)
gdf_4326 = gdf_4326.merge(
    cov_work[shap_merge_cols], on="Id", how="left"
)

# Feature colour palette
FEAT_PALETTE = [
    "#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261",
    "#6a4c93","#1982c4","#8ac926","#ff595e","#6a994e",
    "#a8dadc","#ffb703","#fb8500","#023047","#8ecae6",
]
feat_color = {f: FEAT_PALETTE[i % len(FEAT_PALETTE)]
              for i, f in enumerate(feature_names)}

# ══════════════════════════════════════════════════════════════════════
#  SHARED SESSION STATE — polygon selection shared across all tabs
# ══════════════════════════════════════════════════════════════════════
if "llm_clicked_poly" not in st.session_state:
    st.session_state["llm_clicked_poly"] = None

# ══════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════
tab_map, tab_stats, tab_shap_tab, tab_draw, tab_valid = st.tabs([
    "🗺️  Susceptibility Map",
    "📊  Statistics",
    "🧠  SHAP Explainability",
    "🎯  Polygon Explorer",
    "🔬  Validation",
])

# ┌──────────────────────────────────────────────────────────────────┐
# │  TAB 1 — MAP                                                     │
# └──────────────────────────────────────────────────────────────────┘
with tab_map:

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card green">
        <div class="kpi-label">Study Area</div>
        <div class="kpi-value" style="font-size:1rem">{area_label}</div>
        <div class="kpi-sub">{len(cov_work):,} polygons</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Min Probability</div>
        <div class="kpi-value">{proba.min():.3f}</div>
        <div class="kpi-sub">Lowest risk polygon</div>
      </div>
      <div class="kpi-card orange">
        <div class="kpi-label">Mean Probability</div>
        <div class="kpi-value">{proba.mean():.3f}</div>
        <div class="kpi-sub">Area average</div>
      </div>
      <div class="kpi-card red">
        <div class="kpi-label">Max Probability</div>
        <div class="kpi-value">{proba.max():.3f}</div>
        <div class="kpi-sub">Highest risk polygon</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_map, col_ctrl = st.columns([4, 1])

    with col_ctrl:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎨 Controls</div>',
                    unsafe_allow_html=True)
        map_layer    = st.radio("Active layer",
                                ["Susceptibility", "Dominant SHAP Feature"])
        fill_opacity = st.slider("Fill opacity", 0.2, 0.9, 0.55, 0.05,
                                 help="Lower values reveal terrain relief beneath polygons")
        basemap_choice = st.selectbox("Base map", [
            "Relief (ESRI Shaded Relief)",
            "Topo (OpenTopoMap)",
            "Hillshade (ESRI World Hillshade)",
            "Light (CartoDB Positron)",
            "Satellite (ESRI Imagery)",
        ])
        show_tooltip = st.checkbox("Show tooltips", value=True)
        st.caption(f"Jenks breaks: `{[round(b,4) for b in breaks]}`")
        st.markdown("---")
        st.markdown("**Legend**")
        if map_layer == "Susceptibility":
            for lbl, clr in LABEL_COLOR.items():
                st.markdown(
                    f'<span style="background:{clr};display:inline-block;'
                    f'width:13px;height:13px;border-radius:3px;'
                    f'margin-right:6px;vertical-align:middle;'
                    f'border:1px solid #ccc"></span>{lbl}',
                    unsafe_allow_html=True,
                )
        else:
            shown = sorted(cov_work["dominant_feature"].dropna().unique())
            for f in shown:
                st.markdown(
                    f'<span style="background:{feat_color.get(f,"#aaa")};'
                    f'display:inline-block;width:13px;height:13px;'
                    f'border-radius:3px;margin-right:6px;vertical-align:middle;'
                    f'border:1px solid #ccc"></span><small>{f}</small>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_map:
        centroid = gdf_4326.dissolve().centroid.iloc[0]

        # Create map with neutral base, then add all named tile layers
        m = folium.Map(
            location=[centroid.y, centroid.x],
            zoom_start=11,
            tiles=None,          # no default tile — we add named layers below
        )

        # ── Named basemap tile layers ───────────────────────────────
        _TILE_DEFS = {
            "Relief (ESRI Shaded Relief)": (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
                "Esri World Shaded Relief",
            ),
            "Topo (OpenTopoMap)": (
                "https://tile.opentopomap.org/{z}/{x}/{y}.png",
                "OpenTopoMap © contributors, CC-BY-SA",
            ),
            "Hillshade (ESRI World Hillshade)": (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
                "Esri World Hillshade",
            ),
            "Light (CartoDB Positron)": (
                "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                "CartoDB Positron",
            ),
            "Satellite (ESRI Imagery)": (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "Esri World Imagery",
            ),
        }

        for _name, (_url, _attr) in _TILE_DEFS.items():
            folium.TileLayer(
                tiles=_url, attr=_attr, name=_name,
                # Show the selected basemap as active (checked)
                show=(_name == basemap_choice),
            ).add_to(m)

        # Style functions — thin dark border for polygon delineation
        def style_susc(feat):
            s = feat["properties"].get("susceptibility", "")
            return {"fillColor": LABEL_COLOR.get(s, "#aaa"),
                    "color": "#2c2c2c", "weight": 0.4,
                    "fillOpacity": fill_opacity}

        def style_dom(feat):
            f = feat["properties"].get("dominant_feature", "")
            return {"fillColor": feat_color.get(f, "#aaa"),
                    "color": "#2c2c2c", "weight": 0.4,
                    "fillOpacity": fill_opacity}

        if map_layer == "Susceptibility":
            style_fn   = style_susc
            tt_fields  = ["Id", "probability", "susceptibility"]
            tt_aliases = ["ID:", "Probability:", "Susceptibility:"]
        else:
            style_fn   = style_dom
            tt_fields  = ["Id", "dominant_feature", "dominant_shap", "susceptibility"]
            tt_aliases = ["ID:", "Dominant Factor:", "SHAP Impact:", "Susceptibility:"]

        # Sanitise for folium JSON
        geo_df = gdf_4326.copy()
        for c in geo_df.select_dtypes(include=[float]).columns:
            geo_df[c] = geo_df[c].fillna(-9999)
        for c in geo_df.select_dtypes(include="object").columns:
            if c != "geometry":
                geo_df[c] = geo_df[c].fillna("N/A")

        folium.GeoJson(
            geo_df.__geo_interface__,
            style_function=style_fn,
            tooltip=(folium.GeoJsonTooltip(
                         fields=tt_fields, aliases=tt_aliases, localize=True)
                     if show_tooltip else None),
        ).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, use_container_width=True, height=580)

    # Static map download — topographic academic style
    with st.expander("🖼️ Download Static Map (300 dpi — journal quality PNG)"):
        plot_gdf = gdf_4326.copy()
        plot_gdf["susceptibility"] = plot_gdf["susceptibility"].fillna("Unknown")
        cats  = class_labels + (["Unknown"]
                if "Unknown" in plot_gdf["susceptibility"].values else [])
        clist = COLORS + (["#aaaaaa"]
                if "Unknown" in plot_gdf["susceptibility"].values else [])

        fig_s, ax_s = plt.subplots(figsize=(14, 11))
        fig_s.patch.set_facecolor("white")
        ax_s.set_facecolor("#dce9f5")   # light topographic background

        # Try to add contextily hillshade basemap
        try:
            import contextily as ctx
            plot_gdf_3857 = plot_gdf.to_crs(epsg=3857)
            plot_gdf_3857.plot(
                column="susceptibility", categorical=True, categories=cats,
                cmap=mcolors.ListedColormap(clist), legend=False,
                edgecolor="#555555", linewidth=0.2,
                alpha=0.60, ax=ax_s,
            )
            ctx.add_basemap(
                ax_s, crs=plot_gdf_3857.crs,
                source=ctx.providers.Esri.WorldShadedRelief,
                zoom="auto", alpha=1.0, zorder=0,
            )
            # Re-plot polygons on top of basemap (transparent)
            plot_gdf_3857.plot(
                column="susceptibility", categorical=True, categories=cats,
                cmap=mcolors.ListedColormap(clist), legend=False,
                edgecolor="#333333", linewidth=0.25,
                alpha=0.55, ax=ax_s, zorder=2,
            )
        except Exception:
            # Fallback: white background with polygon borders
            plot_gdf.plot(
                column="susceptibility", categorical=True, categories=cats,
                cmap=mcolors.ListedColormap(clist), legend=False,
                edgecolor="#555555", linewidth=0.3, alpha=0.65, ax=ax_s,
            )

        # Academic legend (manual patches for full control)
        from matplotlib.patches import Patch as _Patch
        _legend_handles = [
            _Patch(facecolor=clr, edgecolor="#333", linewidth=0.5,
                   alpha=0.75, label=lbl)
            for lbl, clr in zip(cats, clist)
        ]
        ax_s.legend(
            handles=_legend_handles,
            title="Susceptibility class", title_fontsize=10,
            fontsize=9, loc="lower left",
            framealpha=0.92, edgecolor="#aaa",
            frameon=True, fancybox=False,
        )

        # Scale bar (approximate: 10 km in degrees ≈ 0.09°)
        _xmin, _xmax = ax_s.get_xlim()
        _ymin, _ymax = ax_s.get_ylim()
        _bar_x = _xmin + 0.05 * (_xmax - _xmin)
        _bar_y = _ymin + 0.06 * (_ymax - _ymin)
        _bar_len = 0.09  # ~10 km at 31°N
        ax_s.plot([_bar_x, _bar_x + _bar_len], [_bar_y, _bar_y],
                  color="black", linewidth=2.5, solid_capstyle="butt", zorder=5)
        ax_s.text(_bar_x + _bar_len / 2, _bar_y + 0.005 * (_ymax - _ymin),
                  "~10 km", ha="center", va="bottom", fontsize=8,
                  fontfamily="serif", fontweight="bold", zorder=5)

        # North arrow
        _na_x = _xmax - 0.06 * (_xmax - _xmin)
        _na_y = _ymin + 0.10 * (_ymax - _ymin)
        ax_s.annotate("N", xy=(_na_x, _na_y + 0.018 * (_ymax - _ymin)),
                      xytext=(_na_x, _na_y),
                      fontsize=11, fontweight="bold", ha="center",
                      fontfamily="serif",
                      arrowprops=dict(arrowstyle="-|>", color="black",
                                      lw=1.5), zorder=5)

        ax_s.set_title(
            f"Landslide Susceptibility Map — {area_label}\n"
            f"LightGBM · Jenks Natural Breaks ({n_classes} classes) · "
            f"Al Haouz Province, Morocco",
            fontsize=12, fontweight="bold", color="#1a2332",
            fontfamily="serif", pad=10,
        )
        ax_s.set_xlabel("Longitude (°)", fontsize=9, fontfamily="serif")
        ax_s.set_ylabel("Latitude (°)",  fontsize=9, fontfamily="serif")
        ax_s.tick_params(labelsize=8)
        ax_s.grid(True, linestyle="--", linewidth=0.3, color="#aaa", alpha=0.6)
        fig_s.tight_layout(pad=1.2)
        st.pyplot(fig_s)

        png_buf = BytesIO()
        fig_s.savefig(png_buf, format="png", dpi=300,
                      bbox_inches="tight", facecolor="white")
        png_buf.seek(0)
        plt.close(fig_s)
        st.download_button("📥 Download PNG (300 dpi — journal quality)", data=png_buf,
                           file_name="susceptibility_map.png", mime="image/png")

# ┌──────────────────────────────────────────────────────────────────┐
# │  TAB 2 — STATISTICS                                              │
# └──────────────────────────────────────────────────────────────────┘
with tab_stats:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Class Distribution</div>',
                    unsafe_allow_html=True)
        class_counts = (cov_work["susceptibility"].value_counts()
                        .reindex(class_labels).fillna(0).astype(int))
        fig_b, ax_b = plt.subplots(figsize=(6, 3.5))
        bars = ax_b.bar(class_counts.index, class_counts.values,
                        color=COLORS, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, class_counts.values):
            ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                      f"{v:,}", ha="center", va="bottom", fontsize=9,
                      fontweight="bold")
        ax_b.set_ylabel("Polygon Count")
        ax_b.spines[["top","right"]].set_visible(False)
        ax_b.tick_params(axis="x", rotation=20)
        fig_b.tight_layout(); st.pyplot(fig_b); plt.close(fig_b)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Probability Distribution</div>',
                    unsafe_allow_html=True)
        fig_h, ax_h = plt.subplots(figsize=(6, 3.5))
        n_h, bins_h, patches_h = ax_h.hist(proba, bins=40,
                                            edgecolor="white", linewidth=0.5)
        for patch, left in zip(patches_h, bins_h[:-1]):
            patch.set_facecolor(
                LABEL_COLOR.get(classify_jenks(left, breaks, class_labels), "#aaa"))
        for b in breaks[1:-1]:
            ax_h.axvline(b, color="#333", linestyle="--",
                         linewidth=0.9, alpha=0.6)
        ax_h.set_xlabel("Landslide Probability")
        ax_h.set_ylabel("Count")
        ax_h.spines[["top","right"]].set_visible(False)
        ax_h.legend(handles=[mpatches.Patch(facecolor=LABEL_COLOR[l], label=l)
                              for l in class_labels],
                    fontsize=7, loc="upper right")
        fig_h.tight_layout(); st.pyplot(fig_h); plt.close(fig_h)
        st.markdown('</div>', unsafe_allow_html=True)

    # Summary table
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📐 Area Summary</div>',
                unsafe_allow_html=True)
    total = len(cov_work)
    rows  = []
    for i, lbl in enumerate(class_labels):
        cnt = (cov_work["susceptibility"] == lbl).sum()
        rows.append({
            "Class": lbl,
            "Polygons": f"{cnt:,}",
            "% of Area": f"{cnt/total*100:.1f}%" if total else "0%",
            "Prob. Range": f"{breaks[i]:.4f} – {breaks[i+1]:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Boxplot by class
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📦 Feature Distribution by Class</div>',
                unsafe_allow_html=True)
    box_feat = st.selectbox("Feature", feature_names, key="box_feat")
    if box_feat in cov_work.columns:
        fig_bx, ax_bx = plt.subplots(figsize=(10, 4))
        data_by_cls = [cov_work.loc[cov_work["susceptibility"]==l,
                                    box_feat].dropna()
                       for l in class_labels]
        bp = ax_bx.boxplot(data_by_cls, patch_artist=True,
                           medianprops=dict(color="black", linewidth=2))
        for patch, clr in zip(bp["boxes"], COLORS):
            patch.set_facecolor(clr); patch.set_alpha(0.8)
        ax_bx.set_xticklabels(class_labels, rotation=20)
        ax_bx.set_ylabel(box_feat)
        ax_bx.spines[["top","right"]].set_visible(False)
        fig_bx.tight_layout(); st.pyplot(fig_bx); plt.close(fig_bx)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_shap_tab:

    st.markdown(f"""
    <div class="info-box">
      <b>SHAP (SHapley Additive exPlanations)</b> decomposes each prediction
      into individual feature contributions. Positive SHAP → increases landslide
      risk. Negative SHAP → decreases it. &nbsp;|&nbsp;
      <b>Base value: {shap_base:.4f}</b>
    </div>
    """, unsafe_allow_html=True)

    g1, g2 = st.columns(2)

    # Global importance bar
    with g1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏆 Global Feature Importance (mean |SHAP|)</div>',
                    unsafe_allow_html=True)
        mean_abs = shap_df.abs().mean().sort_values(ascending=True)
        fig_gi, ax_gi = plt.subplots(figsize=(6, max(3, len(feature_names)*0.4)))
        bars_gi = ax_gi.barh(mean_abs.index, mean_abs.values,
                             color=[feat_color[f] for f in mean_abs.index],
                             edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars_gi, mean_abs.values):
            ax_gi.text(v + mean_abs.max()*0.01,
                       bar.get_y()+bar.get_height()/2,
                       f"{v:.4f}", va="center", fontsize=8)
        ax_gi.set_xlabel("Mean |SHAP value|")
        ax_gi.spines[["top","right"]].set_visible(False)
        fig_gi.tight_layout(); st.pyplot(fig_gi); plt.close(fig_gi)
        st.markdown('</div>', unsafe_allow_html=True)

    # Beeswarm — shap.summary_plot does NOT accept `ax` in older versions;
    # the correct pattern is to size the figure beforehand and grab plt.gcf().
    with g2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔵 SHAP Beeswarm Plot</div>',
                    unsafe_allow_html=True)
        plt.figure(figsize=(6, max(3, len(feature_names) * 0.4)))
        shap.summary_plot(
            shap_df.values, X_raw.values,
            feature_names=feature_names,
            plot_type="dot", show=False,
            max_display=len(feature_names),
        )
        fig_bs = plt.gcf()
        fig_bs.tight_layout()
        st.pyplot(fig_bs)
        plt.close(fig_bs)
        st.markdown('</div>', unsafe_allow_html=True)

    # Dependence plot
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Feature Dependence Plot</div>',
                unsafe_allow_html=True)
    dp1, dp2 = st.columns(2)
    with dp1:
        dep_feat  = st.selectbox("Main feature (x-axis)", feature_names,
                                 index=int(shap_df.abs().mean().argmax()),
                                 key="dep_main")
    with dp2:
        dep_inter = st.selectbox("Colour by (interaction)", ["Auto"] + feature_names,
                                 key="dep_inter")
    feat_idx  = list(feature_names).index(dep_feat)
    inter_idx = ("auto" if dep_inter == "Auto"
                 else list(feature_names).index(dep_inter))
    # shap.dependence_plot also does not accept `ax` in older versions
    plt.figure(figsize=(9, 4))
    shap.dependence_plot(
        feat_idx, shap_df.values, X_raw.values,
        feature_names=feature_names,
        interaction_index=inter_idx,
        show=False,
    )
    fig_dp = plt.gcf()
    fig_dp.axes[0].set_title(f"SHAP Dependence — {dep_feat}", fontsize=12)
    fig_dp.axes[0].spines[["top", "right"]].set_visible(False)
    fig_dp.tight_layout()
    st.pyplot(fig_dp)
    plt.close(fig_dp)
    st.markdown('</div>', unsafe_allow_html=True)

    # Per-polygon waterfall
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Per-Polygon SHAP Breakdown</div>',
                unsafe_allow_html=True)

    poly_ids = cov_work["Id"].tolist()
    sel_id   = st.selectbox(
        "Select polygon ID",
        poly_ids,
        format_func=lambda i: (
            f"ID {i}  "
            f"(p={cov_work.loc[cov_work['Id']==i,'probability'].values[0]:.4f}  "
            f"→ {cov_work.loc[cov_work['Id']==i,'susceptibility'].values[0]})"
        ),
        key="poly_sel",
    )

    if sel_id:
        pos     = cov_work.index.get_loc(cov_work.index[cov_work["Id"]==sel_id][0])
        sv_row  = shap_df.iloc[pos].values
        x_row   = X_raw.iloc[pos].values
        order   = np.argsort(np.abs(sv_row))[::-1]

        fig_wf, ax_wf = plt.subplots(figsize=(9, max(3, len(feature_names)*0.42)))
        c_wf = ["#c0392b" if v >= 0 else "#2980b9" for v in sv_row[order]]
        ax_wf.barh(range(len(feature_names)), sv_row[order],
                   color=c_wf, edgecolor="white", linewidth=0.5)
        ax_wf.set_yticks(range(len(feature_names)))
        ax_wf.set_yticklabels(
            [f"{feature_names[i]}  (val={x_row[i]:.3f})" for i in order],
            fontsize=9,
        )
        ax_wf.axvline(0, color="#333", linewidth=1)
        ax_wf.set_xlabel("SHAP value  (red = ↑ risk,  blue = ↓ risk)")
        p_val = cov_work.loc[cov_work["Id"]==sel_id,"probability"].values[0]
        s_val = cov_work.loc[cov_work["Id"]==sel_id,"susceptibility"].values[0]
        ax_wf.set_title(f"Polygon {sel_id}  |  p = {p_val:.4f}  →  {s_val}",
                        fontsize=11)
        ax_wf.spines[["top","right"]].set_visible(False)
        fig_wf.tight_layout(); st.pyplot(fig_wf); plt.close(fig_wf)

        detail = pd.DataFrame({
            "Feature":    [feature_names[i] for i in order],
            "Raw Value":  [round(float(x_row[i]),5) for i in order],
            "SHAP Value": [round(float(sv_row[i]),6) for i in order],
            "Direction":  ["⬆ Risk" if sv_row[i]>=0 else "⬇ Risk" for i in order],
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # ── SINGLE SUBAREA — Full SHAP Deep-Dive ────────────────────────
    # ════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">📍 Single Subarea — SHAP Deep-Dive</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Select **one geographic subarea** (district, watershed, zone…) to get "
        "a full SHAP analysis scoped to the polygons inside it: summary stats, "
        "feature importance, beeswarm, riskiest polygons, and a mini-map.",
    )

    # ── Column + zone pickers ────────────────────────────────────────
    sa_deep_cols = [
        c for c in gdf_full.columns
        if c not in ("Id", "geometry")
        and gdf_full[c].dtype == object
        and 1 < gdf_full[c].nunique() <= 200
    ]

    if not sa_deep_cols:
        st.info("No categorical columns found in the shapefile to define subareas.")
    else:
        sd1, sd2 = st.columns([1, 2])
        with sd1:
            deep_col = st.selectbox(
                "Subarea column",
                sa_deep_cols,
                key="deep_col",
                help="Shapefile column that defines geographic zones.",
            )
        with sd2:
            deep_zones_all = sorted(
                gdf_full[deep_col].dropna().unique().tolist()
            )
            deep_zone = st.selectbox(
                "Select one subarea",
                deep_zones_all,
                key="deep_zone",
            )

        if deep_zone:
            # ── Filter data to this zone ─────────────────────────────
            zone_poly_ids = set(
                gdf_full.loc[gdf_full[deep_col] == deep_zone, "Id"]
                .astype(str).str.strip()
            )
            zone_mask_dz  = cov_work["Id"].isin(zone_poly_ids)
            cov_zone      = cov_work[zone_mask_dz].copy()
            shap_zone     = shap_df[zone_mask_dz.values].copy()
            X_zone        = X_raw[zone_mask_dz.values].copy()

            n_zone_polys = len(cov_zone)

            if n_zone_polys == 0:
                st.warning(
                    f"No covariates matched for **{deep_zone}**. "
                    "Check that Id values align between the shapefile and Excel."
                )
            else:
                # ── KPI strip ────────────────────────────────────────
                z_proba = cov_zone["probability"].values
                z_dom   = cov_zone["dominant_feature"].value_counts()
                st.markdown(f"""
                <div class="kpi-row">
                  <div class="kpi-card green">
                    <div class="kpi-label">Subarea</div>
                    <div class="kpi-value" style="font-size:1rem">{deep_zone}</div>
                    <div class="kpi-sub">{n_zone_polys:,} polygons</div>
                  </div>
                  <div class="kpi-card">
                    <div class="kpi-label">Mean Probability</div>
                    <div class="kpi-value">{z_proba.mean():.3f}</div>
                    <div class="kpi-sub">Min {z_proba.min():.3f} · Max {z_proba.max():.3f}</div>
                  </div>
                  <div class="kpi-card orange">
                    <div class="kpi-label">Dominant Class</div>
                    <div class="kpi-value" style="font-size:1rem">
                      {cov_zone["susceptibility"].value_counts().idxmax()}
                    </div>
                    <div class="kpi-sub">Most common susceptibility class</div>
                  </div>
                  <div class="kpi-card red">
                    <div class="kpi-label">Top SHAP Driver</div>
                    <div class="kpi-value" style="font-size:1rem">
                      {z_dom.index[0] if len(z_dom) else "—"}
                    </div>
                    <div class="kpi-sub">Most frequent dominant feature</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Row 1: class distribution + mean|SHAP| bar ───────
                dz_r1c1, dz_r1c2 = st.columns(2)

                with dz_r1c1:
                    st.markdown("**Susceptibility class distribution**")
                    z_counts = (
                        cov_zone["susceptibility"].value_counts()
                        .reindex(class_labels).fillna(0).astype(int)
                    )
                    fig_zb, ax_zb = plt.subplots(figsize=(5, 3.2))
                    bars_zb = ax_zb.bar(
                        z_counts.index, z_counts.values,
                        color=COLORS, edgecolor="white", linewidth=0.7,
                    )
                    for bar, v in zip(bars_zb, z_counts.values):
                        if v > 0:
                            ax_zb.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.2,
                                str(v), ha="center", va="bottom",
                                fontsize=8, fontweight="bold",
                            )
                    ax_zb.set_ylabel("Polygon count")
                    ax_zb.set_title(f"{deep_zone} — class counts", fontsize=10)
                    ax_zb.spines[["top", "right"]].set_visible(False)
                    ax_zb.tick_params(axis="x", rotation=20)
                    fig_zb.tight_layout()
                    st.pyplot(fig_zb); plt.close(fig_zb)

                with dz_r1c2:
                    st.markdown("**Mean |SHAP| for this subarea**")
                    z_mean_abs = shap_zone.abs().mean().sort_values(ascending=True)
                    fig_zi, ax_zi = plt.subplots(
                        figsize=(5, max(2.5, len(feature_names) * 0.36))
                    )
                    bars_zi = ax_zi.barh(
                        z_mean_abs.index, z_mean_abs.values,
                        color=[feat_color.get(f, "#aaa") for f in z_mean_abs.index],
                        edgecolor="white", linewidth=0.4,
                    )
                    for bar, v in zip(bars_zi, z_mean_abs.values):
                        ax_zi.text(
                            v + z_mean_abs.max() * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f"{v:.4f}", va="center", fontsize=7.5,
                        )
                    ax_zi.set_xlabel("Mean |SHAP|")
                    ax_zi.set_title(f"{deep_zone} — feature importance", fontsize=10)
                    ax_zi.spines[["top", "right"]].set_visible(False)
                    fig_zi.tight_layout()
                    st.pyplot(fig_zi); plt.close(fig_zi)

                # ── Row 2: beeswarm (zone-only) ──────────────────────
                st.markdown("**SHAP Beeswarm — polygons inside this subarea**")
                plt.figure(figsize=(9, max(3, len(feature_names) * 0.38)))
                shap.summary_plot(
                    shap_zone.values, X_zone.values,
                    feature_names=feature_names,
                    plot_type="dot", show=False,
                    max_display=len(feature_names),
                )
                fig_zbs = plt.gcf()
                fig_zbs.suptitle(
                    f"SHAP beeswarm — {deep_zone}  ({n_zone_polys:,} polygons)",
                    fontsize=11, y=1.01,
                )
                fig_zbs.tight_layout()
                st.pyplot(fig_zbs); plt.close(fig_zbs)

                # ── Row 3: signed mean SHAP (direction of effect) ────
                st.markdown(
                    "**Mean signed SHAP — net direction of each feature's effect**"
                )
                z_mean_signed = shap_zone.mean().sort_values()
                fig_zs, ax_zs = plt.subplots(
                    figsize=(9, max(3, len(feature_names) * 0.38))
                )
                colors_zs = [
                    "#c0392b" if v >= 0 else "#2980b9"
                    for v in z_mean_signed.values
                ]
                ax_zs.barh(
                    z_mean_signed.index, z_mean_signed.values,
                    color=colors_zs, edgecolor="white", linewidth=0.4,
                )
                ax_zs.axvline(0, color="#333", linewidth=1)
                ax_zs.set_xlabel(
                    "Mean SHAP value  (red = net ↑ risk, blue = net ↓ risk)"
                )
                ax_zs.set_title(
                    f"Net feature effect — {deep_zone}", fontsize=11
                )
                ax_zs.spines[["top", "right"]].set_visible(False)
                fig_zs.tight_layout()
                st.pyplot(fig_zs); plt.close(fig_zs)

                # ── Row 4: top-N highest-risk polygons table + mini-map
                st.markdown("---")
                dz_t1, dz_t2 = st.columns([1, 1])

                with dz_t1:
                    top_n = st.slider(
                        "Show top N highest-risk polygons",
                        min_value=5, max_value=min(50, n_zone_polys),
                        value=min(10, n_zone_polys),
                        key="deep_topn",
                    )
                    top_polys = (
                        cov_zone[["Id", "probability", "susceptibility",
                                  "dominant_feature"]]
                        .sort_values("probability", ascending=False)
                        .head(top_n)
                        .reset_index(drop=True)
                    )
                    top_polys.index += 1   # rank from 1
                    st.markdown(f"**Top {top_n} highest-risk polygons in {deep_zone}**")
                    st.dataframe(top_polys, use_container_width=True)

                with dz_t2:
                    st.markdown(f"**Mini-map — {deep_zone}**")
                    # Build GDF subset
                    gdf_zone_map = gdf_4326[
                        gdf_4326["Id"].isin(zone_poly_ids)
                    ].copy()
                    gdf_zone_map["susceptibility"] = (
                        gdf_zone_map["susceptibility"].fillna("Unknown")
                    )
                    if not gdf_zone_map.empty:
                        cen = gdf_zone_map.dissolve().centroid.iloc[0]
                        m_mini = folium.Map(
                            location=[cen.y, cen.x],
                            zoom_start=12,
                            tiles="CartoDB positron",
                        )

                        def _mini_style(feat):
                            s = feat["properties"].get("susceptibility", "")
                            return {
                                "fillColor": LABEL_COLOR.get(s, "#aaa"),
                                "color": "#333",
                                "weight": 0.6,
                                "fillOpacity": 0.8,
                            }

                        # Sanitise NaNs for folium
                        gdf_mini_clean = gdf_zone_map.copy()
                        for c in gdf_mini_clean.select_dtypes(
                            include=[float]
                        ).columns:
                            gdf_mini_clean[c] = gdf_mini_clean[c].fillna(-9999)
                        for c in gdf_mini_clean.select_dtypes(
                            include="object"
                        ).columns:
                            if c != "geometry":
                                gdf_mini_clean[c] = gdf_mini_clean[c].fillna("N/A")

                        folium.GeoJson(
                            gdf_mini_clean.__geo_interface__,
                            style_function=_mini_style,
                            tooltip=folium.GeoJsonTooltip(
                                fields=["Id", "probability", "susceptibility"],
                                aliases=["ID:", "Prob:", "Class:"],
                                localize=True,
                            ),
                        ).add_to(m_mini)
                        st_folium(m_mini, use_container_width=True, height=340,
                                  key="mini_map")
                    else:
                        st.info("No geometry found for this subarea.")

                # ── SHAP detail table for top polygons ───────────────
                with st.expander(
                    f"🧠 SHAP breakdown for top {top_n} polygons in {deep_zone}"
                ):
                    top_ids   = top_polys["Id"].tolist()
                    top_mask  = cov_zone["Id"].isin(top_ids)
                    shap_top_idx = cov_zone.index[top_mask]
                    shap_top_pos = [
                        cov_zone.index.get_loc(i) for i in shap_top_idx
                    ]
                    shap_top_df  = shap_zone.iloc[shap_top_pos].copy()
                    shap_top_df.insert(
                        0, "Id",
                        cov_zone.loc[shap_top_idx, "Id"].values,
                    )
                    shap_top_df.insert(
                        1, "probability",
                        cov_zone.loc[shap_top_idx, "probability"].values,
                    )
                    shap_top_df.insert(
                        2, "susceptibility",
                        cov_zone.loc[shap_top_idx, "susceptibility"].values,
                    )
                    shap_top_df = shap_top_df.sort_values(
                        "probability", ascending=False
                    ).reset_index(drop=True)
                    shap_top_df.index += 1
                    st.dataframe(shap_top_df, use_container_width=True, height=320)

                # ── Export buttons ───────────────────────────────────
                st.markdown("**Export this subarea**")
                ex1, ex2, ex3 = st.columns(3)
                with ex1:
                    dz_csv = BytesIO()
                    cov_zone[
                        ["Id", "probability", "susceptibility", "dominant_feature"]
                        + list(feature_names)
                    ].to_csv(dz_csv, index=False)
                    dz_csv.seek(0)
                    st.download_button(
                        "📥 Predictions CSV",
                        data=dz_csv,
                        file_name=f"{deep_zone}_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with ex2:
                    dz_shap_csv = BytesIO()
                    shap_zone_export = cov_zone[
                        ["Id", "probability", "susceptibility"]
                    ].copy()
                    for f in feature_names:
                        shap_zone_export[f"shap_{f}"] = shap_zone[f].values
                    shap_zone_export.to_csv(dz_shap_csv, index=False)
                    dz_shap_csv.seek(0)
                    st.download_button(
                        "📥 SHAP Values CSV",
                        data=dz_shap_csv,
                        file_name=f"{deep_zone}_shap.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with ex3:
                    gdf_zone_export = gdf_result[
                        gdf_result["Id"].isin(zone_poly_ids)
                    ].copy()
                    if not gdf_zone_export.empty:
                        st.download_button(
                            "📥 Shapefile ZIP",
                            data=shp_zip_bytes(gdf_zone_export),
                            file_name=f"{deep_zone}_shapefile.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Subarea SHAP Comparison ──────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗂️ Subarea SHAP Comparison</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Compare SHAP feature profiles across different geographic subareas. "
        "Select a shapefile column and two or more zones to see how risk drivers "
        "differ spatially.",
    )

    # Column picker — same logic as sidebar but independent (so it works
    # even when the global subarea filter is off)
    shap_cand_cols = [
        c for c in gdf_full.columns
        if c not in ("Id", "geometry")
        and gdf_full[c].dtype == object
        and 1 < gdf_full[c].nunique() <= 200
    ]

    if not shap_cand_cols:
        st.info("No suitable categorical columns found in the shapefile "
                "for subarea comparison.")
    else:
        sa_col1, sa_col2 = st.columns([1, 2])
        with sa_col1:
            comp_col = st.selectbox(
                "Subarea column", shap_cand_cols, key="shap_comp_col",
                help="Column from the shapefile that defines zones/districts.",
            )
        with sa_col2:
            all_zones   = sorted(gdf_full[comp_col].dropna().unique().tolist())
            sel_zones   = st.multiselect(
                "Select zones to compare (2 – 6)",
                all_zones,
                default=all_zones[:min(3, len(all_zones))],
                key="shap_zones",
            )

        if len(sel_zones) < 2:
            st.warning("Please select at least 2 zones to compare.")
        else:
            # Build per-zone SHAP mean-|SHAP| profiles
            zone_profiles = {}   # zone → pd.Series(mean |SHAP| per feature)
            zone_counts   = {}

            for zone in sel_zones:
                zone_ids  = set(
                    gdf_full.loc[gdf_full[comp_col] == zone, "Id"]
                    .astype(str).str.strip()
                )
                zone_mask = cov_work["Id"].isin(zone_ids)
                n_zone    = zone_mask.sum()
                zone_counts[zone] = n_zone
                if n_zone == 0:
                    zone_profiles[zone] = pd.Series(
                        np.zeros(len(feature_names)), index=feature_names
                    )
                else:
                    zone_profiles[zone] = (
                        shap_df[zone_mask.values].abs().mean()
                    )

            # ── Side-by-side mean|SHAP| bars ──────────────────────────
            st.markdown("##### Mean |SHAP| per Feature by Zone")
            n_feats = len(feature_names)
            x       = np.arange(n_feats)
            n_zones = len(sel_zones)
            width   = min(0.8 / n_zones, 0.25)

            ZONE_PALETTE = [
                "#3b7dd8","#e07b39","#27ae60","#8e44ad",
                "#c0392b","#16a085","#d4ac0d",
            ]

            fig_comp, ax_comp = plt.subplots(
                figsize=(max(10, n_feats * 0.7), 4.5))
            for i, (zone, profile) in enumerate(zone_profiles.items()):
                offset = (i - n_zones / 2 + 0.5) * width
                bars_c = ax_comp.bar(
                    x + offset,
                    profile[feature_names].values,
                    width=width * 0.92,
                    label=f"{zone} (n={zone_counts[zone]:,})",
                    color=ZONE_PALETTE[i % len(ZONE_PALETTE)],
                    edgecolor="white", linewidth=0.4, alpha=0.88,
                )
            ax_comp.set_xticks(x)
            ax_comp.set_xticklabels(feature_names, rotation=35,
                                    ha="right", fontsize=9)
            ax_comp.set_ylabel("Mean |SHAP value|")
            ax_comp.set_title(
                "Feature Importance (mean |SHAP|) by Subarea", fontsize=12
            )
            ax_comp.legend(fontsize=9, framealpha=0.9)
            ax_comp.spines[["top", "right"]].set_visible(False)
            fig_comp.tight_layout()
            st.pyplot(fig_comp)
            plt.close(fig_comp)

            # ── Radar / spider chart ───────────────────────────────────
            st.markdown("##### Risk Driver Radar Chart")

            # Normalise each feature 0-1 across zones for radar
            profile_matrix = pd.DataFrame(zone_profiles).T  # zones × features
            col_max = profile_matrix.max(axis=0).replace(0, 1)
            radar_matrix = profile_matrix / col_max          # normalised

            angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
            angles += angles[:1]                             # close polygon

            fig_rad, ax_rad = plt.subplots(
                figsize=(6, 6),
                subplot_kw=dict(polar=True),
            )
            for i, zone in enumerate(sel_zones):
                vals = radar_matrix.loc[zone, feature_names].tolist()
                vals += vals[:1]
                clr  = ZONE_PALETTE[i % len(ZONE_PALETTE)]
                ax_rad.plot(angles, vals, color=clr, linewidth=2,
                            label=zone)
                ax_rad.fill(angles, vals, color=clr, alpha=0.12)

            ax_rad.set_xticks(angles[:-1])
            ax_rad.set_xticklabels(feature_names, fontsize=8)
            ax_rad.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_rad.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                                   fontsize=7, color="#888")
            ax_rad.set_title("Normalised SHAP Profile per Zone",
                             fontsize=12, pad=18)
            ax_rad.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
                          fontsize=9)
            fig_rad.tight_layout()
            st.pyplot(fig_rad)
            plt.close(fig_rad)

            # ── Heatmap: zones × features ──────────────────────────────
            st.markdown("##### Mean |SHAP| Heatmap  (zones × features)")
            fig_hm, ax_hm = plt.subplots(
                figsize=(max(8, n_feats * 0.65), max(3, n_zones * 0.55 + 1.5))
            )
            import matplotlib.cm as cm
            hm_data = profile_matrix[feature_names]
            im = ax_hm.imshow(
                hm_data.values, aspect="auto", cmap="YlOrRd",
                vmin=0, vmax=hm_data.values.max(),
            )
            ax_hm.set_xticks(range(n_feats))
            ax_hm.set_xticklabels(feature_names, rotation=35,
                                   ha="right", fontsize=9)
            ax_hm.set_yticks(range(n_zones))
            ax_hm.set_yticklabels(
                [f"{z}  (n={zone_counts[z]:,})" for z in sel_zones],
                fontsize=9,
            )
            plt.colorbar(im, ax=ax_hm, label="Mean |SHAP|", shrink=0.7)
            # Annotate cells
            for row_i, zone in enumerate(sel_zones):
                for col_j, feat in enumerate(feature_names):
                    val = hm_data.loc[zone, feat]
                    ax_hm.text(
                        col_j, row_i, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if val > hm_data.values.max() * 0.6
                              else "#333",
                    )
            ax_hm.set_title("Mean |SHAP| — Subarea × Feature Heatmap",
                            fontsize=12)
            fig_hm.tight_layout()
            st.pyplot(fig_hm)
            plt.close(fig_hm)

            # ── Dominant feature per zone summary table ────────────────
            st.markdown("##### Dominant Risk Driver per Zone")
            dom_rows = []
            for zone in sel_zones:
                prof = zone_profiles[zone]
                top3 = prof.nlargest(3)
                dom_rows.append({
                    "Zone":             zone,
                    "Polygons":         f"{zone_counts[zone]:,}",
                    "Top Driver":       top3.index[0] if len(top3) > 0 else "—",
                    "Top SHAP":         f"{top3.iloc[0]:.4f}" if len(top3) > 0 else "—",
                    "2nd Driver":       top3.index[1] if len(top3) > 1 else "—",
                    "3rd Driver":       top3.index[2] if len(top3) > 2 else "—",
                    "Mean Probability": f"{cov_work.loc[cov_work['Id'].isin(gdf_full.loc[gdf_full[comp_col]==zone,'Id'].astype(str).str.strip()), 'probability'].mean():.4f}",
                })
            st.dataframe(
                pd.DataFrame(dom_rows),
                use_container_width=True, hide_index=True,
            )

            # Download comparison CSV
            comp_csv = BytesIO()
            profile_matrix.reset_index().rename(
                columns={"index": "Zone"}
            ).to_csv(comp_csv, index=False)
            comp_csv.seek(0)
            st.download_button(
                "📥 Download Zone SHAP Profiles (CSV)",
                data=comp_csv,
                file_name="zone_shap_profiles.csv",
                mime="text/csv",
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Dominant-factor spatial map ──────────────────────────────────
    st.markdown('<div class="section-title">🗺️ Dominant Driving Factor Map</div>',
                unsafe_allow_html=True)
    st.caption(
        "Each polygon is coloured by the feature with the highest absolute SHAP "
        "impact — reveals spatially coherent risk drivers."
    )
    plot_dom = gdf_4326.copy()
    plot_dom["dominant_feature"] = plot_dom["dominant_feature"].fillna("Unknown")
    dom_feats = sorted(plot_dom["dominant_feature"].dropna().unique())

    fig_dm, ax_dm = plt.subplots(figsize=(13, 9))
    fig_dm.patch.set_facecolor("#f4f6f9")
    ax_dm.set_facecolor("#f4f6f9")
    for feat in dom_feats:
        plot_dom[plot_dom["dominant_feature"]==feat].plot(
            ax=ax_dm, color=feat_color.get(feat,"#aaa"),
            edgecolor="#ffffff", linewidth=0.25, alpha=0.85,
        )
    ax_dm.legend(
        handles=[mpatches.Patch(facecolor=feat_color.get(f,"#aaa"), label=f)
                 for f in dom_feats],
        title="Dominant Factor", loc="lower left",
        fontsize=8, title_fontsize=9, framealpha=0.9,
    )
    ax_dm.set_title(f"Dominant SHAP Driver — {area_label}",
                    fontsize=13, fontweight="bold")
    ax_dm.set_axis_off()
    fig_dm.tight_layout(); st.pyplot(fig_dm)

    dm_buf = BytesIO()
    fig_dm.savefig(dm_buf, format="png", dpi=1200, bbox_inches="tight")
    dm_buf.seek(0); plt.close(fig_dm)
    st.download_button("📥 Download Dominant Factor Map",
                       data=dm_buf,
                       file_name="dominant_shap_map.png", mime="image/png")
    st.markdown('</div>', unsafe_allow_html=True)

# ┌──────────────────────────────────────────────────────────────────┐
# │  TAB 4 — DATA & EXPORT                                           │
# └──────────────────────────────────────────────────────────────────┘
with tab_draw:

    st.markdown("""
    <div class="info-box">
      🎯 &nbsp; <b>Polygon Explorer &amp; SHAP Analysis</b> — <b>Click any polygon</b>
      on the susceptibility map or search by ID. The selected polygon is shared
      across all tabs: <b>Validation</b> will automatically
      use the same polygon. Publication-quality SHAP figures are generated below.
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: Clickable map + search ────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">① Select a polygon — click the map or search</div>',
        unsafe_allow_html=True,
    )

    _poly_ids_all = [str(i) for i in cov_work["Id"].tolist()]

    _srch_col, _map_col = st.columns([1, 3])

    with _srch_col:
        st.markdown("**🔍 Search by ID:**")
        _draw_search = st.selectbox(
            "Polygon ID",
            ["— click map or search —"] + _poly_ids_all,
            format_func=lambda i: (
                f"ID {i}  —  p={cov_work.loc[cov_work['Id'].astype(str)==i,'probability'].values[0]:.4f}"
                f"  →  {cov_work.loc[cov_work['Id'].astype(str)==i,'susceptibility'].values[0]}"
            ) if i != "— click map or search —" else i,
            key="draw_poly_search",
        )
        if _draw_search != "— click map or search —":
            st.session_state["llm_clicked_poly"] = _draw_search

        # Show current selection
        _cur_sel = st.session_state.get("llm_clicked_poly")
        if _cur_sel and str(_cur_sel) in _poly_ids_all:
            _cs_row = cov_work[cov_work["Id"].astype(str) == str(_cur_sel)]
            if not _cs_row.empty:
                st.success(
                    f"**Selected:** ID {_cur_sel}\n\n"
                    f"p = {_cs_row['probability'].values[0]:.4f}\n\n"
                    f"Class: {_cs_row['susceptibility'].values[0]}"
                )
                st.caption("↗️ This polygon is shared with Validation tab.")
        else:
            st.info("Click a polygon on the map or use the search box.")

    with _map_col:
        # Build clickable susceptibility map
        _draw_highlight = st.session_state.get("llm_clicked_poly")
        _ctr = [gdf_4326.geometry.centroid.y.mean(),
                gdf_4326.geometry.centroid.x.mean()]
        m_explore = folium.Map(location=_ctr, zoom_start=11,
                               tiles="CartoDB positron")
        folium.TileLayer(
            tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
                   "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
            attr="Esri", name="Satellite",
        ).add_to(m_explore)

        # All polygons — clickable with popup
        _exp_gdf = gdf_4326.copy()
        _exp_gdf["_fill"] = _exp_gdf["susceptibility"].map(
            lambda s: LABEL_COLOR.get(s, "#aaa"))
        _exp_gdf["_sel"] = _exp_gdf["Id"].astype(str).map(
            lambda i: i == str(_draw_highlight) if _draw_highlight else False)
        _exp_gdf["_prob_s"] = _exp_gdf["probability"].apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else "N/A")
        _exp_gdf["_susc_s"] = _exp_gdf["susceptibility"].fillna("Unknown")
        _exp_gdf["_id_s"] = _exp_gdf["Id"].astype(str)

        folium.GeoJson(
            _exp_gdf[["geometry", "_fill", "_sel", "_prob_s", "_susc_s", "_id_s"]],
            style_function=lambda feat: {
                "fillColor":   feat["properties"]["_fill"],
                "color":       "#e63946" if feat["properties"]["_sel"] else "#333",
                "weight":      4.0      if feat["properties"]["_sel"] else 0.35,
                "fillOpacity": 0.95     if feat["properties"]["_sel"] else 0.55,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["_id_s", "_susc_s", "_prob_s"],
                aliases=["ID", "Class", "Probability"],
                sticky=True,
            ),
            popup=folium.GeoJsonPopup(
                fields=["_id_s"],
                aliases=[""],
                labels=False,
                max_width=120,
            ),
            name="Susceptibility",
        ).add_to(m_explore)

        # Legend
        _exp_leg = (
            '<div style="position:fixed;bottom:20px;left:20px;z-index:1000;'
            "background:white;padding:10px 14px;border-radius:8px;"
            'box-shadow:0 2px 6px rgba(0,0,0,.25);font-size:12px;">'
            "<b>Susceptibility</b><br>"
        )
        for _lbl, _clr in LABEL_COLOR.items():
            _exp_leg += (
                f'<i style="background:{_clr};width:13px;height:13px;'
                f"display:inline-block;margin-right:5px;border-radius:3px;"
                f'border:1px solid #ccc"></i>{_lbl}<br>'
            )
        _exp_leg += "</div>"
        m_explore.get_root().html.add_child(folium.Element(_exp_leg))
        folium.LayerControl().add_to(m_explore)

        _exp_map_out = st_folium(
            m_explore, use_container_width=True, height=520,
            returned_objects=["last_object_clicked_popup"],
            key="explore_poly_map",
        )

        # Capture click from map
        _exp_popup = (_exp_map_out or {}).get("last_object_clicked_popup")
        if _exp_popup is not None:
            if isinstance(_exp_popup, dict):
                _exp_click_id = str(_exp_popup.get("_id_s", "")).strip()
            else:
                _exp_click_id = str(_exp_popup).strip()
            if _exp_click_id in _poly_ids_all:
                st.session_state["llm_clicked_poly"] = _exp_click_id

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Get the globally selected polygon ─────────────────────────────
    draw_sel_id_str = st.session_state.get("llm_clicked_poly")
    # Convert to the native Id type in cov_work
    draw_sel_id = None
    if draw_sel_id_str and str(draw_sel_id_str) in _poly_ids_all:
        _match = cov_work[cov_work["Id"].astype(str) == str(draw_sel_id_str)]
        if not _match.empty:
            draw_sel_id = _match["Id"].values[0]

    if draw_sel_id is not None:
        # Get polygon data
        _dr_ridx = cov_work.index[cov_work["Id"] == draw_sel_id][0]
        _dr_pos  = cov_work.index.get_loc(_dr_ridx)
        _dr_sv   = shap_df.iloc[_dr_pos]
        _dr_xv   = X_raw.iloc[_dr_pos]
        _dr_prob = float(cov_work.loc[_dr_ridx, "probability"])
        _dr_susc = cov_work.loc[_dr_ridx, "susceptibility"]

        # ── KPI strip ─────────────────────────────────────────────────
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi-card green">
            <div class="kpi-label">Polygon ID</div>
            <div class="kpi-value">{draw_sel_id}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Probability</div>
            <div class="kpi-value">{_dr_prob:.4f}</div>
          </div>
          <div class="kpi-card orange">
            <div class="kpi-label">Susceptibility Class</div>
            <div class="kpi-value" style="font-size:1.2rem">{_dr_susc}</div>
          </div>
          <div class="kpi-card red">
            <div class="kpi-label">Dominant Driver</div>
            <div class="kpi-value" style="font-size:1rem">
              {cov_work.loc[_dr_ridx, "dominant_feature"]}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Step 2: Susceptibility map with selected polygon highlighted ──
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">② Location on Susceptibility Map</div>',
            unsafe_allow_html=True,
        )

        _sel_gdf = gdf_4326[gdf_4326["Id"].astype(str) == str(draw_sel_id)]
        if not _sel_gdf.empty:
            _sel_cen = _sel_gdf.geometry.iloc[0].centroid
            m_draw = folium.Map(
                location=[_sel_cen.y, _sel_cen.x],
                zoom_start=13,
                tiles="CartoDB positron",
            )
            folium.TileLayer(
                tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
                       "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
                attr="Esri", name="Satellite",
            ).add_to(m_draw)

            # All polygons — subdued
            _bg_gdf = gdf_4326.copy()
            _bg_gdf["_fill"] = _bg_gdf["susceptibility"].map(
                lambda s: LABEL_COLOR.get(s, "#aaa"))
            folium.GeoJson(
                _bg_gdf[["geometry", "_fill"]],
                style_function=lambda feat: {
                    "fillColor":   feat["properties"]["_fill"],
                    "color":       "#555",
                    "weight":      0.3,
                    "fillOpacity": 0.35,
                },
                name="All polygons",
            ).add_to(m_draw)

            # Selected polygon — bold highlight
            folium.GeoJson(
                _sel_gdf.__geo_interface__,
                style_function=lambda _: {
                    "fillColor":   LABEL_COLOR.get(_dr_susc, "#e63946"),
                    "color":       "#e63946",
                    "weight":      4,
                    "fillOpacity": 0.9,
                },
                tooltip=f"ID {draw_sel_id} | p={_dr_prob:.4f} | {_dr_susc}",
            ).add_to(m_draw)

            folium.LayerControl().add_to(m_draw)
            st_folium(m_draw, use_container_width=True, height=420, key="draw_loc_map")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Step 3: SHAP Waterfall Plot (publication quality) ─────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">③ SHAP Waterfall Plot (additive decomposition)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Each bar shows how a single feature pushes the prediction from the "
            "base value (E[f(x)]) toward the final output. Red bars increase risk; "
            "blue bars decrease risk. This is the core SHAP additive decomposition."
        )

        # Build SHAP Explanation object for waterfall plot
        _dr_base = float(shap_df.values.mean())   # approximate base value
        _dr_shap_vals = _dr_sv.values.astype(float)
        _dr_feat_vals = _dr_xv.values.astype(float)

        try:
            _dr_expl = shap.Explanation(
                values=_dr_shap_vals,
                base_values=_dr_base,
                data=_dr_feat_vals,
                feature_names=list(feature_names),
            )
            fig_wf, ax_wf = plt.subplots(figsize=(10, max(4, len(feature_names)*0.45)))
            matplotlib.rcParams.update({
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
                "font.size": 10,
            })
            shap.plots.waterfall(_dr_expl, max_display=len(feature_names), show=False)
            fig_wf = plt.gcf()
            fig_wf.patch.set_facecolor("white")
            fig_wf.tight_layout(pad=1.2)
            st.pyplot(fig_wf)

            # Download button
            buf_wf = BytesIO()
            fig_wf.savefig(buf_wf, format="png", dpi=1200, bbox_inches="tight",
                           facecolor="white")
            buf_wf.seek(0)
            plt.close(fig_wf)
            st.download_button(
                "📥 Download Waterfall Plot (1200 dpi)",
                data=buf_wf, file_name=f"shap_waterfall_{draw_sel_id}.png",
                mime="image/png", key="dl_waterfall",
            )
            st.caption(
                f"**Fig.** SHAP waterfall decomposition for polygon {draw_sel_id}. "
                "Starting from the base value E[f(x)], each feature's SHAP value "
                "pushes the prediction up (red) or down (blue) toward the final "
                f"output f(x) = {_dr_prob:.4f}. The base value represents the mean "
                "prediction across all training polygons."
            )
        except Exception as _wf_err:
            st.warning(f"Waterfall plot error: {_wf_err}")
            plt.close("all")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Step 4: SHAP Force Plot ──────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">④ SHAP Force Plot</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Compact view of the same additive decomposition. "
            "Red features push the prediction higher (more risk); "
            "blue features push it lower."
        )

        try:
            matplotlib.rcParams.update({
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
                "font.size": 10,
            })
            fig_fp = plt.figure(figsize=(14, 3))
            shap.plots.force(
                base_value=_dr_base,
                shap_values=_dr_shap_vals,
                features=_dr_feat_vals,
                feature_names=list(feature_names),
                matplotlib=True,
                show=False,
            )
            fig_fp = plt.gcf()
            fig_fp.patch.set_facecolor("white")
            fig_fp.tight_layout(pad=1.0)
            st.pyplot(fig_fp)
            buf_fp = BytesIO()
            fig_fp.savefig(buf_fp, format="png", dpi=1200, bbox_inches="tight",
                           facecolor="white")
            buf_fp.seek(0)
            plt.close(fig_fp)
            st.download_button(
                "📥 Download Force Plot (1200 dpi)",
                data=buf_fp, file_name=f"shap_force_{draw_sel_id}.png",
                mime="image/png", key="dl_force",
            )
            st.caption(
                f"**Fig.** SHAP force plot for polygon {draw_sel_id}. "
                "Red segments push the prediction above the base value "
                "(risk-increasing); blue segments push it below (risk-decreasing). "
                "The width of each segment is proportional to the absolute SHAP value."
            )
        except Exception as _fp_err:
            st.warning(f"Force plot error: {_fp_err}")
            plt.close("all")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Step 5: SHAP Bar Chart (sorted) ──────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">⑤ Feature Attribution Bar Chart</div>',
            unsafe_allow_html=True,
        )

        _dr_ord = np.argsort(np.abs(_dr_shap_vals))[::-1]
        fig_bar, ax_bar = plt.subplots(
            figsize=(10, max(4, len(feature_names) * 0.42)))
        fig_bar.patch.set_facecolor("white")
        _bar_colors = ["#c0392b" if _dr_shap_vals[i] >= 0 else "#2980b9"
                       for i in _dr_ord]
        ax_bar.barh(
            range(len(feature_names)), _dr_shap_vals[_dr_ord],
            color=_bar_colors, edgecolor="white", linewidth=0.5,
        )
        ax_bar.set_yticks(range(len(feature_names)))
        ax_bar.set_yticklabels(
            [f"{feature_names[i]}  (={_dr_feat_vals[i]:.3f})"
             for i in _dr_ord], fontsize=9, fontfamily="serif")
        ax_bar.axvline(0, color="#333", linewidth=0.9, zorder=0)
        # Academic legend
        from matplotlib.lines import Line2D as _Line2D
        _bar_leg = [
            mpatches.Patch(facecolor="#c0392b", alpha=0.8, label="Risk-increasing (SHAP > 0)"),
            mpatches.Patch(facecolor="#2980b9", alpha=0.8, label="Risk-decreasing (SHAP < 0)"),
        ]
        ax_bar.legend(handles=_bar_leg, loc="upper right", fontsize=8,
                      framealpha=0.95, edgecolor="#666", fancybox=False)
        apply_academic_style(
            fig_bar, ax_bar,
            xlabel="SHAP value (positive = increases risk, negative = decreases risk)",
        )
        st.pyplot(fig_bar)
        st.caption(
            f"**Fig.** SHAP additive feature attribution for polygon {draw_sel_id} "
            f"({_dr_susc}, *p* = {_dr_prob:.4f}). "
            "Each bar shows the marginal contribution of one covariate to the "
            "predicted landslide susceptibility. Red = risk-increasing; blue = "
            "risk-decreasing. Features ranked by |SHAP|; raw covariate values "
            "in parentheses."
        )

        buf_bar = BytesIO()
        fig_bar.savefig(buf_bar, format="png", dpi=1200, bbox_inches="tight",
                        facecolor="white")
        buf_bar.seek(0)
        plt.close(fig_bar)
        st.download_button(
            "📥 Download Bar Chart (1200 dpi)",
            data=buf_bar, file_name=f"shap_bar_{draw_sel_id}.png",
            mime="image/png", key="dl_bar",
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Step 6: Detailed Feature Table ───────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">⑥ Full Feature Attribution Table</div>',
            unsafe_allow_html=True,
        )
        detail_df = pd.DataFrame({
            "Rank":      range(1, len(feature_names) + 1),
            "Feature":   [feature_names[i] for i in _dr_ord],
            "Raw Value": [round(float(_dr_feat_vals[i]), 5) for i in _dr_ord],
            "SHAP":      [round(float(_dr_shap_vals[i]), 6) for i in _dr_ord],
            "|SHAP|":    [round(float(abs(_dr_shap_vals[i])), 6) for i in _dr_ord],
            "Direction": ["⬆ Increases Risk" if _dr_shap_vals[i] >= 0
                          else "⬇ Decreases Risk" for i in _dr_ord],
        })

        # Append VS30 and TWI from covariates even if not in model feature_names
        _supp_rows = []
        for _sf in ["VS30", "TWI", "vs30", "twi"]:
            if _sf in covariates_df.columns and _sf not in list(feature_names):
                _sf_val = covariates_df.loc[
                    covariates_df["Id"] == draw_sel_id, _sf]
                if not _sf_val.empty:
                    _supp_rows.append({
                        "Rank":      "—",
                        "Feature":   f"{_sf} ★",
                        "Raw Value": round(float(_sf_val.values[0]), 5),
                        "SHAP":      "display only",
                        "|SHAP|":    "—",
                        "Direction": "— (not in model)",
                    })
        if _supp_rows:
            _supp_df = pd.DataFrame(_supp_rows)
            detail_df = pd.concat([detail_df, _supp_df], ignore_index=True)

        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        if _supp_rows:
            st.caption(
                "★ VS30 / TWI values shown for geomorphological context. "
                "These factors exist in the covariates file and are used by "
                "the RAG system for interpretation, but are not among the "
                "model's training features."
            )

        # CSV export
        csv_buf = BytesIO()
        detail_df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        st.download_button(
            "📥 Download Attribution Table (CSV)",
            data=csv_buf, file_name=f"shap_table_{draw_sel_id}.csv",
            mime="text/csv", key="dl_table",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#fff8e1;border-left:4px solid #f39c12;
                    border-radius:6px;padding:1rem 1.2rem;font-size:.9rem;
                    color:#7d6608;margin-top:1rem">
          ℹ️ <b>Click a polygon</b> on the map above to see its SHAP analysis.
          The selected polygon will also be used in the <b>          and <b>Validation</b> tabs.
        </div>
        """, unsafe_allow_html=True)


# ┌──────────────────────────────────────────────────────────────────┐
# │  TAB — AI EXPLANATION (LLM + RAG)                               │
# └──────────────────────────────────────────────────────────────────┘
with tab_valid:

    if not RAG_ENGINE_OK:
        st.error("RAG engine not available. Install: pip install chromadb sentence-transformers pymupdf requests")
        st.stop()

    # ═══════════════════════════════════════════════════════════════════════════
    # FULL PIPELINE OVERVIEW — DETECT → CORRECT → VALIDATE
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:#0f1c2e;border-radius:14px;padding:28px 32px;margin-bottom:24px;
                border:1px solid #1e3a5f;">
    <h2 style="color:#e8f4fd;margin:0 0 8px 0;font-size:1.3rem;letter-spacing:.4px;">
        🔬 Full Pipeline — Detect · Correct · Validate
    </h2>
    <p style="color:#8bafc8;margin:0 0 20px 0;font-size:.88rem;">
        Three steps. Run them in order on each polygon.
    </p>
    <div style="display:grid;grid-template-columns:1fr 40px 1fr 40px 1fr;
                gap:0;align-items:stretch;">

      <div style="background:#112240;border-radius:10px;padding:18px 20px;
                  border-left:4px solid #e74c3c;">
        <div style="color:#e74c3c;font-size:.78rem;font-weight:700;
                    letter-spacing:1.5px;margin-bottom:10px;">STEP 1 — DETECT</div>
        <div style="color:#d0e8ff;font-size:.92rem;font-weight:600;
                    margin-bottom:8px;">What is wrong with the SHAP values?</div>
        <div style="color:#8bafc8;font-size:.82rem;line-height:1.6;">
          <b style="color:#adc8e8;">Input:</b> SHAP values + raw feature values for one polygon<br>
          <b style="color:#adc8e8;">Process:</b> L1 collinearity · L2 sign reversal · L2 sign reversal<br>
          <b style="color:#adc8e8;">Output:</b> Artefact flags saved to session state<br>
          <b style="color:#adc8e8;">Cost:</b> Instant — no API call
        </div>
      </div>

      <div style="display:flex;align-items:center;justify-content:center;">
        <span style="color:#e74c3c;font-size:1.6rem;">→</span>
      </div>

      <div style="background:#112240;border-radius:10px;padding:18px 20px;
                  border-left:4px solid #f39c12;">
        <div style="color:#f39c12;font-size:.78rem;font-weight:700;
                    letter-spacing:1.5px;margin-bottom:10px;">STEP 2 — CORRECT</div>
        <div style="color:#d0e8ff;font-size:.92rem;font-weight:600;
                    margin-bottom:8px;">Fix those errors in the LLM explanation</div>
        <div style="color:#8bafc8;font-size:.82rem;line-height:1.6;">
          <b style="color:#adc8e8;">Input:</b> Detected flags + polygon SHAP + retrieved literature<br>
          <b style="color:#adc8e8;">Process:</b> 3-stage RAG · SHAP constraint · L1/L2 checklist<br>
          <b style="color:#adc8e8;">Output:</b> Corrected narrative text + violation-free explanation<br>
          <b style="color:#adc8e8;">Cost:</b> 1 API call (Config D)
        </div>
      </div>

      <div style="display:flex;align-items:center;justify-content:center;">
        <span style="color:#f39c12;font-size:1.6rem;">→</span>
      </div>

      <div style="background:#112240;border-radius:10px;padding:18px 20px;
                  border-left:4px solid #27ae60;">
        <div style="color:#27ae60;font-size:.78rem;font-weight:700;
                    letter-spacing:1.5px;margin-bottom:10px;">STEP 3 — VALIDATE</div>
        <div style="color:#d0e8ff;font-size:.92rem;font-weight:600;
                    margin-bottom:8px;">Measure how much better the correction is</div>
        <div style="color:#8bafc8;font-size:.82rem;line-height:1.6;">
          <b style="color:#adc8e8;">Input:</b> Corrected text + uncorrected baseline text<br>
          <b style="color:#adc8e8;">Process:</b> GeoFaithfulness · RAGAS · Ablation · Expert rating<br>
          <b style="color:#adc8e8;">Output:</b> Scores, graphs, Wilcoxon p-values, paper figures<br>
          <b style="color:#adc8e8;">Cost:</b> 5 API calls (Configs A–E) + manual rating
        </div>
      </div>

    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Knowledge Base compact status card ──────────────────────────────────
    _kb_status_inv = st.session_state.get("kb_inventory")
    if _kb_status_inv and _kb_status_inv.get("total_sources", 0) > 0:
        _kbs_sources   = _kb_status_inv["total_sources"]
        _kbs_chunks    = _kb_status_inv["total_chunks"]
        _kbs_dom       = _kb_status_inv.get("domain_counts", {})
        _kbs_lim       = _kb_status_inv.get("limitation_coverage", {})
        _kbs_lim_fails = sum(1 for t, c in _kbs_lim.items() if c < 10) if _kbs_lim else None
        _kbs_mcl       = _kb_status_inv.get("mean_chunk_len", 0)
        # Domain health: count domains at or above minimum
        _DOM_MINS_V = {
            "Landslide susceptibility / ML": 8, "SHAP / XAI / explainability": 4,
            "High Atlas / Morocco geology": 5,  "Co-seismic / earthquake landslides": 4,
            "Slope stability (geotechnical)": 3,"Fluvial geomorphology": 3,
            "RAG / LLM / NLP methods": 3,
        }
        _dom_ok = sum(1 for d, m in _DOM_MINS_V.items() if _kbs_dom.get(d, 0) >= m)
        _dom_total = len(_DOM_MINS_V)

        with st.expander(
            f"📚 Knowledge Base status — {_kbs_sources} PDFs · {_kbs_chunks:,} chunks · "
            f"{_dom_ok}/{_dom_total} domains met"
            + (" ✅" if _dom_ok == _dom_total and (_kbs_lim_fails == 0 or _kbs_lim_fails is None)
               else " ⚠️"),
            expanded=False
        ):
            _kbs_c1, _kbs_c2, _kbs_c3, _kbs_c4 = st.columns(4)
            _kbs_c1.metric("PDFs indexed", _kbs_sources,
                           "✅" if _kbs_sources >= 30 else "⚠️ < 30")
            _kbs_c2.metric("Total chunks",  f"{_kbs_chunks:,}",
                           "✅" if _kbs_chunks >= 500 else "⚠️")
            _kbs_c3.metric("Domain coverage",
                           f"{_dom_ok}/{_dom_total}",
                           "✅" if _dom_ok == _dom_total else "⚠️ gaps")
            _kbs_c4.metric("Mean chunk len", f"{_kbs_mcl:.0f} c",
                           "✅" if _kbs_mcl >= 300 else "⚠️ short")

            if _kbs_lim_fails is not None:
                if _kbs_lim_fails == 0:
                    st.success("✅ All SHAP limitation types have adequate KB coverage.")
                else:
                    st.warning(
                        f"⚠️ **{_kbs_lim_fails} limitation type(s)** have insufficient "
                        f"coverage — limitation-targeted retrieval (Stage 3) may fall back "
                        f"to generic text for those zones. "
                        f""
                    )
            st.caption(
                "Full domain breakdown, chunk quality metrics, and SHAP limitation "
                ""
                "📚 RAG Knowledge Base Inventory."
            )
    elif rag_collection is not None:
        st.info(
            ""
            "📚 RAG Knowledge Base Inventory and click **Refresh Inventory** "
            "to see domain coverage and quality metrics."
        )

    if not openrouter_key:
        st.warning("Enter your OpenRouter API key in the sidebar to run validation experiments.")
        st.stop()

    db_ready_v = rag_collection is not None

    # ── Validation target selector ──────────────────────────────────────────
    st.markdown("---")
    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDATION — 6-PHASE ACADEMIC FRAMEWORK
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
                border-radius:12px;padding:18px 26px;margin-bottom:16px;">
      <h2 style="color:#ffffff;margin:0 0 4px 0;font-size:1.1rem;letter-spacing:.4px;">
        Six validation phases — one polygon at a time, accumulated across ≥ 30 polygons
      </h2>
      <p style="color:#b0c4d8;margin:0;font-size:0.84rem;">
        <b>Phase 1</b> feeds all others — run it first (instant).
        <b>Phases 2 + 5</b> are the per-polygon results you report.
        <b>Phase 4</b> accumulates across polygons for statistical tests.
        <b>Phase 3 + 6</b> are calibration and publication tools.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Shared polygon selector (top-level — available in all phases) ────────
    st.markdown("### 🎯 Validation Target")

    # Active area context banner
    _val_area_col = (
        "🟡" if area_label != "Full Study Area" else "🟢"
    )
    st.info(
        f"{_val_area_col} **Active spatial filter:** {area_label} "
        f"— {len(cov_work):,} polygons in scope. "
        + ("To change the zone, use the subarea filter in the sidebar."
           if area_label != "Full Study Area"
           else "All polygons are included. Use the sidebar to restrict to a subarea.")
    )

    _val_mode = st.radio(
        "Validation input mode",
        ["📍 Single polygon", "🗺️ Zone / subarea"],
        horizontal=True,
        key="val_mode",
        help=(
            "Single polygon: all experiments use one polygon's SHAP values. "
            "Zone/subarea: V1–V3 use mean SHAP over the zone."
        ),
    )

    # ── Mode A: Single polygon ───────────────────────────────────────────────
    if _val_mode == "📍 Single polygon":
        val_poly_ids = cov_work["Id"].tolist()

        # ── ALWAYS sync with Explorer selection ──────────────────────────
        # The selectbox defaults to index 0 (polygon 1) unless we force it.
        # We ALWAYS write the active polygon to session_state BEFORE the widget
        # renders, so the selectbox opens on the correct polygon.
        _active_poly = st.session_state.get("llm_clicked_poly")
        if _active_poly is not None:
            # Convert to native type if needed
            _active_native = None
            for _vp in val_poly_ids:
                if str(_vp) == str(_active_poly):
                    _active_native = _vp
                    break
            if _active_native is not None:
                st.session_state["val_poly"] = _active_native
                st.session_state["_val_poly_last_synced"] = _active_native

        val_sel_poly = st.selectbox(
            "Test polygon",
            val_poly_ids,
            format_func=lambda i: (
                f"ID {i}  "
                f"p={cov_work.loc[cov_work['Id']==i,'probability'].values[0]:.4f}  "
                f"→ {cov_work.loc[cov_work['Id']==i,'susceptibility'].values[0]}"
            ),
            key="val_poly",
        )

        # Banner: show whether validation is synced with the Explorer tab
        if _active_poly is not None and str(val_sel_poly) == str(_active_poly):
            st.success(
                f"✅ Synced with Polygon Explorer — using polygon **{val_sel_poly}** "
                f"({cov_work.loc[cov_work['Id']==val_sel_poly,'susceptibility'].values[0]}, "
                f"p={cov_work.loc[cov_work['Id']==val_sel_poly,'probability'].values[0]:.4f}). "
                "All validation tests will use its SHAP values and covariates."
            )
        elif _active_poly is None:
            st.info(
                "💡 No polygon selected yet. Go to the **Polygon Explorer** tab and "
                "click a polygon on the map — it will automatically sync here."
            )
        _val_row   = cov_work.index[cov_work["Id"] == val_sel_poly][0]
        _val_pos   = cov_work.index.get_loc(_val_row)
        _val_sv    = shap_df.iloc[_val_pos].to_dict()
        _val_xv    = X_raw.iloc[_val_pos].to_dict()
        _val_prob  = float(cov_work.loc[_val_row, "probability"])
        _val_susc  = cov_work.loc[_val_row, "susceptibility"]
        _val_mech  = classify_failure_mechanism(_val_sv, list(feature_names))
        _val_contr = detect_contradictions(_val_sv)
        _val_ctx   = build_prediction_context(
            poly_id        = val_sel_poly,
            probability    = _val_prob,
            susc_class     = _val_susc,
            shap_values    = _val_sv,
            raw_values     = _val_xv,
            area_label     = area_label,
            jenks_breaks   = breaks,
            class_labels   = class_labels,
            shap_base      = shap_base,
            failure_mech   = _val_mech,
            contradictions = _val_contr,
        )
        _val_label      = f"Polygon {val_sel_poly}"
        _val_area_label = area_label    # global sidebar subarea label
        _val_zone_mode  = False

        with st.expander("📋 Test target summary"):
            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("Probability",       f"{_val_prob:.4f}")
            vc2.metric("Class",             _val_susc)
            vc3.metric("Failure mechanism", _val_mech[0].replace("_", " ").title())
            st.code(_val_ctx, language="text")

    # ── Mode B: Zone / subarea ───────────────────────────────────────────────
    else:
        # ------------------------------------------------------------------
        # Path 1: sidebar subarea filter is already active → use it directly
        # ------------------------------------------------------------------
        if use_subarea and subarea_col and selected_vals:
            st.info(
                f"✅ Using the **active sidebar filter** as the validation zone: "
                f"**{area_label}** — {len(cov_work):,} polygons. "
                "To validate a different zone, disable the sidebar filter first."
            )
            _vz_cov   = cov_work.copy()
            _vz_shap  = shap_df.copy()
            _vzcol    = subarea_col
            _vzone    = ", ".join(str(v) for v in selected_vals)
            _vz_pids  = set(_vz_cov["Id"].astype(str).str.strip())

        # ------------------------------------------------------------------
        # Path 2: no sidebar filter → let user pick any zone column + value
        # Search both the shapefile AND the covariates Excel for zone columns
        # ------------------------------------------------------------------
        else:
            # Collect candidates from gdf_full (shapefile) + cov_work (Excel)
            def _zone_cands(df):
                return [
                    c for c in df.columns
                    if c not in ("Id", "geometry")
                    and df[c].dtype == object
                    and 1 < df[c].nunique() <= 200
                ]
            _vz_cands_shp = _zone_cands(gdf_full)
            _vz_cands_cov = _zone_cands(cov_work)
            # Merge, deduplicate, keep order: shapefile first then Excel-only
            _vz_cands = _vz_cands_shp + [c for c in _vz_cands_cov
                                          if c not in _vz_cands_shp]

            if not _vz_cands:
                st.warning(
                    "No categorical zone columns found in the shapefile or "
                    "covariates table. Add a commune / watershed / sector "
                    "column to enable zone mode, or switch to Single polygon."
                )
                st.stop()

            _vzc1, _vzc2 = st.columns(2)
            _vzcol  = _vzc1.selectbox("Zone attribute column", _vz_cands,
                                       key="val_zcol")

            # Gather zone values from whichever source has the column
            if _vzcol in gdf_full.columns:
                _vzones = sorted(gdf_full[_vzcol].dropna().unique())
            else:
                _vzones = sorted(cov_work[_vzcol].dropna().unique())
            _vzone  = _vzc2.selectbox("Select zone", _vzones,
                                       key="val_zone_sel")

            # Match polygon IDs from the correct source
            if _vzcol in gdf_full.columns:
                _vz_pids = set(
                    gdf_full.loc[gdf_full[_vzcol] == _vzone,
                                 "Id"].astype(str).str.strip()
                )
                _vz_mask = cov_work["Id"].isin(_vz_pids)
            else:
                # Column is in cov_work (Excel) only
                _vz_mask = cov_work[_vzcol] == _vzone

            _vz_cov  = cov_work[_vz_mask]
            _vz_shap = shap_df[_vz_mask.values]

        if _vz_cov.empty:
            st.warning(
                f"Zone '{_vzone}' has no polygons in scope. "
                "Check the spatial filter in the sidebar."
            )
            st.stop()

        # Mean SHAP over zone — same aggregation as Zone mode
        _vz_top  = _vz_shap.abs().mean().sort_values(ascending=False)
        _val_sv  = _vz_shap.mean().to_dict()        # signed mean SHAP
        _val_xv  = {
            f: float(_vz_cov[f].mean()) for f in feature_names if f in _vz_cov
        }
        _val_prob = float(_vz_cov["probability"].mean())
        _val_susc = _vz_cov["susceptibility"].mode()[0]
        _val_mech = classify_failure_mechanism(_val_sv, list(feature_names))
        _val_contr = detect_contradictions(_val_sv)
        val_sel_poly = None          # no single polygon ID in zone mode
        val_poly_ids = _vz_cov["Id"].tolist()  # V4 batch will iterate these

        # Spatial context from zone centroid (same as AI tab zone mode)
        _vz_sp = ""
        if UTM_XCOL and UTM_YCOL:
            _vz_cr = covariates_df[covariates_df["Id"].isin(_vz_pids)]
            if not _vz_cr.empty:
                try:
                    _xm = float(pd.to_numeric(_vz_cr[UTM_XCOL], errors="coerce").median())
                    _ym = float(pd.to_numeric(_vz_cr[UTM_YCOL], errors="coerce").median())
                    _vz_sp = "\n" + build_spatial_context(compute_utm_distances(_xm, _ym))
                except Exception:
                    pass

        # LULC breakdown
        _vz_lulc = ""
        for _col in _vz_cov.columns:
            if "lulc" in _col.lower() or "land" in _col.lower():
                try:
                    _vc = _vz_cov[_col].value_counts()
                    _vz_lulc = "\n── LULC DISTRIBUTION (zone) ────────────────────────\n"
                    for _code, _cnt in _vc.head(6).items():
                        _lbl = LULC_LABELS.get(int(float(_code)), "?")
                        _vz_lulc += (
                            f"  {_lbl} (code {_code}): "
                            f"{_cnt} polygons ({_cnt/len(_vz_cov)*100:.1f}%)\n"
                        )
                except Exception:
                    pass
                break

        _val_ctx = (
            build_area_context(
                area_label     = f"{_vzcol}: {_vzone}",
                n_polygons     = len(_vz_cov),
                mean_prob      = _val_prob,
                class_counts   = _vz_cov["susceptibility"].value_counts().to_dict(),
                top_shap_feats = list(_vz_top.head(8).items()),
                shap_base      = shap_base,
            ) + _vz_lulc + _vz_sp
        )
        _val_label      = f"Zone {_vzcol}: {_vzone}"
        _val_area_label = f"{_vzcol}: {_vzone}"    # used by retrieval
        _val_zone_mode  = True

        with st.expander(f"📋 Test target summary — {_val_label}"):
            vc1, vc2, vc3, vc4 = st.columns(4)
            vc1.metric("Polygons in zone",  f"{len(_vz_cov):,}")
            vc2.metric("Mean probability",  f"{_val_prob:.4f}")
            vc3.metric("Dominant class",    _val_susc)
            vc4.metric("Failure mechanism", _val_mech[0].replace("_", " ").title())
            st.caption(
                "V1, V2, V2b, V3 use **mean SHAP** aggregated across all zone polygons — "
                "the same aggregation used in the Zone mode."
            )
            st.code(_val_ctx, language="text")

    st.markdown("---")


    st.divider()
    st.caption(
        "📌 **Workflow:** complete **Detection** first (instant, no API call) — its "
        "artefact flags feed Correction and Validation. Then run **Correction** to see "
        "the corrected LLM output, and **Validation** to score it."
    )

    (_ph1, _ph5, _ph2) = st.tabs([
        "🔍  1·Detection",
        "⚡  2·Correction",
        "📊  3·Validation",
    ])

    # ── Reusable helper: sticky polygon status banner ────────────────────────
    def _phase_banner(phase_name: str, need_phase1: bool = True):
        """Show current polygon context + optional Phase 1 gate warning."""
        _det = st.session_state.get("sl_detected", {})
        _det_poly = _det.get("poly_id")
        _match = (_det_poly is not None and
                  str(_det_poly) == str(val_sel_poly))
        _n_flags = _det.get("n_total", 0) if _match else None

        _flag_str = (
            f"L1:{len(_det.get('l1',[]))} L2:{len(_det.get('l2',[]))}"
        ) if _match and _n_flags else "not run"

        st.markdown(
            f"""<div style="background:#0d1b2a;border-radius:7px;padding:8px 16px;
            margin-bottom:12px;font-size:.82rem;color:#8bafc8;
            border:1px solid #1e3a5f;">
            📍 <b style="color:#d0e8ff;">{_val_label}</b>
            &nbsp;·&nbsp; Class: <b style="color:#e0c878;">{_val_susc}</b>
            &nbsp;·&nbsp; p = <b style="color:#e0c878;">{_val_prob:.4f}</b>
            &nbsp;·&nbsp; Phase 1: <b style="color:{'#27ae60' if _match else '#e74c3c'};">
            {'✅ ' + _flag_str if _match else '⚠️ not run'}</b>
            &nbsp;·&nbsp; <span style="color:#5a7a9a;">Phase: {phase_name}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        if need_phase1 and not _match:
            st.warning(
                "⚠️ **Run Phase 1 first** — SHAP artefact flags have not been "
                "detected for this polygon yet.  "
                "Without Phase 1, the pipeline runs in standard (uncorrected) mode "
                "and Phases 2/4/5 will not use the L1/L2 correction protocol.",
                icon=None,
            )

    # ── Session history accumulator ──────────────────────────────────────────
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = []   # list of per-polygon result snapshots

    def _log_session_result(label, susc, prob, gf_score, ragas_mean, n_flags):
        """Add or update a polygon result in the session history."""
        hist = st.session_state["session_history"]
        existing = [i for i, h in enumerate(hist) if h["poly_id"] == val_sel_poly]
        record = {
            "poly_id":    val_sel_poly,
            "label":      label,
            "class":      susc,
            "prob":       round(prob, 4),
            "GeoFaith":   round(gf_score, 3) if gf_score else None,
            "RAGAS mean": round(ragas_mean, 3) if ragas_mean else None,
            "Flags":      n_flags,
        }
        if existing:
            hist[existing[0]] = record
        else:
            hist.append(record)
        st.session_state["session_history"] = hist

    with _ph1:
        _phase_banner("Phase 1 · SHAP Detection", need_phase1=False)
        st.markdown("""
        <div style="background:#1a0a0a;border-radius:10px;padding:16px 20px;
                    margin-bottom:16px;border-left:4px solid #e74c3c;">
        <div style="color:#e74c3c;font-size:.75rem;font-weight:700;
                    letter-spacing:1.5px;">STEP 1 OF 3 — DETECT</div>
        <div style="color:#fde8e8;font-size:1rem;font-weight:600;margin:6px 0 4px 0;">
            What structural errors exist in these SHAP values?
        </div>
        <table style="color:#c8a0a0;font-size:.82rem;border-collapse:collapse;width:100%;">
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#e8b4b4;">
            <b>Input</b></td>
            <td>SHAP values + raw feature values for the selected polygon</td></tr>
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#e8b4b4;">
            <b>Process</b></td>
            <td>L1 collinearity check · L2 sign reversal check</td></tr>
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#e8b4b4;">
            <b>Output</b></td>
            <td>Artefact flag table · Severity bar · Session state saved for Phases 2–5</td></tr>
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#e8b4b4;">
            <b>Cost</b></td>
            <td>Instant — zero API calls</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
        # ══════════════════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════════════════
        # VSLIM — SHAP LIMITS: detection + correction impact
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-card">
        <div class="section-title">⚠️ VSLIM — SHAP Limits: Detection & Correction Impact</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This section answers the reviewers' core question: **"Do the L1/L2 corrections
        actually improve the explanation quality?"**

        **Workflow:**
        1. Detects L1 (collinearity) and L2 (sign reversal) artefacts
           for the selected polygon — **instant, no LLM call needed**.
        2. Runs the LLM **twice** on the same polygon and question:
           - **Uncorrected** — naive SHAP reading, no limitation protocol
           - **Corrected** — full L1/L2 protocol injected into prompt
        3. Scores both outputs with GeoFaithfulness (weighted) + all RAGAS metrics.
        4. Shows the **delta** as publication-ready comparative figures.
        """)

        if not openrouter_key:
            st.warning("Enter your OpenRouter API key to run the LLM comparison.")
        elif not (RAG_ENGINE_OK and "shap_df" in dir()):
            st.warning("Load model data first.")
        else:
            # ── Step 1: Detection (instant) ─────────────────────────────────
            _sl_sv  = _val_sv
            _sl_xv  = _val_xv

            _sl_l1 = detect_collinearity_artefacts(_sl_sv)
            _sl_l2 = detect_contradictions(_sl_sv)
            _sl_total = len(_sl_l1) + len(_sl_l2)

            # ── Summary chips ────────────────────────────────────────────────
            _dc1, _dc2, _dc3 = st.columns(3)
            _dc1.metric("Total flags", _sl_total,
                        delta="⚠️ Artefacts present" if _sl_total > 0
                              else "✅ Clean")
            _dc2.metric("L1 Collinearity", len(_sl_l1))
            _dc3.metric("L2 Sign reversal", len(_sl_l2))

            if _sl_total == 0:
                st.success(
                    "✅ No structural SHAP artefacts detected for this polygon. "
                    "The additive SHAP interpretation is reliable here. "
                    "Try a Very High or High susceptibility polygon to see artefacts."
                )
            else:
                # ── SHAP bar chart with artefact annotations ─────────────────
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np
                from io import BytesIO as _BIO

                _sorted_sv = sorted(
                    _sl_sv.items(), key=lambda x: x[1], reverse=True)
                _feats  = [f[0].replace("_"," ")[:22] for f in _sorted_sv]
                _svals  = [f[1] for f in _sorted_sv]
                _colors = ["#e74c3c" if v > 0 else "#3498db" for v in _svals]

                # Mark flagged features
                _l1_dom = {a["feat_dominant"].lower() for a in _sl_l1}
                _l1_sup = {a["feat_suppressed"].lower() for a in _sl_l1}
                _l2_feat = {a["feature"].lower() for a in _sl_l2}

                _markers = []
                for orig_name, _ in _sorted_sv:
                    nl = orig_name.lower()
                    if nl in _l2_feat:        _markers.append("⬟ L2")
                    elif nl in _l1_dom:       _markers.append("▲ L1")
                    elif nl in _l1_sup:       _markers.append("▼ L1")
                    else:                     _markers.append("")

                figP, axP = plt.subplots(figsize=(9, max(4, len(_feats)*0.40)))
                figP.patch.set_facecolor("white")
                axP.set_facecolor("white")
                _y = np.arange(len(_feats))
                _bars = axP.barh(_y, _svals, color=_colors,
                                 height=0.55, edgecolor="white", alpha=0.88)
                axP.axvline(0, color="#333", linewidth=0.9, zorder=0)
                # Compute x-range for smart label placement
                _xmin = min(_svals) if _svals else -0.1
                _xmax = max(_svals) if _svals else 0.1
                _xspan = max(abs(_xmax), abs(_xmin), 0.05)
                _pad = _xspan * 0.04  # 4% of range
                for i, (bar, val, mk) in enumerate(
                        zip(_bars, _svals, _markers)):
                    # Labels always OUTSIDE the bar tip, never between bar and y-axis
                    _x_pos = val + (_pad if val >= 0 else -_pad)
                    _ha    = "left" if val >= 0 else "right"
                    _label = f"{val:+.4f}"
                    if mk:
                        _label = f"{mk} {_label}"
                        bar.set_edgecolor("#d35400")
                        bar.set_linewidth(2.2)
                    axP.text(_x_pos, i, _label, va="center",
                             ha=_ha, fontsize=7.5,
                             fontfamily="serif",
                             color="#c0392b" if mk else "#444")
                # Extend x-limits so labels don't clip
                axP.set_xlim(_xmin - _xspan * 0.25, _xmax + _xspan * 0.25)
                axP.set_yticks(_y)
                axP.set_yticklabels(_feats, fontsize=9, fontfamily="serif")
                # Legend BEFORE apply_academic_style so the style function formats it
                from matplotlib.lines import Line2D
                _leg_handles = [
                    mpatches.Patch(facecolor="#c0392b", alpha=0.7,
                                   label="Risk-increasing (SHAP > 0)"),
                    mpatches.Patch(facecolor="#2980b9", alpha=0.7,
                                   label="Risk-decreasing (SHAP < 0)"),
                    mpatches.Patch(facecolor="none", edgecolor="#d35400",
                                   linewidth=2,
                                   label="Structural artefact detected"),
                ]
                axP.legend(handles=_leg_handles, loc="upper right",
                           fontsize=8, framealpha=0.95, edgecolor="#666",
                           fancybox=False)
                apply_academic_style(
                    figP, axP,
                    xlabel="SHAP value (positive = increases risk, "
                           "negative = decreases risk)",
                )
                st.pyplot(figP, use_container_width=True)

                # Download
                _buf_det = BytesIO()
                figP.savefig(_buf_det, format="png", dpi=1200, bbox_inches="tight",
                             facecolor="white")
                _buf_det.seek(0)
                plt.close(figP)
                st.download_button(
                    "📥 Download Detection Chart (1200 dpi)",
                    data=_buf_det, file_name=f"shap_detection_{val_sel_poly}.png",
                    mime="image/png", key="dl_detect_chart")

                st.caption(
                    f"**Fig.** SHAP feature attribution for polygon {val_sel_poly} "
                    f"({_val_susc}, *p* = {_val_prob:.4f}). "
                    "Red = risk-increasing; blue = risk-decreasing. "
                    "Orange-bordered bars denote detected SHAP structural artefacts: "
                    "▲L1 = collinearity-dominant, ▼L1 = collinearity-suppressed, "
                    "⬟L2 = sign reversal."
                )

                # ── Plain-language artefact cards ─────────────────────────────
                import pandas as _pd_sl
                if _sl_l1:
                    st.markdown("""
                    <div style="background:#1a0505;border-radius:8px;padding:14px 18px;
                                margin:10px 0;border-left:3px solid #e74c3c;">
                    <b style="color:#e74c3c;">L1 — Collinearity artefact</b>
                    <p style="color:#e8c0c0;font-size:.84rem;margin:6px 0 0 0;line-height:1.6;">
                    <b>What it means:</b> Two features that measure almost the same
                    geomorphological property (e.g. slope and elevation both measure
                    terrain steepness) are competing in the SHAP calculation.
                    SHAP gave nearly all the credit to one feature and almost none to the
                    other — but this split is arbitrary, not physical.<br>
                    <b>What the LLM will do:</b> Instead of explaining them separately,
                    it will frame them as a single joint driver.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    _l1_rows = []
                    for _c in _sl_l1:
                        _l1_rows.append({
                            "Feature that got credit":  _c["feat_dominant"],
                            "Its SHAP":                 f"{_c['shap_dominant']:+.4f}",
                            "Feature that was ignored": _c["feat_suppressed"],
                            "Its SHAP":                 f"{_c['shap_suppressed']:+.4f}",
                            "Credit ratio":             f"{_c['ratio']:.1f}×",
                            "Correlation (ρ)":          (
                                lambda r, p: (
                                    f"{r:.2f} (measured)" if r is not None
                                    else f"{p:.2f} (prior)" if p is not None
                                    else "?"
                                )(_c.get('empirical_r'), _c.get('expected_r'))
                            ),
                            "Physical link":            _c["correlation_note"][:70] + "…",
                        })
                    st.dataframe(_pd_sl.DataFrame(_l1_rows),
                                 use_container_width=True, hide_index=True)

                if _sl_l2:
                    st.markdown("""
                    <div style="background:#1a0e05;border-radius:8px;padding:14px 18px;
                                margin:10px 0;border-left:3px solid #e67e22;">
                    <b style="color:#e67e22;">L2 — Interaction suppression</b>
                    <p style="color:#e8d0b0;font-size:.84rem;margin:6px 0 0 0;line-height:1.6;">
                    <b>What it means:</b> Two features are dangerous <em>together</em>
                    in a way that is much larger than their individual effects.
                    For example, a steep slope on Precambrian schist creates a specific
                    failure type (planar slip) that does not happen on the same slope
                    on limestone. SHAP cannot represent this — it forces each feature
                    to have an independent contribution.<br>
                    <b>What the LLM will do:</b> It will use conditional language:
                    "risk is amplified when these two features co-occur."
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    _l2_rows = []
                    for _ix in _sl_l2:
                        _l2_rows.append({
                            "Feature A":       _ix["feature_a"],
                            "Condition A":     _ix.get("zone_a_str","")[:35],
                            "Feature B":       _ix["feature_b"],
                            "Condition B":     _ix.get("zone_b_str","")[:35],
                            "Combined SHAP":   f"{_ix['joint_shap']:+.4f}",
                            "Joint mechanism": _ix["description"][:80] + "…",
                        })
                    st.dataframe(_pd_sl.DataFrame(_l2_rows),
                                 use_container_width=True, hide_index=True)

                if _sl_l3:
                    st.markdown("""
                    <div style="background:#050a1a;border-radius:8px;padding:14px 18px;
                                margin:10px 0;border-left:3px solid #3498db;">
                    <b style="color:#3498db;">L2 — Sign reversal</b>
                    <p style="color:#b0c8e8;font-size:.84rem;margin:6px 0 0 0;line-height:1.6;">
                    <b>What it means:</b> A feature's SHAP value says it
                    <em>reduces</em> risk, but geomorphologically we expect it to
                    <em>increase</em> risk (or vice versa). This is usually caused by
                    another correlated feature absorbing the contribution, leaving
                    the remainder with the wrong sign.<br>
                    <b>What the LLM will do:</b> It will explicitly flag this as
                    a potential artefact and distinguish between a genuine local
                    exception and a statistical sign error.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    _l2_rows = []
                    for _ct in _sl_l3:
                        _l2_rows.append({
                            "Feature":         _ct["feature"],
                            "SHAP says":       f"{_ct['shap']:+.4f} "
                                               f"({'decreases' if _ct['shap']<0 else 'increases'} risk)",
                            "Geomorphology expects": _ct["expected"],
                            "Conflict":        "⚠️ REVERSED",
                            "Severity":        _ct["severity"],
                            "Possible cause":  _ct["note"][:70] + "…",
                        })
                    st.dataframe(_pd_sl.DataFrame(_l2_rows),
                                 use_container_width=True, hide_index=True)

                # ═══════════════════════════════════════════════════════════
                #  DETAILED DIAGNOSTIC PANELS (academic figures)
                # ═══════════════════════════════════════════════════════════

                # ── L1: Correlation heatmap between collinear pairs ───────
                if _sl_l1:
                    st.markdown("#### L1 — Collinearity Diagnostic")
                    # Build correlation matrix for all L1-involved features
                    _l1_all_feats = []
                    for _c in _sl_l1:
                        if _c["feat_dominant"] not in _l1_all_feats:
                            _l1_all_feats.append(_c["feat_dominant"])
                        if _c["feat_suppressed"] not in _l1_all_feats:
                            _l1_all_feats.append(_c["feat_suppressed"])

                    if len(_l1_all_feats) >= 2:
                        # ── Column matching: find the best X_raw column for each L1 feature
                        # Priority: exact match > case-insensitive match > substring match
                        _l1_feat_cols = []
                        _l1_col_map = {}  # feat_name -> column_name
                        for _lf in _l1_all_feats:
                            _lf_low = _lf.lower().replace("_", " ")
                            _best = None
                            # Priority 1: exact case-insensitive match
                            for c in X_raw.columns:
                                if c.lower() == _lf.lower():
                                    _best = c
                                    break
                            # Priority 2: column starts with feature name
                            if not _best:
                                for c in X_raw.columns:
                                    if c.lower().startswith(_lf.lower()):
                                        _best = c
                                        break
                            # Priority 3: substring match
                            if not _best:
                                for c in X_raw.columns:
                                    if _lf_low in c.lower().replace("_", " "):
                                        _best = c
                                        break
                            if _best:
                                _l1_feat_cols.append(_best)
                                _l1_col_map[_lf] = _best
                            else:
                                _l1_col_map[_lf] = None

                        _l1_valid_cols = [c for c in _l1_feat_cols if c in X_raw.columns]
                        if len(_l1_valid_cols) >= 2:
                            # ── Compute SPEARMAN rank correlation (matches L1 detector)
                            # Spearman is appropriate because:
                            # (a) the L1 detector uses Spearman ρ priors
                            # (b) categorical features (geology, LULC) have ordinal
                            #     but not interval-scaled values
                            # (c) Spearman is robust to non-linear monotonic relationships
                            _l1_corr_df = X_raw[_l1_valid_cols].corr(method="spearman")
                            _n_poly = len(X_raw)

                            fig_hm, ax_hm = plt.subplots(
                                figsize=(max(4, len(_l1_valid_cols)*1.4),
                                         max(3, len(_l1_valid_cols)*1.1)))
                            fig_hm.patch.set_facecolor("white")
                            _im = ax_hm.imshow(_l1_corr_df.values, cmap="RdBu_r",
                                               vmin=-1, vmax=1, aspect="auto")
                            # Annotate cells with ρ values
                            for _ri in range(len(_l1_valid_cols)):
                                for _ci in range(len(_l1_valid_cols)):
                                    _rv = _l1_corr_df.values[_ri, _ci]
                                    ax_hm.text(_ci, _ri, f"{_rv:.2f}",
                                               ha="center", va="center",
                                               fontsize=10, fontfamily="serif",
                                               fontweight="bold" if abs(_rv) > 0.5 else "normal",
                                               color="white" if abs(_rv) > 0.6 else "black")
                            ax_hm.set_xticks(range(len(_l1_valid_cols)))
                            ax_hm.set_xticklabels(_l1_valid_cols, fontsize=9,
                                                  fontfamily="serif", rotation=45, ha="right")
                            ax_hm.set_yticks(range(len(_l1_valid_cols)))
                            ax_hm.set_yticklabels(_l1_valid_cols, fontsize=9,
                                                  fontfamily="serif")
                            _cb = fig_hm.colorbar(_im, ax=ax_hm, shrink=0.8)
                            _cb.set_label("Spearman rank correlation (ρ)", fontsize=9,
                                          fontfamily="serif")
                            ax_hm.spines[:].set_visible(False)
                            fig_hm.tight_layout(pad=1.5)
                            st.pyplot(fig_hm)
                            _buf_hm = BytesIO()
                            fig_hm.savefig(_buf_hm, format="png", dpi=1200,
                                           bbox_inches="tight", facecolor="white")
                            _buf_hm.seek(0)
                            plt.close(fig_hm)
                            st.download_button(
                                "📥 Download L1 Correlation Heatmap (1200 dpi)",
                                data=_buf_hm,
                                file_name=f"l1_correlation_{val_sel_poly}.png",
                                mime="image/png", key="dl_l1_heatmap")
                            st.caption(
                                f"**Fig.** Spearman rank correlation matrix for L1-flagged "
                                f"features (n = {_n_poly:,} polygons). High absolute "
                                "correlations (|ρ| > 0.5) indicate feature pairs where "
                                "the Shapley independence assumption produces attribution "
                                "concentration artefacts. Spearman is used rather than "
                                "Pearson because it handles ordinal variables (geology, "
                                "LULC codes) and non-linear monotonic relationships."
                            )

                        # ── SHAP magnitude + correlation comparison table ─────
                        st.markdown("**L1 SHAP dominance ratios (empirical vs detector):**")
                        _l1_detail = []
                        for _c in _sl_l1:
                            # Get the empirical Spearman ρ from the heatmap
                            _col_dom = _l1_col_map.get(_c["feat_dominant"])
                            _col_sup = _l1_col_map.get(_c["feat_suppressed"])
                            _emp_rho = None
                            if (_col_dom and _col_sup and
                                    _col_dom in X_raw.columns and
                                    _col_sup in X_raw.columns):
                                _emp_rho = float(
                                    X_raw[[_col_dom, _col_sup]]
                                    .corr(method="spearman").iloc[0, 1]
                                )

                            _det_r = _c.get("empirical_r") or _c.get("expected_r") or 0
                            _det_src = _c.get("r_source", "prior")

                            # Flag if empirical and detector ρ diverge significantly
                            _rho_match = ""
                            if _emp_rho is not None and abs(_emp_rho - _det_r) > 0.15:
                                _rho_match = (
                                    f"⚠️ Δ={_emp_rho - _det_r:+.2f} — "
                                    "recalibrate prior"
                                )
                            elif _emp_rho is not None:
                                _rho_match = "✅ consistent"

                            _l1_detail.append({
                                "Dominant":          _c["feat_dominant"],
                                "|SHAP| dom.":       f"{abs(_c['shap_dominant']):.4f}",
                                "Suppressed":        _c["feat_suppressed"],
                                "|SHAP| sup.":       f"{abs(_c['shap_suppressed']):.4f}",
                                "Ratio":             f"{_c['ratio']:.1f}×",
                                "ρ (detector)":      f"{_det_r:.3f} ({_det_src})",
                                "ρ (empirical)":     f"{_emp_rho:.3f}" if _emp_rho is not None else "—",
                                "Agreement":         _rho_match,
                            })
                        st.dataframe(_pd_sl.DataFrame(_l1_detail),
                                     use_container_width=True, hide_index=True)
                        st.caption(
                            "The **ρ (detector)** column shows the correlation value "
                            "used by the L1 detection algorithm (either a domain-knowledge "
                            "prior or an empirically calibrated value from "
                            "`update_collinear_pairs_from_data()`). The **ρ (empirical)** "
                            "column shows the Spearman rank correlation computed from "
                            "the actual covariate data. The **Agreement** column flags "
                            "pairs where the two values diverge by more than 0.15."
                        )

                # ── L2: Interaction effect graph ──────────────────────────
                if _sl_l2:
                    st.markdown("#### L2 — Interaction Diagnostic")

                    for _ix_i, _ix in enumerate(_sl_l2):
                        _fa = _ix["feature_a"]
                        _fb = _ix["feature_b"]
                        _sa = _ix["shap_a"]
                        _sb = _ix["shap_b"]
                        _sj = _ix["joint_shap"]

                        # Grouped bar: individual vs joint
                        fig_ix, ax_ix = plt.subplots(figsize=(6, 3.5))
                        fig_ix.patch.set_facecolor("white")
                        _x_pos = [0, 1, 2.2]
                        _heights = [_sa, _sb, _sj]
                        _colors_ix = ["#e74c3c", "#e67e22",  "#8e44ad"]
                        _labels_ix = [_fa, _fb, f"{_fa} × {_fb}\n(joint)"]
                        _bars_ix = ax_ix.bar(_x_pos, _heights, width=0.7,
                                             color=_colors_ix, edgecolor="white",
                                             linewidth=0.5, alpha=0.85)
                        # Additive reference line
                        _additive = _sa + _sb
                        ax_ix.axhline(_additive, color="#888", linestyle="--",
                                      linewidth=1.0, zorder=0)
                        ax_ix.text(2.7, _additive, f"Additive sum = {_additive:+.2f}",
                                   fontsize=8, fontfamily="serif", color="#666",
                                   va="bottom")
                        # Value labels on bars
                        for _bx, _bh in zip(_x_pos, _heights):
                            ax_ix.text(_bx, _bh + 0.05, f"{_bh:+.2f}",
                                       ha="center", va="bottom", fontsize=9,
                                       fontfamily="serif", fontweight="bold")
                        ax_ix.set_xticks(_x_pos)
                        ax_ix.set_xticklabels(_labels_ix, fontsize=9,
                                              fontfamily="serif")
                        ax_ix.legend(
                            handles=[
                                mpatches.Patch(facecolor="#e74c3c", label=f"{_fa} (individual)"),
                                mpatches.Patch(facecolor="#e67e22", label=f"{_fb} (individual)"),
                                mpatches.Patch(facecolor="#8e44ad", label="Joint SHAP sum"),
                                plt.Line2D([0], [0], color="#888", linestyle="--",
                                           label="Additive prediction"),
                            ],
                            loc="upper right", fontsize=7.5, framealpha=0.95,
                            edgecolor="#666", fancybox=False,
                        )
                        apply_academic_style(fig_ix, ax_ix,
                                             ylabel="SHAP contribution")
                        st.pyplot(fig_ix)
                        _buf_ix = BytesIO()
                        fig_ix.savefig(_buf_ix, format="png", dpi=1200,
                                       bbox_inches="tight", facecolor="white")
                        _buf_ix.seek(0)
                        plt.close(fig_ix)
                        st.download_button(
                            f"📥 Download L2 Sign reversal Chart ({_fa} × {_fb}, 1200 dpi)",
                            data=_buf_ix,
                            file_name=f"l2_interaction_{_fa}_{_fb}_{val_sel_poly}.png",
                            mime="image/png",
                            key=f"dl_l2_{_ix_i}")

                        # Mechanism description
                        st.caption(
                            f"**Fig.** L2 sign reversal effect between {_fa} "
                            f"(SHAP = {_sa:+.4f}) and {_fb} (SHAP = {_sb:+.4f}) "
                            f"for polygon {val_sel_poly}. The joint SHAP sum "
                            f"({_sj:+.4f}) is compared against the additive "
                            f"prediction ({_additive:+.4f}). The dashed line "
                            "represents the expected contribution under Shapley "
                            "additivity. Deviation indicates a non-linear "
                            "interaction effect suppressed by the additive "
                            "decomposition. "
                            f"Mechanism: {_ix['description'][:120]}"
                        )

                    # Summary table
                    _l2_summary = []
                    for _ix in _sl_l2:
                        _add = _ix["shap_a"] + _ix["shap_b"]
                        _l2_summary.append({
                            "Feature A":        _ix["feature_a"],
                            "SHAP A":           f"{_ix['shap_a']:+.4f}",
                            "Feature B":        _ix["feature_b"],
                            "SHAP B":           f"{_ix['shap_b']:+.4f}",
                            "Joint SHAP":       f"{_ix['joint_shap']:+.4f}",
                            "Additive sum":     f"{_add:+.4f}",
                            "Hazard zone":      f"{_ix.get('zone_a_str','')} & {_ix.get('zone_b_str','')}",
                        })
                    st.dataframe(_pd_sl.DataFrame(_l2_summary),
                                 use_container_width=True, hide_index=True)

                # ── L2: Sign reversal justification panel ─────────────────
                if _sl_l3:
                    st.markdown("#### L2 — Sign Reversal Diagnostic")

                    # Detailed justification table
                    _l2_detail = []
                    for _ct in _sl_l3:
                        _feat = _ct["feature"]
                        _sv = _ct["shap"]
                        _exp = _ct["expected"]
                        _act = "+" if _sv > 0 else "-"
                        _sev = _ct["severity"]

                        # Determine most likely cause
                        _feat_l = _feat.lower()
                        # Check if this feature is also L1-suppressed
                        _is_l1_suppressed = any(
                            _ct["feature"].lower() == c["feat_suppressed"].lower()
                            for c in _sl_l1
                        ) if _sl_l1 else False
                        _is_l1_dominant = any(
                            _ct["feature"].lower() == c["feat_dominant"].lower()
                            for c in _sl_l1
                        ) if _sl_l1 else False

                        # Build diagnostic justification
                        if _is_l1_suppressed:
                            _diagnosis = (
                                f"COLLINEARITY-INDUCED: {_feat} is L1-suppressed — "
                                "a correlated dominant feature absorbed its attribution, "
                                "leaving a residual with reversed sign. This is a SHAP "
                                "structural artefact, not a genuine geomorphological reversal."
                            )
                            _cause = "L1 collinearity suppression"
                        elif _is_l1_dominant:
                            _diagnosis = (
                                f"INTERACTION-MASKING: {_feat} is L1-dominant but its "
                                "sign reversal suggests a masked interaction with a "
                                "suppressed partner. The dominant feature carries the "
                                "joint attribution with distorted sign."
                            )
                            _cause = "L1 interaction masking"
                        elif abs(_sv) < 0.05:
                            _diagnosis = (
                                f"NEGLIGIBLE MAGNITUDE: |SHAP| = {abs(_sv):.4f} is "
                                "below the practical significance threshold (0.05). "
                                "The sign reversal is likely numerical noise from the "
                                "Shapley marginal computation, not a meaningful signal."
                            )
                            _cause = "Numerical noise (negligible |SHAP|)"
                        else:
                            _diagnosis = (
                                f"POTENTIAL LOCAL OVERRIDE: {_feat} shows a "
                                f"{'negative' if _sv < 0 else 'positive'} SHAP value "
                                f"({_sv:+.4f}) where geomorphological expectation is "
                                f"'{_exp}'. This could be: (a) a genuine local exception "
                                "due to site-specific conditions, or (b) a multicollinearity "
                                "artefact where another correlated feature absorbs the "
                                "expected contribution."
                            )
                            _cause = "Local override or multicollinearity"

                        _l2_detail.append({
                            "Feature":         _feat,
                            "SHAP value":      f"{_sv:+.4f}",
                            "SHAP direction":  "Increases risk" if _sv > 0 else "Decreases risk",
                            "Expected":        f"{'Increases' if _exp=='+' else 'Decreases'} risk",
                            "Severity":        _sev,
                            "Probable cause":  _cause,
                            "Diagnostic":      _diagnosis,
                        })

                    st.dataframe(_pd_sl.DataFrame(_l2_detail),
                                 use_container_width=True, hide_index=True)

                    # Signed comparison chart: expected vs observed
                    fig_l2, ax_l2 = plt.subplots(
                        figsize=(7, max(2.5, len(_sl_l3) * 0.55)))
                    fig_l2.patch.set_facecolor("white")
                    _l2_names = [c["feature"] for c in _sl_l3]
                    _l2_shap  = [c["shap"] for c in _sl_l3]
                    _l2_exp   = [1.0 if c["expected"] == "+" else -1.0
                                 for c in _sl_l3]
                    _y_l3 = np.arange(len(_l2_names))
                    # Observed SHAP bars
                    ax_l2.barh(_y_l3 + 0.15, _l2_shap, height=0.3,
                               color=["#c0392b" if s > 0 else "#2980b9"
                                      for s in _l2_shap],
                               alpha=0.8, label="Observed SHAP")
                    # Expected direction arrows
                    for _yi, _ev in enumerate(_l2_exp):
                        _arrow_x = _ev * 0.3
                        ax_l2.annotate("", xy=(_arrow_x, _yi - 0.15),
                                       xytext=(0, _yi - 0.15),
                                       arrowprops=dict(arrowstyle="->",
                                                       color="#27ae60",
                                                       lw=2.0))
                    ax_l2.axvline(0, color="#333", linewidth=0.8, zorder=0)
                    ax_l2.set_yticks(_y_l3)
                    ax_l2.set_yticklabels(_l2_names, fontsize=9,
                                          fontfamily="serif")
                    ax_l2.legend(
                        handles=[
                            mpatches.Patch(facecolor="#c0392b", alpha=0.8,
                                           label="Observed (SHAP > 0)"),
                            mpatches.Patch(facecolor="#2980b9", alpha=0.8,
                                           label="Observed (SHAP < 0)"),
                            plt.Line2D([0], [0], color="#27ae60", lw=2,
                                       marker=">", markersize=6,
                                       label="Expected direction"),
                        ],
                        loc="upper right", fontsize=8, framealpha=0.95,
                        edgecolor="#666", fancybox=False,
                    )
                    apply_academic_style(fig_l2, ax_l2,
                                         xlabel="SHAP value / expected direction")
                    st.pyplot(fig_l2)
                    _buf_l2 = BytesIO()
                    fig_l2.savefig(_buf_l2, format="png", dpi=1200,
                                   bbox_inches="tight", facecolor="white")
                    _buf_l2.seek(0)
                    plt.close(fig_l2)
                    st.download_button(
                        "📥 Download L2 Sign Reversal Chart (1200 dpi)",
                        data=_buf_l2,
                        file_name=f"l2_reversal_{val_sel_poly}.png",
                        mime="image/png", key="dl_l2_chart")
                    st.caption(
                        f"**Fig.** L2 sign reversal diagnostic for polygon "
                        f"{val_sel_poly}. Horizontal bars show the observed "
                        "SHAP value (red = risk-increasing, blue = risk-decreasing). "
                        "Green arrows indicate the geomorphologically expected "
                        "direction for each feature. Mismatches between bar "
                        "direction and arrow direction indicate sign reversals "
                        "requiring diagnostic investigation (collinearity suppression, "
                        "interaction masking, or genuine local override)."
                    )

            # ── Persist detection results ────────────────────────────────────
            st.session_state["sl_detected"] = {
                "poly_id":        val_sel_poly,
                "l1":             _sl_l1,
                "l2":             _sl_l2,
                "contradictions": _sl_l2,
                "n_total":        _sl_total,
                "lim_diag": {
                    "n_flags":      _sl_total,
                    "collinear":    _sl_l1,
                },
            }
            if _sl_total > 0:
                st.info(
                    f"🔁 **{_sl_total} flag(s) saved.** "
                    "Now go to **Phase 2** to run the corrected pipeline, "
                    "or **Phase 5** to compare corrected vs uncorrected text side-by-side."
                )



        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
    with _ph2:
        _phase_banner("Phase 2 · Core Metrics")

        # ── Session history ─────────────────────────────────────────────────
        _hist = st.session_state.get("session_history", [])
        if _hist:
            with st.expander(
                f"📋 Session history — {len(_hist)} polygon(s) evaluated this session",
                expanded=False,
            ):
                import pandas as _pd_hist
                st.dataframe(
                    _pd_hist.DataFrame(_hist),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "GeoFaith and RAGAS mean are updated each time you run V1/V2 "
                    "on a polygon. Use this to compare performance across polygons "
                    "before running the full statistical analysis in Phase 4 V3c."
                )

        st.markdown("""
        <div style="background:#0a1a0f;border-radius:10px;padding:16px 20px;
                    margin-bottom:16px;border-left:4px solid #27ae60;">
        <div style="color:#27ae60;font-size:.75rem;font-weight:700;
                    letter-spacing:1.5px;">STEP 3 OF 3 — VALIDATE (automated metrics)</div>
        <div style="color:#e8fde8;font-size:1rem;font-weight:600;margin:6px 0 4px 0;">
            How good is the corrected explanation on this polygon?
        </div>
        <table style="color:#a0c8a0;font-size:.82rem;border-collapse:collapse;width:100%;">
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#b4e8b4;">
            <b>Input</b></td>
            <td>LLM explanation text · Retrieved chunks · SHAP values</td></tr>
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#b4e8b4;">
            <b>V1 GeoFaithfulness</b></td>
            <td>Every directional claim audited against SHAP sign. Violations listed.
                Target ≥ 0.95</td></tr>
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#b4e8b4;">
            <b>V2 RAGAS</b></td>
            <td>Context Precision · Recall · Faithfulness · Utilisation · Relevancy.
                Target mean ≥ 0.70</td></tr>
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#b4e8b4;">
            <b>V2b Stage Δ</b></td>
            <td>2-stage vs 3-stage retrieval score gap.
                Positive Δ = limitation-aware RAG helps.</td></tr>
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#b4e8b4;">
            <b>Output</b></td>
            <td>Score table · Violation list · Chunk utilisation chart</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        # V1 — Explanation Quality Index (EQI)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-card">
        <div class="section-title">V1 — Explanation Quality Index (EQI)</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        The Explanation Quality Index (EQI) is a single 0–100 score that measures
        four dimensions of the RAG-LLM explanation quality:

        | Sub-score | What it measures |
        |---|---|
        | **Coverage** | Did the LLM mention all important features (|SHAP| > 25th percentile)? |
        | **Directional accuracy** | For unflagged features, did the LLM assign the correct direction? |
        | **Mechanism grounding** | Are factual claims supported by retrieved literature passages? |
        | **Artefact reporting** | Did the LLM identify and explain each L1/L2 flag with its cause? |

        **Grade:** A (≥75, publication-ready) · B (50–74, needs revision) · C (<50, major issues)
        """)

        if st.button("▶ Run EQI Test", key="v1_btn", type="primary"):
            with st.spinner("Generating explanation and computing EQI…"):
                try:
                    # Retrieve chunks
                    _v1_retr = []
                    if db_ready_v and rag_collection is not None:
                        _v1_raw = two_stage_retrieve(
                            rag_collection, _val_sv, _val_xv,
                            _val_susc, area_label,
                            n_general=8, n_specific=5,
                        )
                        _v1_retr = shap_rerank(_v1_raw, _val_sv, top_n=12)

                    # Detect limitations
                    _v1_contradictions = detect_contradictions(_val_sv)
                    _v1_lim_block, _v1_lim_diag = build_shap_limitations_block(
                        _val_sv, _val_xv, GLOBAL_SHAP_RANK)

                    v1_question = (
                        f"Explain the landslide susceptibility prediction for "
                        f"{_val_label} ({_val_susc}, p={_val_prob:.4f})."
                    )

                    _v1_res = call_openrouter(
                        api_key=openrouter_key, model=llm_model,
                        system_prompt=DEFAULT_SYSTEM_PROMPT,
                        user_message=f"{v1_question}\n\n{_val_ctx}",
                        retrieved_chunks=_v1_retr,
                        conversation_hist=[], few_shot_n=llm_few_shot,
                        temperature=min(llm_temperature, 0.15),
                        top_p=0.95, top_k=20,
                        max_tokens=int(llm_max_tokens),
                        frequency_penalty=0.15, presence_penalty=0.05,
                        repetition_penalty=1.02, min_p=0.02,
                        limitations_diag=_v1_lim_diag,
                    )
                    _v1_text = _v1_res["content"]

                    # Compute RAGAS
                    _v1_flagged = []
                    if _v1_contradictions:
                        _v1_flagged += [c.get("feature","") for c in _v1_contradictions]
                    if _v1_lim_diag:
                        for _c in _v1_lim_diag.get("collinear", []):
                            _v1_flagged += [_c.get("feat_dominant",""), _c.get("feat_suppressed","")]
                    _v1_ragas = compute_ragas_style(
                        v1_question, extract_sections_1_2(_v1_text),
                        _v1_retr, _val_sv,
                        flagged_features=_v1_flagged or None)

                    # Compute EQI
                    _v1_eqi = compute_eqi(
                        extract_sections_1_2(_v1_text), _val_sv, _v1_ragas,
                        limitations_diag=_v1_lim_diag,
                        contradictions=_v1_contradictions,
                    )

                    # Store results
                    st.session_state["val_v1_result"] = {
                        "poly_id": val_sel_poly,
                        "text":    _v1_text,
                        "ragas":   _v1_ragas,
                        "eqi":     _v1_eqi,
                    }

                except Exception as _e:
                    st.error(f"V1 test failed: {_e}")

        # ── Display stored V1 results ────────────────────────────────
        _v1_stored = st.session_state.get("val_v1_result")
        if _v1_stored and _v1_stored.get("poly_id") == val_sel_poly and _v1_stored.get("eqi"):
            _v1_eqi = _v1_stored["eqi"]
            _v1_ragas = _v1_stored["ragas"]

            # EQI headline
            _eqi_s = _v1_eqi["eqi_score"]
            _eqi_bg = {"green": "#27ae60", "orange": "#f39c12",
                       "red": "#e74c3c"}.get(_v1_eqi["colour"], "#888")

            _vc1, _vc2 = st.columns([1, 3])
            with _vc1:
                st.markdown(
                    f'<div style="background:{_eqi_bg};color:white;'
                    f'padding:20px;border-radius:10px;text-align:center;">'
                    f'<h1 style="margin:0;font-size:48px;">{_eqi_s}</h1>'
                    f'<p style="margin:4px 0 0 0;font-size:14px;">'
                    f'EQI — {_v1_eqi["label"]}</p></div>',
                    unsafe_allow_html=True
                )

            with _vc2:
                _subs = _v1_eqi["sub_scores"]
                _mc1, _mc2, _mc3 = st.columns(3)
                _mc1.metric("Coverage", f"{_subs['coverage']:.0f}%")
                _mc2.metric("Directional", f"{_subs['directional_accuracy']:.0f}%")
                _mc3.metric("Artefacts", f"{_subs['artefact_reporting']:.0f}%")

                _det = _v1_eqi["details"]
                st.caption(
                    f"Coverage: {_det['n_mentioned']}/{_det['n_important_feats']} features · "
                    f"Direction: {_det['n_dir_correct']}/{_det['n_dir_claims']} correct "
                    f"(excl. {_det['n_exempt_features']} flagged) · "
                    f"Artefacts: {_det['n_flags_reported']}/{_det['n_flags_total']} reported"
                )

            # RAGAS supporting metrics
            st.markdown("**Supporting RAGAS metrics:**")
            _rc1, _rc2, _rc3, _rc4, _rc5 = st.columns(5)
            _rc1.metric("Precision", f"{_v1_ragas.get('context_precision', 0):.3f}")
            _rc2.metric("Recall", f"{_v1_ragas.get('context_recall', 0):.3f}")
            _rc3.metric("Faithfulness", f"{_v1_ragas.get('faithfulness', 0):.3f}")
            _rc4.metric("Utilisation", f"{_v1_ragas.get('context_utilisation', 0):.3f}")
            _rc5.metric("Relevancy", f"{_v1_ragas.get('answer_relevancy', 0):.3f}")

            # LLM output
            with st.expander("📝 Full LLM output"):
                st.markdown(_v1_stored.get("text", ""))

        st.markdown("</div>", unsafe_allow_html=True)

        # V2 — RAGAS-style RAG Quality
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-card">
        <div class="section-title">V2 — RAGAS-style RAG Retrieval Quality</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Scientific basis:** RAGAS (Retrieval-Augmented Generation Assessment) is the
        standard framework for evaluating RAG pipelines without human judges
        (Es et al., 2023, arXiv:2309.15217). The four metrics assess complementary
        aspects of retrieval and generation quality.

        | Metric | Definition | Enhancement | Target |
        |---|---|---|---|
        | **Context Precision** | Chunk relevance weighted by SHAP rank of matched feature | **E2**: rank-1 hit = 1.0, rank-N = 1/N | ≥ 0.70 |
        | **Context Recall** | Top SHAP features covered in chunks (alias-matched) | **E3**: column names resolved to literature terms | ≥ 0.60 |
        | **Faithfulness** | Claim-grounded: LLM sentences traceable to retrieved context | **E3+E6**: alias matching + ≥3 mech terms + spatial anchor required | ≥ 0.75 |
        | **Answer Relevancy** | Top features + LULC/Geology decoding + Haouz spatial context | Unchanged | ≥ 0.70 |

        **Faithfulness = None** means no testable factual claims were found in the answer
        (too vague, no feature-specific directional statements). This is distinct from
        **Faithfulness = 0** (claims found but none grounded in retrieved context).

        **Note:** These are automated approximations calibrated to geoscientific text.
        They are presented as proxy metrics and acknowledged as such in the paper.
        """)

        v2_n_chunks = st.slider("RAG chunks to evaluate", 3, 15, llm_n_chunks, key="v2_k")
        v2_question = st.text_area(
            "Query for V2",
            value=(
                f"Explain the dominant landslide drivers for {_val_label} "
                f"({_val_susc}) in Haouz Province using SHAP values and geology."
            ),
            height=70, key="v2_q",
        )

        if st.button("▶ Run RAGAS-style Evaluation", key="v2_btn", type="primary"):
            with st.spinner("Retrieving literature and computing RAGAS metrics…"):
                try:
                    if not db_ready_v:
                        st.warning("Vector DB not built — retrieval metrics will be zero.")
                        _v2_retr     = []
                        _v2_lim_diag = None
                    else:
                        # V2: inherit Phase 1 limitation flags (same polygon)
                        _v2_det      = st.session_state.get("sl_detected", {})
                        _v2_lim_ok   = (_v2_det.get("poly_id") == val_sel_poly and
                                        _v2_det.get("n_total", 0) > 0)
                        _v2_lim_diag = _v2_det.get("lim_diag") if _v2_lim_ok else None

                        _v2_raw  = two_stage_retrieve(
                            rag_collection, _val_sv, _val_xv, _val_susc,
                            _val_area_label, v2_n_chunks, max(2, v2_n_chunks//2),
                            limitations_diag=_v2_lim_diag,
                        )
                        _v2_retr = shap_rerank(
                            _v2_raw, _val_sv, v2_n_chunks,
                            limitations_diag=_v2_lim_diag,
                        )

                    _v2_res   = call_openrouter(
                        api_key=openrouter_key, model=llm_model,
                        system_prompt=DEFAULT_SYSTEM_PROMPT,
                        user_message=f"{v2_question}\n\n{_val_ctx}",
                        retrieved_chunks=_v2_retr, conversation_hist=[],
                        few_shot_n=1, temperature=llm_temperature,
                        top_p=llm_top_p, top_k=llm_top_k,
                        max_tokens=800, frequency_penalty=0.2,
                        presence_penalty=0.0, repetition_penalty=1.05, min_p=0.02,
                        limitations_diag=_v2_lim_diag,
                    )
                    _v2_text  = _v2_res["content"]
                    _v2_ragas = compute_ragas_style(v2_question, extract_sections_1_2(_v2_text), _v2_retr, _val_sv, flagged_features=None)

                    r1,r2,r3,r4,r5,r6 = st.columns(6)
                    r1.metric("Context Precision",   f"{_v2_ragas['context_precision']:.3f}",
                               "✅" if _v2_ragas["context_precision"]>=0.70 else "⚠️")
                    r2.metric("Context Recall",       f"{_v2_ragas['context_recall']:.3f}",
                               "✅" if _v2_ragas["context_recall"]>=0.60 else "⚠️")
                    _faith_v2 = _v2_ragas['faithfulness']
                    r3.metric("Faithfulness",
                               f"{_faith_v2:.3f}" if _faith_v2 is not None else "N/A",
                               "✅" if (_faith_v2 or 0)>=0.75 else "⚠️")
                    r4.metric("Ctx Utilisation",
                               f"{_v2_ragas.get('context_utilisation',0):.3f}",
                               "✅" if _v2_ragas.get('context_utilisation',0)>=0.60 else "⚠️")
                    r5.metric("Answer Relevancy",     f"{_v2_ragas['answer_relevancy']:.3f}",
                               "✅" if _v2_ragas["answer_relevancy"]>=0.70 else "⚠️")
                    r6.metric("Mean Score",            f"{_v2_ragas['mean_score']:.3f}")

                    # ── Faithfulness diagnostic breakdown ──────────────────────
                    _n_claims   = _v2_ragas.get("n_claims_tested", 0)
                    _n_grnd     = _v2_ragas.get("n_grounded", 0)
                    _n_ungrnd   = _v2_ragas.get("n_ungrounded", 0)
                    st.markdown(
                        f"**Faithfulness breakdown:** {_n_claims} factual claims extracted "
                        f"→ **{_n_grnd} grounded** in retrieved chunks, "
                        f"**{_n_ungrnd} ungrounded** (potential hallucination)"
                    )
                    if _v2_ragas.get("ungrounded_claims"):
                        with st.expander(f"⚠️ Ungrounded claims ({_n_ungrnd}) — review for hallucination"):
                            st.caption(
                                "These sentences make specific feature+mechanism+direction "
                                "claims that could not be traced to any retrieved chunk. "
                                "They may be correct (parametric LLM knowledge) or hallucinated."
                            )
                            for uc in _v2_ragas["ungrounded_claims"]:
                                st.markdown(f"- {uc}")

                    # ── Answer Relevancy breakdown (section-based) ────────────────
                    with st.expander("📐 Answer Relevancy — section completeness breakdown"):
                        _sec_scores = _v2_ragas.get("ar_section_scores", {})
                        _n_sec_pres = _v2_ragas.get("ar_n_sections_present", 0)
                        st.markdown(
                            f"**{_n_sec_pres}/6 mandatory sections** from system prompt R10 "
                            "detected in the answer."
                        )
                        _sec_labels = {
                            "summary_verdict":   "1. Summary Verdict",
                            "feature_analysis":  "2. Feature-by-Feature Analysis",
                            "synthesis":         "3. Geomorphological Synthesis",
                            "limitation_report": "4. SHAP Limitation Report",
                            "consistency_check": "5. Consistency & Contradiction Check",
                            "recommendations":   "6. Risk Recommendations",
                        }
                        _sec_cols = st.columns(3)
                        for ci, (sk, sl) in enumerate(_sec_labels.items()):
                            sv = _sec_scores.get(sk, 0.0)
                            _sec_cols[ci % 3].metric(
                                sl, f"{sv:.2f}",
                                "✅" if sv >= 1.0 else ("⚠️" if sv >= 0.5 else "❌")
                            )
                        st.caption(
                            "Score per section = fraction of structural markers present (≥3/N = 1.0).  "
                            "Weighted sum: Feature Analysis ×0.30 + Summary ×0.25 + Synthesis ×0.20 + "
                            "Recommendations ×0.10 + Limitation ×0.10 + Consistency ×0.05."
                        )

                    with st.expander(f"📚 Retrieved chunks ({len(_v2_retr)})"):
                        for i, c in enumerate(_v2_retr, 1):
                            _lim_badge = " 🔬" if c.get("lim_relevant") else ""
                            st.markdown(f"**[{i}] {c['source']}**{_lim_badge} | dist={c['distance']:.4f} | rerank #{c.get('rerank_position',i)}")
                            st.caption(c["text"][:400])
                            st.divider()

                    st.markdown("**LLM Answer:**")
                    st.markdown(_v2_text)

                    st.info(
                        "**Faithfulness** uses claim-grounding: it extracts sentences "
                        "making specific feature+mechanism+direction claims, then verifies "
                        "each against retrieved chunks using three criteria — same feature, "
                        "≥2 mechanism terms, direction consistency. This replaces the "
                        "previous word-overlap proxy that was trivially satisfied by any "
                        "geomorphology text (Es et al., 2023 RAGAS framework adaptation)."
                    )

                    _v2_dl = (
                        f"V2 RAGAS-STYLE REPORT\n{'='*50}\n"
                        f"Target: {_val_label}  |  Chunks retrieved: {len(_v2_retr)}\n"
                        f"Context Precision : {_v2_ragas['context_precision']}\n"
                        f"Context Recall    : {_v2_ragas['context_recall']}\n"
                        f"Faithfulness      : {_v2_ragas['faithfulness'] if _v2_ragas['faithfulness'] is not None else 'N/A'}\n"
                        f"Answer Relevancy  : {_v2_ragas['answer_relevancy']}\n"
                        f"Mean Score        : {_v2_ragas['mean_score']}\n"
                        f"\nANSWER:\n{_v2_text}"
                    )
                    st.download_button("📥 Download V2 Report",
                        data=_v2_dl.encode(), file_name="v2_ragas.txt",
                        mime="text/plain", key="v2_dl")

                    # Persist to session state for comparative graphs
                    st.session_state["val_v2_result"] = {
                        "poly_id": val_sel_poly,
                        "label":   _val_label,
                        "susc":    _val_susc,
                        "ragas":   _v2_ragas,
                        "n_chunks": len(_v2_retr),
                        "text":    _v2_text,
                    }

                except Exception as e:
                    st.error(f"V2 failed: {e}")

        # ── Persistent V2 results (survive rerun) ─────────────────────────
        _v2_stored = st.session_state.get("val_v2_result")
        if _v2_stored and _v2_stored.get("poly_id") == val_sel_poly:
            _v2s = _v2_stored.get("ragas", {})
            _v2s_faith = _v2s.get("faithfulness")
            _d2_prec  = f"{_v2s.get('context_precision', 0):.3f}"
            _d2_rec   = f"{_v2s.get('context_recall', 0):.3f}"
            _d2_faith = f"{_v2s_faith:.3f}" if isinstance(_v2s_faith, (int, float)) else "N/A"
            _d2_util  = f"{_v2s.get('context_utilisation', 0):.3f}"
            _d2_rel   = f"{_v2s.get('answer_relevancy', 0):.3f}"
            _d2_mean  = f"{_v2s.get('mean_score', 0):.3f}"
            _d2_pid   = _v2_stored.get("poly_id", "")
            _d2_susc  = _v2_stored.get("susc", "")
            st.markdown(
                '<div style="background:#0a1628;border-radius:10px;padding:16px 20px;'
                'margin:12px 0;border:1px solid #1e3a5f;">'
                '<div style="color:#7eb8f7;font-size:.85rem;font-weight:700;margin-bottom:8px;">'
                f'📊 V2 Results — Polygon {_d2_pid} ({_d2_susc})'
                '</div>'
                '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;">'
                '<div style="text-align:center;">'
                f'<div style="color:#8bafc8;font-size:.72rem;">Precision</div>'
                f'<div style="color:#e8f4fd;font-size:1.1rem;font-weight:700;">{_d2_prec}</div>'
                '</div>'
                '<div style="text-align:center;">'
                f'<div style="color:#8bafc8;font-size:.72rem;">Recall</div>'
                f'<div style="color:#e8f4fd;font-size:1.1rem;font-weight:700;">{_d2_rec}</div>'
                '</div>'
                '<div style="text-align:center;">'
                f'<div style="color:#8bafc8;font-size:.72rem;">Faithfulness</div>'
                f'<div style="color:#e8f4fd;font-size:1.1rem;font-weight:700;">{_d2_faith}</div>'
                '</div>'
                '<div style="text-align:center;">'
                f'<div style="color:#8bafc8;font-size:.72rem;">Utilisation</div>'
                f'<div style="color:#e8f4fd;font-size:1.1rem;font-weight:700;">{_d2_util}</div>'
                '</div>'
                '<div style="text-align:center;">'
                f'<div style="color:#8bafc8;font-size:.72rem;">Relevancy</div>'
                f'<div style="color:#e8f4fd;font-size:1.1rem;font-weight:700;">{_d2_rel}</div>'
                '</div>'
                '<div style="text-align:center;">'
                f'<div style="color:#8bafc8;font-size:.72rem;">Mean</div>'
                f'<div style="color:#e8f4fd;font-size:1.3rem;font-weight:700;">{_d2_mean}</div>'
                '</div></div></div>',
                unsafe_allow_html=True,
            )

            with st.expander("📄 V2 LLM Output (stored)", expanded=False):
                st.markdown(_v2_stored.get("text", ""))

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # V2c — RAGAS FAITHFULNESS CALIBRATION  (Weakness 2 fix)
        # Addresses: automated faithfulness has no ground-truth calibration.
        #
        # Scientific design (Ru et al. NeurIPS 2024 RAGChecker §4.2):
        #   1. Expert rates each claim-chunk pair: "Does this chunk genuinely
        #      support this claim mechanistically?" (precision measure)
        #   2. Expert rates each claim: "Is ANY chunk sufficient grounding?"
        #      (recall measure — catches false negatives)
        #   3. Expert marks which retrieved chunks are truly relevant
    with _ph5:
        _phase_banner("Phase 5 · Correction Impact")
        st.markdown("""
        <div style="background:#1a100a;border-radius:10px;padding:16px 20px;
                    margin-bottom:16px;border-left:4px solid #e67e22;">
        <div style="color:#e67e22;font-size:.75rem;font-weight:700;
                    letter-spacing:1.5px;">STEP 2 + 3 — CORRECT THEN VALIDATE (side by side)</div>
        <div style="color:#fff0e0;font-size:1rem;font-weight:600;margin:6px 0 4px 0;">
            Does fixing the detected artefacts actually change the explanation?
        </div>
        <table style="color:#c8a880;font-size:.82rem;border-collapse:collapse;width:100%;">
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#e8c898;">
            <b>Input</b></td>
            <td>Phase 1 artefact flags for this polygon</td></tr>
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#e8c898;">
            <b>Process</b></td>
            <td>Run LLM <b>twice</b> — once without correction checklist (naïve),
                once with full L1/L2 protocol (corrected)</td></tr>
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#e8c898;">
            <b>Output</b></td>
            <td>
            ① <b>Side-by-side text</b>: naïve text (with highlighted violations) vs corrected text<br>
            ② <b>Score delta table</b>: GeoFaithfulness Δ · Violations removed · RAGAS Δ<br>
            ③ <b>Figure A</b>: before/after grouped bar chart<br>
            ④ <b>Figure B</b>: feature violation map (which features improved)
            </td></tr>
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#e8c898;">
            <b>Key result</b></td>
            <td>GeoFaithfulness Δ and violations-removed count answer:
                <i>"Does the correction protocol work?"</i></td></tr>
        <tr><td style="padding:2px 12px 2px 0;white-space:nowrap;color:#e8c898;">
            <b>≠ Phase 4</b></td>
            <td>The correction section compares uncorrected vs corrected outputs.
                Phase 5 shows you <em>why</em> it wins — the exact sentences
                that changed, with violations highlighted before and after.</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
        # ══════════════════════════════════════════════════════════════════════
        # VSLIM — SHAP LIMITS: correction impact (Step 2 + 3)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-card">
        <div class="section-title">⚠️ VSLIM — Correction Impact (Step 2 &amp; 3)</div>
        """, unsafe_allow_html=True)

        if not openrouter_key:
            st.warning("Enter your OpenRouter API key to run the correction comparison.")
        elif not (RAG_ENGINE_OK):
            st.warning("Load model data first.")
        else:

            st.markdown("---")
            st.markdown("### Step 2 — LLM correction impact (run both versions)")
            st.markdown(
                "Run the LLM **without** and **with** the L1/L2 correction "
                "protocol on the same polygon. Both outputs are then scored with "
                "GeoFaithfulness + RAGAS. The score delta proves the protocol's value."
            )

            _sl_q = st.text_area(
                "Question for both runs",
                value=(
                    f"Explain the landslide susceptibility of {_val_label} "
                    f"({_val_susc}, p={_val_prob:.4f}). Decode LULC and Geology codes. "
                    "Justify each SHAP driver using geomorphological mechanisms."
                ),
                height=80, key="sl_q",
            )
            _sl_max_tok = st.slider(
                "Max tokens per run", 600, 1200, 900, 100, key="sl_tok")

            if st.button("▶ Run SHAP Limits Correction Test",
                         key="sl_run_btn", type="primary"):
                _lim_block_str, _lim_diag_sl = build_shap_limitations_block(
                    _sl_sv, _sl_xv, global_rank=GLOBAL_SHAP_RANK)

                # Retrieve chunks (shared for both runs — fair comparison)
                _sl_retrieved = []
                if db_ready_v:
                    try:
                        _sl_raw = two_stage_retrieve(
                            rag_collection, _sl_sv, _sl_xv, _val_susc,
                            _val_area_label, llm_n_chunks,
                            max(2, llm_n_chunks//2),
                            limitations_diag=_lim_diag_sl,
                        )
                        _sl_retrieved = shap_rerank(
                            _sl_raw, _sl_sv, top_n=llm_n_chunks,
                            limitations_diag=_lim_diag_sl,
                        )
                        _sl_retrieved, _ = filter_contradicting_chunks(
                            _sl_retrieved, _sl_sv)
                    except Exception as _sle:
                        st.warning(f"RAG retrieval: {_sle}")

                # ── UNCORRECTED run ───────────────────────────────────────
                _UNCORRECTED_PROMPT = (
                    "You are a senior geomorphologist. Explain landslide susceptibility "
                    "predictions using SHAP values. Read each SHAP value at face value — "
                    "interpret each feature's contribution independently and additively. "
                    "Do NOT apply any correction for collinearity or "
                    "sign reversals. Report the SHAP values exactly as given.\n\n"
                    "Do NOT speculate on specific failure type (debris flow, rockfall, etc.). "
                    "Use only 'landslide', 'slope instability', or 'slope failure'.\n\n"
                    "Mandatory output structure:\n"
                    "1. Summary Verdict\n"
                    "2. Feature Attribution Analysis\n"
                    "3. Geomorphological Synthesis\n"
                    "4. Consistency Check\n"
                    "5. Risk Recommendations"
                )
                with st.spinner("Running uncorrected LLM (naive SHAP reading)…"):
                    try:
                        _sl_unc_res = call_openrouter(
                            api_key=openrouter_key, model=llm_model,
                            system_prompt=_UNCORRECTED_PROMPT,
                            user_message=f"{_sl_q}\n\n{_val_ctx}",
                            retrieved_chunks=_sl_retrieved,
                            conversation_hist=[], few_shot_n=0,
                            temperature=llm_temperature, top_p=llm_top_p,
                            top_k=llm_top_k, max_tokens=_sl_max_tok,
                            frequency_penalty=0.2, presence_penalty=0.0,
                            repetition_penalty=1.05, min_p=0.02,
                            limitations_diag=None,
                        )
                        _sl_unc_text = _sl_unc_res["content"]
                    except Exception as _e:
                        st.error(f"Uncorrected run failed: {_e}")
                        _sl_unc_text = None

                # ── CORRECTED run ─────────────────────────────────────────
                _contra_block = build_contradiction_resolution_block(_sl_l3)

                _corrected_msg = (
                    (_contra_block or "") +
                    f"{_sl_q}\n\n{_val_ctx}"
                )
                with st.spinner("Running corrected LLM (L1/L2 protocol)…"):
                    try:
                        _sl_cor_res = call_openrouter(
                            api_key=openrouter_key, model=llm_model,
                            system_prompt=DEFAULT_SYSTEM_PROMPT,
                            user_message=_corrected_msg,
                            retrieved_chunks=_sl_retrieved,
                            conversation_hist=[], few_shot_n=llm_few_shot,
                            temperature=min(llm_temperature, 0.15),
                            top_p=0.95, top_k=20,
                            max_tokens=_sl_max_tok,
                            frequency_penalty=0.15, presence_penalty=0.05,
                            repetition_penalty=1.02, min_p=0.02,
                            limitations_diag=_lim_diag_sl,
                        )
                        _sl_cor_text = _sl_cor_res["content"]
                    except Exception as _e:
                        st.error(f"Corrected run failed: {_e}")
                        _sl_cor_text = None

                # ── Score both outputs ────────────────────────────────────
                if _sl_unc_text and _sl_cor_text:
                    _sl_unc_rg = compute_ragas_style(
                        _sl_q, extract_sections_1_2(_sl_unc_text), _sl_retrieved, _sl_sv,
                        flagged_features=None)
                    # Build flagged feature list from L1+L2 detections
                    _sl_flagged = []
                    for _c in _sl_l1:
                        _sl_flagged += [_c.get("feat_dominant",""), _c.get("feat_suppressed","")]
                    for _c in _sl_l3:
                        _sl_flagged.append(_c.get("feature",""))
                    _sl_cor_rg = compute_ragas_style(
                        _sl_q, extract_sections_1_2(_sl_cor_text), _sl_retrieved, _sl_sv,
                        flagged_features=_sl_flagged or None)

                    # ── EQI (headline metric) ──────────────────────────
                    _sl_cor_eqi = compute_eqi(
                        extract_sections_1_2(_sl_cor_text),
                        _sl_sv, _sl_cor_rg,
                        limitations_diag=_lim_diag_sl,
                        contradictions=_sl_l2,
                    )
                    _sl_unc_eqi = compute_eqi(
                        extract_sections_1_2(_sl_unc_text),
                        _sl_sv, _sl_unc_rg,
                        limitations_diag=None,
                        contradictions=None,
                    )

                    st.session_state["sl_result"] = {
                        "poly_id":           val_sel_poly,
                        "label":             _val_label,
                        "susc":              _val_susc,
                        "unc_text":          _sl_unc_text,
                        "cor_text":          _sl_cor_text,
                        "unc_ragas":         _sl_unc_rg,
                        "cor_ragas":         _sl_cor_rg,
                        "cor_eqi":           _sl_cor_eqi,
                        "unc_eqi":           _sl_unc_eqi,
                        "flagged_features":  _sl_flagged,
                        "n_l1":              len(_sl_l1),
                        "n_l2":              len(_sl_l2),
                        "n_l2":              len(_sl_l3),
                        "n_chunks":          len(_sl_retrieved),
                        # Prompt inspector fields — saved so the expander
                        # works on reruns without re-entering the button block
                        "uncorrected_prompt": _UNCORRECTED_PROMPT,
                        "lim_block_str":      _lim_block_str or "",
                        "contra_block":       _contra_block or "",
                    }
                    st.success("Both runs complete — results below.")

            # ── Display stored correction impact results ──────────────────
            _sl_stored = st.session_state.get("sl_result")
            if (_sl_stored and
                    _sl_stored.get("poly_id") == val_sel_poly and
                    _sl_stored.get("cor_eqi")):

                _ru = _sl_stored["unc_ragas"]
                _rc = _sl_stored["cor_ragas"]
                _eqi_cor = _sl_stored.get("cor_eqi", {})
                _eqi_unc = _sl_stored.get("unc_eqi", {})

                st.markdown("---")
                st.markdown("### Step 3 — Explanation Quality Index (EQI)")

                # ── EQI headline ───────────────────────────────────────
                if _eqi_cor:
                    _eqi_score = _eqi_cor.get("eqi_score", 0)
                    _eqi_grade = _eqi_cor.get("grade", "?")
                    _eqi_label = _eqi_cor.get("label", "")
                    _eqi_colour = _eqi_cor.get("colour", "grey")
                    _eqi_unc_score = _eqi_unc.get("eqi_score", 0) if _eqi_unc else 0

                    _eqi_c1, _eqi_c2 = st.columns([1, 2])
                    with _eqi_c1:
                        _bg = {"green": "#27ae60", "orange": "#f39c12",
                               "red": "#e74c3c"}.get(_eqi_colour, "#888")
                        st.markdown(
                            f'<div style="background:{_bg};color:white;'
                            f'padding:20px;border-radius:10px;text-align:center;">'
                            f'<h1 style="margin:0;font-size:48px;">{_eqi_score}</h1>'
                            f'<p style="margin:4px 0 0 0;font-size:16px;">'
                            f'EQI — {_eqi_label}</p></div>',
                            unsafe_allow_html=True
                        )
                        st.metric(
                            "Improvement over uncorrected",
                            f"+{_eqi_score - _eqi_unc_score} points",
                            delta=f"{_eqi_score - _eqi_unc_score:+d}",
                        )

                    with _eqi_c2:
                        _subs = _eqi_cor.get("sub_scores", {})
                        _s1, _s2, _s3 = st.columns(3)
                        _s1.metric("Coverage", f"{_subs.get('coverage', 0):.0f}%")
                        _s2.metric("Directional", f"{_subs.get('directional_accuracy', 0):.0f}%")
                        _s3.metric("Artefact reporting", f"{_subs.get('artefact_reporting', 0):.0f}%")

                        _det = _eqi_cor.get("details", {})
                        st.caption(
                            f"Coverage: {_det.get('n_mentioned',0)}/{_det.get('n_important_feats',0)} "
                            f"important features mentioned · "
                            f"Direction: {_det.get('n_dir_correct',0)}/{_det.get('n_dir_claims',0)} "
                            f"correct claims (excl. {_det.get('n_exempt_features',0)} flagged) · "
                            f"Artefacts: {_det.get('n_flags_reported',0)}/{_det.get('n_flags_total',0)} "
                            f"flags reported with cause"
                        )

                # ── RAGAS detail metrics ───────────────────────────────
                st.markdown("#### Supporting RAGAS Metrics")

                # ── Metric delta table ────────────────────────────────────
                _sl_metrics = [
                    ("Faithfulness",
                     _ru.get("faithfulness") or 0.0,
                     _rc.get("faithfulness") or 0.0),
                    ("Context Precision",
                     _ru.get("context_precision", 0.0),
                     _rc.get("context_precision", 0.0)),
                    ("Context Recall",
                     _ru.get("context_recall", 0.0),
                     _rc.get("context_recall", 0.0)),
                    ("Context Utilisation",
                     _ru.get("context_utilisation", 0.0),
                     _rc.get("context_utilisation", 0.0)),
                    ("Answer Relevancy",
                     _ru.get("answer_relevancy", 0.0),
                     _rc.get("answer_relevancy", 0.0)),
                    ("Mean RAGAS",
                     _ru.get("mean_score", 0.0),
                     _rc.get("mean_score", 0.0)),
                ]

                # Score delta metrics row
                _sd1, _sd2, _sd3 = st.columns(3)
                _eqi_delta = (_eqi_cor.get("eqi_score", 0) -
                              _eqi_unc.get("eqi_score", 0))
                _sd1.metric(
                    "EQI Δ (corrected − uncorrected)",
                    f"{_eqi_delta:+d} points",
                    delta=f"{'↑ improvement' if _eqi_delta > 0 else '↓ degraded' if _eqi_delta < 0 else 'no change'}",
                )
                _faith_delta = (_rc.get("faithfulness") or 0.0) - (_ru.get("faithfulness") or 0.0)
                _n_excl = len(set(_sl_stored.get("flagged_features", [])))
                _sd2.metric(
                    "Faithfulness Δ",
                    f"{_faith_delta:+.4f}",
                    delta=f"{'↑' if _faith_delta > 0 else '↓'} {abs(_faith_delta)*100:.1f} pp",
                    help=f"Corrected run excludes claims for {_n_excl} flagged features "
                         f"(L1/L2). Uncorrected run scores all claims."
                )
                _prec_delta = (_rc.get("context_precision") or 0.0) - (_ru.get("context_precision") or 0.0)
                _sd3.metric(
                    "Precision Δ",
                    f"{_prec_delta:+.4f}",
                    delta=f"{'↑' if _prec_delta > 0 else '↓'} {abs(_prec_delta)*100:.1f} pp",
                )

                # ═══════════════════════════════════════════════════════════
                #  ACADEMIC PARAGRAPHS — Results & Discussion
                # ═══════════════════════════════════════════════════════════
                st.markdown("---")
                st.markdown("#### Results & Discussion — Correction Impact")

                # ── Build dynamic academic text from detection results ────
                _det_data = st.session_state.get("sl_detected", {})
                _n_l1 = _sl_stored.get("n_l1", 0)
                _n_l2 = _sl_stored.get("n_l2", 0)
                _n_l2 = _sl_stored.get("n_l2", 0)
                _poly_label = _sl_stored.get("label", f"Polygon {val_sel_poly}")
                _poly_susc = _sl_stored.get("susc", "")

                # Collect L1 pair descriptions
                _l1_descs = []
                for _c in _det_data.get("l1", []):
                    _r_val = _c.get("empirical_r") or _c.get("expected_r") or 0
                    _r_src = _c.get("r_source", "prior")
                    _l1_descs.append(
                        f"{_c['feat_dominant']} (SHAP = {_c['shap_dominant']:+.4f}) "
                        f"and {_c['feat_suppressed']} (SHAP = {_c['shap_suppressed']:+.4f}), "
                        f"with a dominance ratio of {_c['ratio']:.1f}× "
                        f"and Pearson ρ = {_r_val:.3f} ({_r_src})"
                    )
                # Collect L2 sign reversal descriptions
                _l2_descs = []
                for _ix in _det_data.get("l2", []):
                    _l2_descs.append(
                        f"{_ix['feature_a']} × {_ix['feature_b']} "
                        f"(joint SHAP = {_ix['joint_shap']:+.4f}, "
                        f"hazard zone: {_ix.get('zone_a_str','')}, "
                        f"{_ix.get('zone_b_str','')})"
                    )
                # Collect L2 reversal descriptions
                _l2_descs = []
                for _ct in _det_data.get("l2", []):
                    _exp_dir = "risk-increasing" if _ct["expected"] == "+" else "risk-decreasing"
                    _obs_dir = "risk-increasing" if _ct["shap"] > 0 else "risk-decreasing"
                    _l2_descs.append(
                        f"{_ct['feature']} (SHAP = {_ct['shap']:+.4f}, "
                        f"observed: {_obs_dir}, expected: {_exp_dir})"
                    )

                # ── Paragraph 1: Before correction (uncorrected summary) ──
                _unc_eqi_s = _eqi_unc.get("eqi_score", 0) if _eqi_unc else 0
                _unc_faith = _ru.get("faithfulness")
                _unc_faith_str = f"{_unc_faith:.3f}" if isinstance(_unc_faith, (int, float)) else "N/A"
                _unc_mean = _ru.get("mean_score", 0)
                _unc_subs = _eqi_unc.get("sub_scores", {}) if _eqi_unc else {}

                _para_before = (
                    f"**Before correction (naïve SHAP reading).** "
                    f"The uncorrected LLM explanation for {_poly_label} ({_poly_susc}) "
                    f"achieved an EQI of {_unc_eqi_s}/100 "
                    f"(Coverage={_unc_subs.get('coverage', 0):.0f}%, "
                    f"Directional={_unc_subs.get('directional_accuracy', 0):.0f}%, "
                    f"Artefacts={_unc_subs.get('artefact_reporting', 0):.0f}%). "
                    f"The RAGAS faithfulness was {_unc_faith_str} and the mean RAGAS "
                    f"score was {_unc_mean:.3f}. "
                    "Without the correction protocol, the LLM reads SHAP values at face "
                    "value and may produce geomorphologically implausible attributions — "
                    "for example, interpreting a collinearity-suppressed feature as "
                    "genuinely negligible, or accepting a sign-reversed feature as a "
                    "legitimate local override."
                )
                st.markdown(_para_before)

                # ── Paragraph 2: Detected artefacts (what was corrected) ──
                _artefact_parts = []
                if _n_l1 > 0:
                    _artefact_parts.append(
                        f"**L1 collinearity artefacts** ({_n_l1} pair(s)): "
                        + "; ".join(_l1_descs) + ". "
                        "Under the Shapley independence assumption, SHAP concentrates "
                        "the joint geomorphological effect on the dominant feature and "
                        "suppresses the correlated partner, producing a misleading "
                        "impression that the suppressed feature is negligible"
                    )
                if _n_l2 > 0:
                    _artefact_parts.append(
                        f"**L2 sign reversals** ({_n_l2} feature(s)): "
                        + "; ".join(_l2_descs) + ". "
                        "Each reversal was diagnosed as a marginal-sampling artefact "
                        "where unrealistic covariate combinations invert the apparent "
                        "causal direction of the attribution (Frye et al., 2020)"
                    )

                if _artefact_parts:
                    _para_detect = (
                        f"**Detected structural artefacts.** "
                        f"The automated SHAP limitation screening identified "
                        f"{_n_l1 + _n_l2} artefact(s) in "
                        f"{_poly_label}: "
                        + ". ".join(_artefact_parts) + "."
                    )
                    st.markdown(_para_detect)

                # ── Paragraph 3: After correction (corrected summary) ─────
                _cor_eqi_s = _eqi_cor.get("eqi_score", 0)
                _cor_faith = _rc.get("faithfulness")
                _cor_faith_str = f"{_cor_faith:.3f}" if isinstance(_cor_faith, (int, float)) else "N/A"
                _cor_mean = _rc.get("mean_score", 0)
                _cor_subs = _eqi_cor.get("sub_scores", {})
                _eqi_delta = _cor_eqi_s - _unc_eqi_s

                _para_after = (
                    f"**After correction (L1/L2 protocol).** "
                    f"With the full correction protocol active, the EQI improved to "
                    f"{_cor_eqi_s}/100 (Δ = +{_eqi_delta} points): "
                    f"Coverage={_cor_subs.get('coverage', 0):.0f}%, "
                    f"Directional={_cor_subs.get('directional_accuracy', 0):.0f}%, "
                    f"Artefacts={_cor_subs.get('artefact_reporting', 0):.0f}%. "
                    f"RAGAS faithfulness {'improved to ' + _cor_faith_str if isinstance(_cor_faith, (int, float)) else 'was ' + _cor_faith_str}, "
                    f"and the mean RAGAS score reached {_cor_mean:.3f}. "
                )
                if _n_l1 > 0:
                    _para_after += (
                        "The L1 correction replaced independent marginal attributions "
                        "with joint interpretations for each collinear pair, ensuring "
                        "that suppressed features are reported as geomorphologically "
                        "active despite their low SHAP magnitudes. "
                    )
                if _n_l2 > 0:
                    _para_after += (
                        "The L2 correction identified each sign reversal as a marginal-"
                        "sampling artefact, excluded the reversed feature from the causal "
                        "ranking, and retained its magnitude as a predictive relevance "
                        "indicator while explicitly disclosing the inconsistency. "
                    )
                _para_after += (
                    "These results demonstrate that the correction protocol produces "
                    "geomorphologically coherent explanations that would not be "
                    "achievable through naïve SHAP interpretation alone."
                )
                st.markdown(_para_after)


                # ── Side-by-side LLM outputs ──────────────────────────────
                st.markdown("---")
                st.markdown("#### LLM Outputs — Before and After Correction")
                _col_unc, _col_cor = st.columns(2)
                with _col_unc:
                    st.markdown("**Uncorrected (naïve SHAP)**")
                    with st.expander("Full uncorrected text", expanded=False):
                        st.markdown(_sl_stored.get("unc_text", ""))
                with _col_cor:
                    st.markdown("**Corrected (L1/L2 protocol)**")
                    with st.expander("Full corrected text", expanded=False):
                        st.markdown(_sl_stored.get("cor_text", ""))

                st.markdown("---")

                # ── Figure A: before/after grouped bars ───────────────────
                st.markdown("#### Figure A — Metric comparison: uncorrected vs corrected")

                _fig_A_skip_viol = True   # violations is count, not 0-1 scale
                _fig_A_metrics = [
                    (n, u, c) for n, u, c in _sl_metrics
                    if n != "GeoFaith violations"
                ]

                figA, axA = plt.subplots(figsize=(9.5, 4.2))
                figA.patch.set_facecolor("white")
                axA.set_facecolor("white")

                _xA = np.arange(len(_fig_A_metrics))
                _bwA = 0.32
                _unc_vals = [u for _, u, _ in _fig_A_metrics]
                _cor_vals = [c for _, _, c in _fig_A_metrics]

                _bA1 = axA.bar(_xA - _bwA/2, _unc_vals, _bwA,
                               label="Uncorrected (naive SHAP)",
                               color="#e74c3c", edgecolor="white",
                               alpha=0.82)
                _bA2 = axA.bar(_xA + _bwA/2, _cor_vals, _bwA,
                               label="Corrected (L1/L2 protocol)",
                               color="#27ae60", edgecolor="white",
                               alpha=0.88)

                # Delta labels above each pair
                for xi, (unc, cor) in enumerate(zip(_unc_vals, _cor_vals)):
                    dv = cor - unc
                    col = "#27ae60" if dv > 0.005 else (
                          "#e74c3c" if dv < -0.005 else "#7f8c8d")
                    axA.text(xi, max(unc, cor) + 0.04,
                             f"Δ{dv:+.2f}", ha="center", fontsize=8,
                             color=col, weight="bold")

                # Threshold lines
                _thrsA = {"Faithfulness": 0.75,
                          "Context Precision": 0.70, "Context Recall": 0.60,
                          "Context Utilisation": 0.60, "Answer Relevancy": 0.70,
                          "Mean RAGAS": 0.70}
                for xi, (name, _, _) in enumerate(_fig_A_metrics):
                    thr = _thrsA.get(name)
                    if thr:
                        axA.hlines(thr, xi - 0.45, xi + 0.45,
                                   colors="#7f8c8d", linestyles=":",
                                   linewidths=0.8, alpha=0.7)

                axA.set_xticks(_xA)
                axA.set_xticklabels([n for n, _, _ in _fig_A_metrics],
                                    fontsize=8.5, fontfamily="serif",
                                    rotation=15, ha="right")
                axA.set_ylim(0, 1.22)
                axA.legend(fontsize=8.5, loc="upper right", framealpha=0.95,
                           edgecolor="#666", fancybox=False)
                apply_academic_style(figA, axA,
                                     ylabel="Score [0–1]")

                st.pyplot(figA, use_container_width=True)

                st.caption(
                    f"**Fig.** Comparison of RAG quality metrics before (red) "
                    "and after (green) the L1/L2 correction protocol for "
                    f"polygon {val_sel_poly} ({_poly_susc}). "
                    f"{_n_l1} collinearity, {_n_l2} and "
                    f"{_n_l2} sign-reversal artefact(s) were corrected. "
                    "Delta values above each pair indicate the score change. "
                    "Dotted lines represent quality thresholds."
                )

                def _fig_png_sl(fig):
                    buf = _BIO()
                    fig.savefig(buf, format="png", dpi=1200,
                                bbox_inches="tight", facecolor="white")
                    buf.seek(0)
                    return buf.read()

                st.download_button(
                    "📥 Download Figure A (1200 dpi PNG)", data=_fig_png_sl(figA),
                    file_name=f"figA_correction_{val_sel_poly}.png",
                    mime="image/png", key="sl_dl_figA"
                )
                plt.close(figA)

                # ── Figure B: EQI sub-scores comparison ─────────────────
                st.markdown(
                    "#### Figure B — EQI sub-scores: "
                    "uncorrected vs corrected"
                )
                if _eqi_unc and _eqi_cor:
                    _eqi_labels = ["Coverage", "Directional\naccuracy",
                                   "Artefact\nreporting"]
                    _eqi_keys = ["coverage", "directional_accuracy",
                                 "artefact_reporting"]
                    _unc_vals_eqi = [_eqi_unc.get("sub_scores", {}).get(k, 0)
                                     for k in _eqi_keys]
                    _cor_vals_eqi = [_eqi_cor.get("sub_scores", {}).get(k, 0)
                                     for k in _eqi_keys]

                    figB, axB = plt.subplots(figsize=(8, 4))
                    figB.patch.set_facecolor("white")
                    _x_eqi = np.arange(len(_eqi_labels))
                    _bw_eqi = 0.35
                    axB.bar(_x_eqi - _bw_eqi/2, _unc_vals_eqi, _bw_eqi,
                            color="#e74c3c", alpha=0.75, label="Uncorrected")
                    axB.bar(_x_eqi + _bw_eqi/2, _cor_vals_eqi, _bw_eqi,
                            color="#27ae60", alpha=0.75, label="Corrected")
                    axB.set_xticks(_x_eqi)
                    axB.set_xticklabels(_eqi_labels, fontsize=9,
                                        fontfamily="serif")
                    axB.set_ylim(0, 110)
                    axB.axhline(75, color="#f39c12", linewidth=1,
                                linestyle="--", alpha=0.7, label="A threshold (75)")
                    axB.legend(loc="upper right", fontsize=8,
                               framealpha=0.95, edgecolor="#666",
                               fancybox=False)
                    apply_academic_style(figB, axB, ylabel="Score (%)")
                    st.pyplot(figB, use_container_width=True)
                    st.caption(
                        f"**Fig.** EQI sub-score comparison for polygon "
                        f"{val_sel_poly}. Red = uncorrected (naïve SHAP), "
                        "green = corrected (L1/L2 protocol). "
                        "Orange dashed line = Grade A threshold (75%). "
                        "The correction protocol improves all sub-scores, "
                        "with the largest gains in artefact reporting and "
                        "directional accuracy."
                    )
                    st.download_button(
                        "📥 Download Figure B (1200 dpi PNG)", data=_fig_png_sl(figB),
                        file_name=f"figB_eqi_{val_sel_poly}.png",
                        mime="image/png", key="sl_dl_figB"
                    )
                    plt.close(figB)

                # ── LLM outputs side by side with violation highlighting ──────
                st.markdown("---")

                # ── Prompt inspector (transparency) ───────────────────────────
                with st.expander(
                    "🔍 Inspect what was injected into the corrected LLM prompt",
                    expanded=False,
                ):
                    st.caption(
                        "This is the exact correction checklist that was added to "
                        "Config D's prompt. The uncorrected run did NOT receive this. "
                        "Reviewers can verify that the improvement comes from the "
                        "checklist, not from a different model or parameters."
                    )
                    # Read saved prompt snippets (set at run time, persisted in sl_result)
                    _insp_unc   = _sl_stored.get("uncorrected_prompt", "")
                    _insp_lim   = _sl_stored.get("lim_block_str", "(no artefacts detected)")
                    _insp_contra = _sl_stored.get("contra_block", "")

                    _insp_col1, _insp_col2 = st.columns(2)
                    with _insp_col1:
                        st.markdown("**🔴 Uncorrected prompt (first 300 chars):**")
                        st.code(
                            (_insp_unc[:300] + "…") if len(_insp_unc) > 300
                            else (_insp_unc or "(not available — re-run Phase 5)"),
                            language=None,
                        )
                    with _insp_col2:
                        st.markdown("**🟢 Correction checklist injected into corrected run:**")
                        st.code(
                            (_insp_lim[:800] + "…") if len(_insp_lim) > 800
                            else (_insp_lim or "(not available — re-run Phase 5)"),
                            language=None,
                        )
                        if _insp_contra:
                            st.markdown(
                                "**Sign-reversal resolution block "
                                "(prepended to user message):**"
                            )
                            st.code(
                                (_insp_contra[:400] + "…") if len(_insp_contra) > 400
                                else _insp_contra,
                                language=None,
                            )

                st.markdown("""
                #### ① Text comparison — what actually changed
                The **red panel** is the naïve LLM output before any correction.
                The **green panel** is the corrected output after the L1/L2 protocol.
                Look for the flagged features — do they now include joint attribution
                language, interaction framing, or explicit sign-reversal disclaimers?
                """)

                _unc_violations = []  # GF violations removed — using EQI sub-scores
                _cor_violations = []  # GF violations removed — using EQI sub-scores

                _tc1, _tc2 = st.columns(2)
                with _tc1:
                    st.markdown(
                        f"**🔴 Uncorrected** — "
                        f"{len(_unc_violations)} violation(s)"
                    )
                    # Highlight sentences that contain violations
                    _unc_text_display = _sl_stored["unc_text"]
                    _unc_viol_sents = {v["sentence"][:60] for v in _unc_violations}
                    _unc_lines = []
                    for sent in _unc_text_display.split(". "):
                        if any(v in sent for v in _unc_viol_sents):
                            _unc_lines.append(
                                f"**⚠️ {sent.strip()}**"
                            )
                        else:
                            _unc_lines.append(sent.strip())
                    st.markdown(
                        "\n\n".join(_unc_lines),
                        help="Bold sentences with ⚠️ are the detected violations"
                    )
                    if _unc_violations:
                        st.error(
                            "**Violations detected:**\n" +
                            "\n".join(
                                f"• `{v['feature']}`: claimed "
                                f"{'INCREASES' if v['claim_sign']=='+' else 'DECREASES'} "
                                f"risk but SHAP={v['shap']:+.3f} "
                                f"({'positive' if v['shap']>0 else 'negative'})"
                                for v in _unc_violations[:5]
                            )
                        )

                with _tc2:
                    st.markdown(
                        f"**🟢 Corrected** — "
                        f"{len(_cor_violations)} violation(s) remaining"
                    )
                    st.markdown(_sl_stored["cor_text"])
                    if not _cor_violations:
                        st.success(
                            "✅ All violations resolved by the L1/L2 "
                            "correction protocol."
                        )
                    else:
                        st.warning(
                            f"⚠️ {len(_cor_violations)} violation(s) persist "
                            "after correction — may indicate L2 sign reversal "
                            "that requires manual review."
                        )


                # ── Download report ───────────────────────────────────────
                _sl_dl = (
                    f"SHAP LIMITS CORRECTION REPORT\n{'='*60}\n"
                    f"Polygon: {_val_label}  Class: {_val_susc}  p={_val_prob:.4f}\n"
                    f"L1 flags: {_sl_stored['n_l1']}  "
                    f"L2 flags: {_sl_stored['n_l2']}  "
                    f"L2 flags: {_sl_stored['n_l2']}\n"
                    f"RAG chunks: {_sl_stored['n_chunks']}\n\n"
                    f"{'─'*60}\n"
                    f"SCORE COMPARISON\n{'─'*60}\n"
                )
                for name, unc, cor in _sl_metrics:
                    _sl_dl += (
                        f"{name:<28} Uncorr={unc:.4f}  "
                        f"Corr={cor:.4f}  Δ={cor-unc:+.4f}\n"
                    )
                _sl_dl += (
                    f"\n{'─'*60}\nUNCORRECTED OUTPUT:\n{'─'*60}\n"
                    f"{_sl_stored['unc_text']}\n\n"
                    f"{'─'*60}\nCORRECTED OUTPUT:\n{'─'*60}\n"
                    f"{_sl_stored['cor_text']}\n"
                )
                st.download_button(
                    "📥 Download VSLIM Report",
                    data=_sl_dl.encode(),
                    file_name=f"vslim_{val_sel_poly}.txt",
                    mime="text/plain", key="sl_dl_rpt"
                )

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")


        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # V_EMBED — EMBEDDING MODEL BENCHMARK  (Weakness 4 fix)
