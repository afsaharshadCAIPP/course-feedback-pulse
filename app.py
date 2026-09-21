import io
import re
import os
import json
import html as html_lib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except Exception:
    TRANSLATOR_AVAILABLE = False

try:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               AutoModelForSeq2SeqLM)
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

DISTILBERT_REPO = "Afsah-2027/coursera-multilingual-distilbert"
NLLB_REPO = "facebook/nllb-200-distilled-600M"

# FLORES-200 language codes used by NLLB-200, keyed by our detected language code
NLLB_LANG_CODES = {
    "en": "eng_Latn", "ur": "urd_Arab", "ar": "arb_Arab", "fa": "pes_Arab",
    "hi": "hin_Deva", "bn": "ben_Beng", "zh-cn": "zho_Hans", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "ru": "rus_Cyrl", "es": "spa_Latn",
    "fr": "fra_Latn", "de": "deu_Latn", "pt": "por_Latn", "it": "ita_Latn",
    "tr": "tur_Latn", "id": "ind_Latn", "vi": "vie_Latn", "nl": "nld_Latn",
}

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="Course Feedback Sentiment Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

SENT_COLORS = {"POSITIVE": "#22C55E", "NEUTRAL": "#F5C518", "NEGATIVE": "#DC2626"}
SENT_ORDER = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
SENT_EMOJI = {"POSITIVE": "😊", "NEUTRAL": "😐", "NEGATIVE": "😟"}
SENT_DOT = {"POSITIVE": "🟢", "NEUTRAL": "🟡", "NEGATIVE": "🔴"}

# =============================================================
# GLOBAL STYLE
# =============================================================
st.markdown("""
<style>
.main-header { font-size: 1.9rem !important; color: #0F172A !important; font-weight: 800 !important; margin-bottom: 0.2rem !important; line-height: 1.25 !important; }
.sub-header { font-size: 0.95rem !important; color: #4B5563 !important; margin-bottom: 1.2rem !important; }
.section-header { font-size: 1.55rem !important; font-weight: 800 !important; color: #0F172A !important; margin: 1.2rem 0 0.3rem 0 !important; line-height: 1.25 !important; }
.app-footer { text-align:center; color:#6B7280; font-size:0.85rem; margin-top:2.5rem; padding-top:1rem; border-top:1px solid #E5E7EB; }

.info-badge-card {
    background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px;
    padding:0.8rem 1rem; height:100%;
}
.info-badge-card .badge-title { font-weight:800; color:#1D4ED8; font-size:1rem; margin-bottom:0.3rem; }
.info-badge-card .badge-body { color:#1D4ED8; font-size:0.88rem; line-height:1.5; }

.model-desc-box {
    background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
    padding:0.8rem 1rem; height:100%;
}
.model-desc-box .model-desc-title { font-weight:800; color:#1D4ED8; margin-bottom:0.25rem; }
.model-desc-box .model-desc-body { color:#1D4ED8; font-size:0.88rem; line-height:1.45; }

.insight-box {
    background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
    padding:1rem 1.2rem;
}

.highlight-box {
    background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
    padding:1rem 1.2rem; color:#1D4ED8;
}
.highlight-box b { color:#1D4ED8; }

.pill {
    display:inline-block; border:1px solid #D1D5DB; border-radius:8px;
    padding:0.35rem 0.8rem; margin:0.2rem; font-size:0.92rem; color:#111827;
}

.check-row { padding:0.6rem 0; border-bottom:1px solid #F1F5F9; }

section[data-testid="stSidebar"] {
    background-color: #0B1B38;
}
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
section[data-testid="stSidebar"] * { color: #E5E7EB; }

.sidebar-logo {
    width: 56px; height: 56px; border-radius: 14px;
    background: #17233F; display:flex; align-items:center; justify-content:center;
    font-size: 28px; margin-bottom: 0.6rem;
}
.sidebar-title { font-size: 1.45rem; font-weight: 800; color: #FFFFFF; line-height:1.25; margin-bottom: 0.4rem;}
.sidebar-sub { font-size: 0.92rem; color: #9CA3AF; margin-bottom: 0.35rem; }
.sidebar-author { font-size: 0.88rem; color: #93A3B8; margin-bottom: 0.8rem; }
.sidebar-divider { border-top: 1px solid #1F2E4D; margin: 0.7rem 0 0.9rem 0; }

section[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    text-align: left;
    border-radius: 10px;
    border: none;
    background-color: #14213F;
    color: #E5E7EB !important;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: none;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #1D2E52;
    color: #FFFFFF !important;
    border: none;
}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background-color: #3B82F6 !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
    background-color: #2563EB !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================
# MODEL LOADING
# =============================================================
@st.cache_resource(show_spinner="Loading Logistic Regression model...")
def load_models():
    models = {}
    model_exists = os.path.exists("coursera_tfidf_logistic_model.pkl")
    vec_exists = os.path.exists("coursera_tfidf_vectorizer.pkl")
    try:
        if model_exists:
            models["model"] = joblib.load("coursera_tfidf_logistic_model.pkl")
            # Safety net: older/newer scikit-learn versions have added/removed internal
            # LogisticRegression attributes over time (e.g. `multi_class`). If the pickled
            # model is missing an attribute the installed sklearn's predict_proba expects,
            # patch in a sensible default rather than crashing.
            for attr, default in [("multi_class", "auto")]:
                if not hasattr(models["model"], attr):
                    setattr(models["model"], attr, default)
        if vec_exists:
            models["vectorizer"] = joblib.load("coursera_tfidf_vectorizer.pkl")
    except Exception:
        pass
    return models

loaded_models = load_models()
MODEL_READY = "model" in loaded_models and "vectorizer" in loaded_models


@st.cache_data
def load_metrics_json():
    """Loads the real, pre-computed evaluation metrics for both models (no live computation)."""
    result = {"baseline": None, "distilbert": None}
    try:
        if os.path.exists("metrics.json"):
            with open("metrics.json") as f:
                result["baseline"] = json.load(f)
    except Exception:
        pass
    try:
        if os.path.exists("distilbert_metrics.json"):
            with open("distilbert_metrics.json") as f:
                result["distilbert"] = json.load(f)
    except Exception:
        pass
    return result



@st.cache_resource(show_spinner="Loading fine-tuned Multilingual DistilBERT (541 MB) — first run may take a minute...")
def load_distilbert():
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(DISTILBERT_REPO)
        model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_REPO)
        model.eval()
        return {"tokenizer": tok, "model": model}
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading local NLLB-200 translation model — first run may take a minute...")
def load_nllb():
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(NLLB_REPO)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_REPO)
        model.eval()
        return {"tokenizer": tok, "model": model}
    except Exception:
        return None


def normalize_label(lbl):
    return str(lbl).strip().upper()


def align_probs(probs, classes):
    """Reorders an (N, k) probability matrix with arbitrary class order into SENT_ORDER columns."""
    probs = np.asarray(probs)
    out = np.zeros((probs.shape[0], len(SENT_ORDER)))
    for j, cls in enumerate(SENT_ORDER):
        if cls in classes:
            out[:, j] = probs[:, classes.index(cls)]
    sums = out.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return out / sums


# =============================================================
# CORE PREDICTION HELPERS
# =============================================================
def batch_predict_lr(texts):
    """Raw TF-IDF + Logistic Regression predictions. Returns (probs, classes)."""
    clean = [t if isinstance(t, str) and t.strip() else " " for t in texts]
    if not MODEL_READY:
        n = len(clean)
        return np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1)), SENT_ORDER
    try:
        vec = loaded_models["vectorizer"].transform(clean)
        model = loaded_models["model"]
        probs = model.predict_proba(vec)
        classes = [normalize_label(c) for c in model.classes_]
        return probs, classes
    except Exception:
        n = len(clean)
        return np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1)), SENT_ORDER


# Known fixed label order from the model's real training run (see distilbert_metrics.json),
# used as a fallback when the checkpoint's config.id2label doesn't contain readable
# sentiment names (e.g. generic "LABEL_0"/"LABEL_1"/"LABEL_2").
DISTILBERT_FALLBACK_LABEL_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def batch_predict_distilbert(texts, bundle, chunk_size=32):
    """Raw fine-tuned multilingual DistilBERT predictions. Returns (probs, classes).
    Processed in chunks to keep memory bounded for larger batches (e.g. CSV uploads)."""
    clean = [t if isinstance(t, str) and t.strip() else " " for t in texts]
    tok = bundle["tokenizer"]
    model = bundle["model"]
    all_probs = []
    classes = None
    for i in range(0, len(clean), chunk_size):
        chunk = clean[i:i + chunk_size]
        inputs = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
        # DistilBERT's forward() doesn't accept `token_type_ids` (unlike full BERT),
        # but some tokenizer configs still generate it — drop it if present.
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()
        if classes is None:
            raw_classes = [normalize_label(model.config.id2label[i]) for i in range(probs.shape[1])]
            # If the checkpoint's labels don't look like real sentiment names (e.g. "LABEL_0"),
            # fall back to the known training label order instead of silently losing this model's vote.
            if not any(c in SENT_ORDER for c in raw_classes) and len(raw_classes) == len(DISTILBERT_FALLBACK_LABEL_ORDER):
                classes = DISTILBERT_FALLBACK_LABEL_ORDER
            else:
                classes = raw_classes
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0), classes


def batch_predict(texts, engine="Combined"):
    """Main entry point used across the app. Returns (labels, confidences) aligned to SENT_ORDER."""
    lr_probs, lr_classes = batch_predict_lr(texts)
    lr_aligned = align_probs(lr_probs, lr_classes)

    if engine == "Logistic Regression" or not TRANSFORMERS_AVAILABLE:
        final = lr_aligned
    else:
        bundle = load_distilbert()
        if bundle is None:
            final = lr_aligned  # honest fallback if the HF model can't be loaded
        else:
            try:
                db_probs, db_classes = batch_predict_distilbert(texts, bundle)
                db_aligned = align_probs(db_probs, db_classes)
                final = db_aligned if engine == "DistilBERT" else (lr_aligned + db_aligned) / 2
            except Exception:
                final = lr_aligned

    idx = final.argmax(axis=1)
    labels = np.array([SENT_ORDER[i] for i in idx])
    conf = final.max(axis=1)
    return labels, conf


def predict_sentiment(text, model_type="Combined"):
    preds, confs = batch_predict([text], engine=model_type)
    return preds[0], float(confs[0])


def get_full_probs(text, engine="Combined"):
    """Returns dict {POSITIVE: p, NEUTRAL: p, NEGATIVE: p} for one text, using the chosen engine."""
    if not text.strip():
        return {"POSITIVE": 1 / 3, "NEUTRAL": 1 / 3, "NEGATIVE": 1 / 3}
    lr_probs, lr_classes = batch_predict_lr([text])
    lr_aligned = align_probs(lr_probs, lr_classes)[0]
    if engine == "Logistic Regression" or not TRANSFORMERS_AVAILABLE:
        final = lr_aligned
    else:
        bundle = load_distilbert()
        if bundle is None:
            final = lr_aligned
        else:
            try:
                db_probs, db_classes = batch_predict_distilbert([text], bundle)
                db_aligned = align_probs(db_probs, db_classes)[0]
                final = db_aligned if engine == "DistilBERT" else (lr_aligned + db_aligned) / 2
            except Exception:
                final = lr_aligned
    return {cls: float(final[i]) for i, cls in enumerate(SENT_ORDER)}


def get_engine_confidences(text):
    """Returns (lr_conf, distilbert_conf, combined_conf, distilbert_available)."""
    lr_probs, lr_classes = batch_predict_lr([text])
    lr_conf = float(lr_probs.max(axis=1)[0])
    db_conf = lr_conf
    db_available = False
    if TRANSFORMERS_AVAILABLE:
        bundle = load_distilbert()
        if bundle is not None:
            try:
                db_probs, _ = batch_predict_distilbert([text], bundle)
                db_conf = float(db_probs.max(axis=1)[0])
                db_available = True
            except Exception:
                pass
    combined_conf = (lr_conf + db_conf) / 2
    return lr_conf, db_conf, combined_conf, db_available


def get_word_contributions(text, pred_label):
    """Returns a DataFrame of Word, Contribution for a linear-model explanation."""
    if not MODEL_READY:
        return pd.DataFrame(columns=["Word", "Contribution"])
    vectorizer = loaded_models["vectorizer"]
    model = loaded_models["model"]
    vec = vectorizer.transform([text])
    classes = [normalize_label(c) for c in model.classes_]
    if pred_label not in classes:
        return pd.DataFrame(columns=["Word", "Contribution"])
    class_idx = classes.index(pred_label)
    feature_names = np.array(vectorizer.get_feature_names_out())
    row = vec.toarray()[0]
    nonzero = np.nonzero(row)[0]
    contributions = model.coef_[class_idx][nonzero] * row[nonzero]
    words = feature_names[nonzero]
    contrib_df = pd.DataFrame({"Word": words, "Contribution": contributions})
    contrib_df = contrib_df.reindex(contrib_df["Contribution"].abs().sort_values(ascending=False).index)
    return contrib_df.reset_index(drop=True)


def generate_key_insight(sentiment, probs):
    label = sentiment.lower()
    others = {k: v for k, v in probs.items() if k != sentiment}
    spread = max(others.values()) if others else 0
    top = probs.get(sentiment, 0)
    if sentiment == "NEUTRAL" and spread > 0.15:
        return f"The model predicts a {label} sentiment. The feedback contains a mixture of positive and negative observations."
    elif top > 0.75:
        return f"The model predicts a {label} sentiment with high confidence."
    else:
        return f"The model predicts a {label} sentiment, though the confidence is moderate."


LANG_NAMES = {
    "en": "English", "ur": "Urdu", "zh-cn": "Chinese", "zh": "Chinese", "ko": "Korean",
    "ru": "Russian", "es": "Spanish", "fr": "French", "de": "German", "hi": "Hindi",
    "ar": "Arabic", "pt": "Portuguese", "ja": "Japanese", "it": "Italian", "tr": "Turkish",
    "id": "Indonesian", "vi": "Vietnamese", "bn": "Bengali", "fa": "Persian", "nl": "Dutch",
}

def detect_language(text):
    try:
        code = detect(text)
        return code, LANG_NAMES.get(code, code.upper())
    except Exception:
        return "unknown", "Unknown"


def translate_to_english(text, lang_code):
    """Best-effort translation: local NLLB-200 first (fully offline), then Google Translate,
    then the original text. This function never raises."""
    if not text or not text.strip():
        return text
    if lang_code in ("en", "unknown"):
        return text

    # 1) Local NLLB-200 model — matches the project's fully-offline translation design.
    if TRANSFORMERS_AVAILABLE and lang_code in NLLB_LANG_CODES:
        bundle = load_nllb()
        if bundle is not None:
            try:
                tok = bundle["tokenizer"]
                model = bundle["model"]
                src_code = NLLB_LANG_CODES[lang_code]
                tok.src_lang = src_code
                encoded = tok(text, return_tensors="pt", truncation=True, max_length=200)
                try:
                    forced_bos = tok.lang_code_to_id["eng_Latn"]
                except AttributeError:
                    forced_bos = tok.convert_tokens_to_ids("eng_Latn")
                with torch.no_grad():
                    generated = model.generate(**encoded, forced_bos_token_id=forced_bos, max_new_tokens=200)
                return tok.batch_decode(generated, skip_special_tokens=True)[0]
            except Exception:
                pass

    # 2) Google Translate fallback (needs outbound internet — normally available in production).
    if TRANSLATOR_AVAILABLE:
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception:
            pass

    # 3) Last resort — never raise, just show the original text.
    return text


@st.cache_data(show_spinner=False)
def cached_translate(text, lang_code):
    return translate_to_english(text, lang_code)


# =============================================================
# ASPECT KEYWORDS (15 aspects)
# =============================================================
ASPECT_KEYWORDS = {
    "Course Content": ["content", "course content", "material", "materials", "topics",
                        "topic", "lessons", "lesson", "curriculum", "concepts", "concept",
                        "theory", "information", "syllabus"],
    "Instructor": ["instructor", "teacher", "professor", "lecturer", "mentor", "teaching",
                   "teach", "taught", "explained", "explanation", "lecture", "lectures"],
    "Assignments": ["assignment", "assignments", "homework", "exercise", "exercises",
                     "project", "projects", "task", "tasks"],
    "Quizzes & Assessments": ["quiz", "quizzes", "test", "tests", "exam", "exams",
                               "assessment", "assessments", "grading", "grade", "grades"],
    "Difficulty": ["difficult", "difficulty", "easy", "easier", "hard", "challenging",
                   "challenge", "complex", "complicated", "simple", "beginner", "advanced"],
    "Learning Experience": ["learn", "learned", "learning", "experience", "understand",
                             "understanding", "helpful", "useful", "skill", "skills",
                             "improved", "improve"],
    "Course Structure": ["structure", "structured", "organized", "organised", "organization",
                          "sequence", "order", "module", "modules", "section", "sections"],
    "Platform": ["platform", "website", "app", "application", "interface", "portal",
                 "system", "dashboard", "navigation"],
    "Video & Audio": ["video", "videos", "audio", "sound", "recording", "recordings",
                       "playback", "visuals", "voice"],
    "Certificates": ["certificate", "certificates", "certification", "credential",
                      "credentials", "diploma"],
    "Duration": ["duration", "length", "time", "hours", "hour", "week", "weeks", "short", "long", "pace", "pacing"],
    "Value": ["value", "worth", "price", "cost", "affordable", "expensive", "cheap",
              "money"],
    "Practical Application": ["practical", "application", "applications", "real-world",
                               "real world", "hands-on", "hands on", "apply", "applied",
                               "implementation", "practice"],
    "Relevance": ["relevant", "relevance", "up-to-date", "up to date", "outdated",
                  "current", "industry"],
    "Overall Experience": ["overall", "experience", "satisfied", "satisfaction", "enjoyed",
                            "enjoy", "recommend", "recommended", "great course", "amazing"],
}
ASPECT_NAMES = list(ASPECT_KEYWORDS.keys())

ASPECT_ICONS = {
    "Course Content": "📖", "Instructor": "👤", "Assignments": "📋",
    "Quizzes & Assessments": "📝", "Difficulty": "🎯", "Learning Experience": "🎓",
    "Course Structure": "🏗️", "Platform": "💻", "Video & Audio": "🎥",
    "Certificates": "🏆", "Duration": "⏱️", "Value": "💰",
    "Practical Application": "🔧", "Relevance": "🔗", "Overall Experience": "⭐",
}

# Actionable suggestions shown when an aspect comes back Negative or Neutral.
ASPECT_SUGGESTIONS = {
    "Course Content": {
        "NEGATIVE": "Review and refresh the course material — add more examples and simplify dense sections.",
        "NEUTRAL": "Consider adding supplementary examples or reading material to deepen the content.",
    },
    "Instructor": {
        "NEGATIVE": "Provide the instructor with feedback on pacing and clarity of explanations.",
        "NEUTRAL": "Encourage more interactive Q&A or office hours to strengthen the instructor connection.",
    },
    "Assignments": {
        "NEGATIVE": "Re-calibrate assignment difficulty and provide clearer instructions or worked examples.",
        "NEUTRAL": "Add a few practice assignments with varying difficulty to build confidence gradually.",
    },
    "Quizzes & Assessments": {
        "NEGATIVE": "Review quiz difficulty and grading transparency; consider adding practice quizzes.",
        "NEUTRAL": "Offer optional practice assessments so students can self-check before graded ones.",
    },
    "Difficulty": {
        "NEGATIVE": "Add a beginner-friendly on-ramp or prerequisite guide to ease the learning curve.",
        "NEUTRAL": "Offer optional advanced and beginner tracks so pacing fits more skill levels.",
    },
    "Learning Experience": {
        "NEGATIVE": "Gather specific feedback on what hindered learning and iterate on the weakest module.",
        "NEUTRAL": "Add more interactive or hands-on elements to make learning more engaging.",
    },
    "Course Structure": {
        "NEGATIVE": "Reorganize modules into a clearer, more logical sequence with defined milestones.",
        "NEUTRAL": "Add a visual course roadmap so students can see how modules connect.",
    },
    "Platform": {
        "NEGATIVE": "Investigate platform usability issues — navigation, load times, or broken links.",
        "NEUTRAL": "Collect specific UX feedback to identify small friction points in the platform.",
    },
    "Video & Audio": {
        "NEGATIVE": "Improve recording audio quality and consider re-recording unclear video segments.",
        "NEUTRAL": "Add captions or transcripts to improve accessibility and clarity.",
    },
    "Certificates": {
        "NEGATIVE": "Clarify certificate requirements and streamline the issuance process.",
        "NEUTRAL": "Highlight the certificate's value and how it can be used professionally.",
    },
    "Duration": {
        "NEGATIVE": "Reassess course length — consider splitting into shorter, focused modules.",
        "NEUTRAL": "Offer a suggested pacing guide so students can plan their time better.",
    },
    "Value": {
        "NEGATIVE": "Reassess pricing relative to content depth, or add more bonus resources.",
        "NEUTRAL": "Communicate the course's value proposition more clearly upfront.",
    },
    "Practical Application": {
        "NEGATIVE": "Add more real-world projects or case studies to bridge theory and practice.",
        "NEUTRAL": "Include at least one capstone-style project to reinforce practical skills.",
    },
    "Relevance": {
        "NEGATIVE": "Update course content to reflect current industry practices and tools.",
        "NEUTRAL": "Periodically refresh examples to keep the material feeling current.",
    },
    "Overall Experience": {
        "NEGATIVE": "Conduct a broader review of the course to identify the biggest pain points.",
        "NEUTRAL": "Small, consistent improvements across modules could lift the overall experience.",
    },
}


def get_aspect_suggestion(aspect, sentiment):
    return ASPECT_SUGGESTIONS.get(aspect, {}).get(sentiment)


def split_sentences(text):
    text = str(text)
    parts = re.split(r"(?<=[.!?۔])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_aspect_mentions(text):
    found = {}
    sentences = split_sentences(text) or [text]
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for sent in sentences:
            low = sent.lower()
            if any(kw in low for kw in keywords):
                found[aspect] = sent
                break
    return found


# =============================================================
# COLUMN DETECTION
# =============================================================
REVIEW_COL_CANDIDATES = ["feedback", "review", "reviews", "text", "comment", "comments",
                          "description", "student_feedback", "student review", "student_review"]
COURSE_COL_CANDIDATES = ["course", "courseid", "course_id", "coursename", "course_name",
                          "course title", "coursetitle", "course_title", "course name"]

def detect_review_column(df):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in REVIEW_COL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    obj_cols = list(df.select_dtypes(include="object").columns)
    if not obj_cols:
        return df.columns[0]
    return max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())


def detect_course_column(df, review_col):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in COURSE_COL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    for c in df.columns:
        if c != review_col and df[c].dtype == object:
            return c
    return None


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def majority_sentiment(pos, neu, neg):
    counts = {"POSITIVE": pos, "NEUTRAL": neu, "NEGATIVE": neg}
    return max(counts, key=counts.get)


# =============================================================
# SHARED RENDER HELPERS
# =============================================================
def render_hero():
    title_col, badge_col = st.columns([3, 1])
    with title_col:
        st.markdown('<p class="main-header">🎓 Course Feedback Sentiment Analysis</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Analyze student feedback, discover key aspects, and understand what drives sentiment — powered by AI.</p>', unsafe_allow_html=True)
    with badge_col:
        st.markdown(
            '<div class="info-badge-card">'
            '<div class="badge-title">🎓 AI Education Analytics</div>'
            '<div class="badge-body">Sentiment • Aspects • Explainability</div>'
            '</div>', unsafe_allow_html=True)
    st.markdown("---")


def render_footer():
    st.markdown('<div class="app-footer">🎓 Course Feedback Sentiment Analysis • AI-Powered Education Analytics • Created By Afsah Arshad</div>', unsafe_allow_html=True)


def render_aspect_cards(mentions_dict):
    """mentions_dict: {aspect: sentence}. Renders a 3-col card grid with sentiment + progress bar."""
    if not mentions_dict:
        st.info("No specific course aspects (Content, Instructor, Difficulty, etc.) were detected in this review.")
        return None
    items = list(mentions_dict.items())
    preds, confs = batch_predict([s for _, s in items], engine="Logistic Regression")
    cols_per_row = 3
    for i in range(0, len(items), cols_per_row):
        row_items = list(zip(items[i:i + cols_per_row], preds[i:i + cols_per_row], confs[i:i + cols_per_row]))
        cols = st.columns(cols_per_row)
        for col, ((aspect, sentence), pred, conf) in zip(cols, row_items):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{ASPECT_ICONS.get(aspect, '🔹')} {aspect}**")
                    st.markdown(f"{SENT_DOT[pred]} {pred.capitalize()}")
                    st.progress(float(conf))
                    st.caption(f"Confidence: {conf:.4f}")
                    suggestion = get_aspect_suggestion(aspect, pred)
                    if suggestion:
                        st.markdown(f"💡 *Suggestion: {suggestion}*")
    return pd.DataFrame({
        "Aspect": [a for a, _ in items],
        "Sentiment": preds,
        "Confidence": np.round(confs, 4),
        "Sentence": [s for _, s in items],
    })


def generate_explanation_summary(contrib_df, pred_label, top_n=3):
    label = pred_label.capitalize()
    if contrib_df.empty:
        return f"The model predicted **{label}**, but none of the words in this review were recognized by its vocabulary."
    positive_words = contrib_df[contrib_df["Contribution"] > 0].head(top_n)["Word"].tolist()
    negative_words = contrib_df[contrib_df["Contribution"] < 0].head(top_n)["Word"].tolist()
    parts = [f"This review was classified as **{label}**"]
    if positive_words:
        quoted = ", ".join(f"“{w}”" for w in positive_words)
        parts.append(f"mainly because of words like {quoted}, which pushed the score toward {label.lower()}")
    if negative_words:
        quoted = ", ".join(f"“{w}”" for w in negative_words)
        connector = ", while" if positive_words else "but"
        parts.append(f"{connector} words like {quoted} pulled it in the opposite direction")
    return " ".join(parts) + "."


def render_highlighted_text(text, contrib_df, max_words=20):
    """Returns HTML with single-word contributors color-highlighted (green=toward the
    predicted class, red=away from it). Multi-word n-gram features are skipped."""
    word_map = {}
    for _, row in contrib_df.head(max_words).iterrows():
        w = row["Word"]
        if " " not in w:
            word_map[w.lower()] = row["Contribution"]
    if not word_map:
        return html_lib.escape(text)
    max_abs = max(abs(v) for v in word_map.values()) or 1
    tokens = re.findall(r"\w+|[^\w\s]|\s+", text, flags=re.UNICODE)
    parts = []
    for tok in tokens:
        key = tok.lower()
        if key in word_map:
            val = word_map[key]
            intensity = min(abs(val) / max_abs, 1.0)
            if val > 0:
                bg = f"rgba(34,197,94,{0.15 + 0.55 * intensity:.2f})"
            else:
                bg = f"rgba(220,38,38,{0.15 + 0.55 * intensity:.2f})"
            parts.append(f'<span style="background:{bg};border-radius:4px;padding:0 2px;">{html_lib.escape(tok)}</span>')
        else:
            parts.append(html_lib.escape(tok))
    return "".join(parts)


def render_explainable_section(text, pred_label, show_header=True):
    """Rich SHAP-style explanation: plain-language summary, highlighted review text,
    top-contributing-words chart, and a full contribution table. Returns the figure (or None)."""
    if show_header:
        st.markdown('<p class="section-header">💡 Explainable AI (SHAP)</p>', unsafe_allow_html=True)
        st.caption("Words that influenced the prediction (based on the Logistic Regression model's TF-IDF weights)")
    contrib_df = get_word_contributions(text, pred_label)
    if contrib_df.empty:
        st.info("None of the words in this review were recognized by the model's vocabulary.")
        return None

    st.markdown(f'<div class="insight-box">🧠 {generate_explanation_summary(contrib_df, pred_label)}</div>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown("**Highlighted review** — 🟩 pushed toward the prediction, 🟥 pulled away from it")
    st.markdown(f'<div class="insight-box">{render_highlighted_text(text, contrib_df)}</div>', unsafe_allow_html=True)
    st.markdown("")

    top = contrib_df.head(10).sort_values("Contribution")
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(top))))
    colors = ["#22C55E" if v > 0 else "#DC2626" for v in top["Contribution"]]
    ax.barh(top["Word"], top["Contribution"], color=colors)
    ax.set_title("Top Contributing Words")
    ax.axvline(0, color="#9CA3AF", linewidth=0.8)
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("📋 Full word-contribution table"):
        st.dataframe(contrib_df.rename(columns={"Contribution": f"Contribution to '{pred_label}'"}),
                     use_container_width=True, hide_index=True)

    return fig


# =============================================================
# ONE-TIME TOAST
# =============================================================
st.toast("Turning student feedback into actionable insights", icon="🎓")

# =============================================================
# SIDEBAR NAVIGATION
# =============================================================
NAV_ITEMS = [
    ("Single Review Analysis", "💬"),
    ("CSV Analysis", "📄"),
    ("Aspect Analysis", "🔗"),
    ("Explainable AI (SHAP)", "💡"),
    ("Model Performance", "📈"),
    ("About", "ℹ️"),
]

if "nav" not in st.session_state:
    st.session_state.nav = "Single Review Analysis"

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎓</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Course Feedback<br>Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">AI-Powered Insights for Better Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-author">Created By Afsah Arshad</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    for name, icon in NAV_ITEMS:
        is_active = st.session_state.nav == name
        if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.nav = name
            st.rerun()

app_mode = st.session_state.nav

# =============================================================
# SAMPLE FEEDBACKS & MODEL DESCRIPTIONS (Single Review)
# =============================================================
SAMPLE_FEEDBACKS = {
    "Choose a feedback...": "",
    "Excellent course": "This course is excellent and very easy to follow. I learned a lot.",
    "Very helpful": "The explanations are clear and the practice was very useful.",
    "Good but difficult": "The instructor was good and the content was useful, but some assignments were difficult.",
    "Average experience": "The course was okay, but some topics needed more examples.",
    "Poor experience": "The lessons were confusing and the exercises were too hard.",
    "Excellent instructor": "The instructor explained everything clearly and the lectures were engaging.",
    "Difficult assignments": "The assignments were too difficult and took much longer than expected.",
    "Mixed feelings": "The course content was great, but the platform kept crashing and the certificate never arrived.",
    "Very negative experience": "This was a waste of time and money. The instructor was unclear and the assignments made no sense.",
    "Platform issues": "The video kept buffering and the platform navigation was confusing, but the material itself was solid.",
    "Great value for money": "For the price, this course offers excellent value — the content is worth far more than what I paid.",
    "Outdated content": "The material feels outdated and doesn't reflect current industry practices, though the instructor was engaging.",
    "Multilingual (Urdu)": "کورس بہت اچھا تھا لیکن اسائنمنٹس تھوڑے مشکل تھے۔",
    "Multilingual (Chinese)": "这门课程非常棒，内容清晰，而且我学到了很多东西。",
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": ("Logistic Regression",
        "Fast, lightweight linear model trained on TF-IDF features. Ideal for quick testing."),
    "DistilBERT": ("DistilBERT",
        "Fine-tuned multilingual transformer (Afsah-2027/coursera-multilingual-distilbert), loaded from Hugging Face Hub. First use in a session downloads ~541 MB, so it may take a moment; if it can't load, results fall back to Logistic Regression."),
    "Combined": ("Combined (Recommended)",
        "Uses both Logistic Regression and multilingual DistilBERT. Averages the probability outputs of the two trained classifiers and is recommended for the final presentation."),
}

def _apply_sample_feedback():
    choice = st.session_state.get("sample_choice")
    if choice and SAMPLE_FEEDBACKS.get(choice):
        st.session_state["single_review_text"] = SAMPLE_FEEDBACKS[choice]

def _apply_sample_feedback_to(choice_key, text_key):
    def _cb():
        choice = st.session_state.get(choice_key)
        if choice and SAMPLE_FEEDBACKS.get(choice):
            st.session_state[text_key] = SAMPLE_FEEDBACKS[choice]
    return _cb

# =============================================================
# VIEW: SINGLE REVIEW ANALYSIS
# =============================================================
if app_mode == "Single Review Analysis":
    render_hero()
    st.markdown('<p class="section-header">💬 Single Review Analysis</p>', unsafe_allow_html=True)
    st.caption("Analyze one course review using the trained AI models.")

    st.markdown('<p class="section-header" style="font-size:1.3rem !important;">💬 Single Review</p>', unsafe_allow_html=True)

    sample_col, text_col = st.columns(2)
    with sample_col:
        st.selectbox("Select a sample feedback (optional)", list(SAMPLE_FEEDBACKS.keys()),
                     key="sample_choice", on_change=_apply_sample_feedback)
    with text_col:
        user_review = st.text_area("Type your own review", max_chars=1000,
                                    key="single_review_text", placeholder="Type or paste feedback here...",
                                    value=st.session_state.get("single_review_text",
                                          "The instructor was good and the content was useful, but some assignments were difficult."))

    st.markdown("")
    st.markdown("### Select Model")
    radio_col, desc_col, btn_col = st.columns([1, 2, 1])
    with radio_col:
        selected_model = st.radio(
            "Model Selection", ["Logistic Regression", "DistilBERT", "Combined"],
            label_visibility="collapsed", key="single_model_choice", index=2
        )
    with desc_col:
        title, body = MODEL_DESCRIPTIONS[selected_model]
        st.markdown(
            f'<div class="model-desc-box">'
            f'<div class="model-desc-title">🔵 {title}</div>'
            f'<div class="model-desc-body">{body}</div>'
            f'</div>', unsafe_allow_html=True)
    with btn_col:
        st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
        analyze_clicked = st.button("▶ Analyze Review", type="primary", use_container_width=True)

    if analyze_clicked:
        if user_review.strip() == "":
            st.warning("Please enter some review text first.")
        else:
            with st.spinner("Analyzing sentiment..."):
                lang_code, lang_name = detect_language(user_review)
                translated = translate_to_english(user_review, lang_code)
                probs = get_full_probs(translated, engine=selected_model)
                sentiment = max(probs, key=probs.get)
                conf = probs[sentiment]

                st.markdown('<p class="section-header">📋 Single Review Analysis</p>', unsafe_allow_html=True)

                with st.container(border=True):
                    left, right = st.columns([1, 1])
                    with left:
                        st.markdown("### Sentiment Prediction")
                        st.markdown(f"## {SENT_EMOJI[sentiment]} {sentiment.capitalize()}")
                        st.markdown(f"Confidence: **{conf:.2f}**")
                    with right:
                        for cls in SENT_ORDER:
                            p = probs[cls]
                            st.markdown(f"{cls.capitalize()}&nbsp;&nbsp;&nbsp;**{p*100:.0f}%**", unsafe_allow_html=True)
                            st.progress(float(p))

                st.markdown("")
                with st.container(border=True):
                    st.markdown("#### 💡 Key Insight")
                    st.write(generate_key_insight(sentiment, probs))

                if lang_code != "en":
                    st.info(f"🌍 **Detected language:** {lang_name}  \n**English Translation:** {translated}")

                st.markdown('<p class="section-header">🔗 Aspect Analysis</p>', unsafe_allow_html=True)
                st.caption("Key aspects detected in the feedback and their sentiment")
                mentions = extract_aspect_mentions(translated)
                render_aspect_cards(mentions)

                shap_fig = render_explainable_section(translated, sentiment)

                st.markdown('<p class="section-header">📊 Model Confidence</p>', unsafe_allow_html=True)
                lr_conf, distil_conf, combined_conf, distil_available = get_engine_confidences(translated)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Logistic Regression**")
                    st.markdown(f"## {lr_conf:.2f}")
                    if selected_model == "Logistic Regression":
                        st.success("↑ Best")
                with c2:
                    st.markdown("**DistilBERT**")
                    st.markdown(f"## {distil_conf:.2f}")
                    if selected_model == "DistilBERT":
                        st.success("↑ Best")
                with c3:
                    st.markdown("**Combined**")
                    st.markdown(f"## {combined_conf:.2f}")
                    if selected_model == "Combined":
                        st.success("↑ Best")
                st.caption("Combined averages the Logistic Regression and fine-tuned Multilingual DistilBERT probability outputs.")

                st.markdown('<p class="section-header">⚡ Quick Actions</p>', unsafe_allow_html=True)
                qa1, qa2, qa3 = st.columns(3)
                report_txt = (
                    f"Review: {user_review}\n\nLanguage: {lang_name}\nTranslation: {translated}\n\n"
                    f"Sentiment: {sentiment} (confidence {conf:.2f})\n\n"
                    f"Model Confidence -> Logistic Regression: {lr_conf:.2f}, DistilBERT: {distil_conf:.2f}, Combined: {combined_conf:.2f}\n"
                )
                with qa1:
                    st.download_button("📄 Download Detailed Report", report_txt.encode("utf-8"),
                                        "detailed_report.txt", "text/plain", use_container_width=True)
                with qa2:
                    result_df = pd.DataFrame([{"Review": user_review, "Sentiment": sentiment, "Confidence": round(conf, 4)}])
                    st.download_button("⬇️ Download Results", df_to_csv_bytes(result_df),
                                        "single_review_result.csv", "text/csv", use_container_width=True)
                with qa3:
                    if shap_fig is not None:
                        st.download_button("🖼️ Download Explainable AI Graph", fig_to_png_bytes(shap_fig),
                                            "explainable_ai_graph.png", "image/png", use_container_width=True)
                    else:
                        st.button("🖼️ Download Explainable AI Graph", disabled=True, use_container_width=True)

                st.info("ℹ️ SHAP shows which words and tokens contributed to the model's prediction, helping you understand the reasoning behind each sentiment.")

    render_footer()

# =============================================================
# VIEW: CSV ANALYSIS
# =============================================================
elif app_mode == "CSV Analysis":
    render_hero()
    st.markdown('<p class="main-header">Batch CSV Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a CSV file containing course reviews to analyze trends in bulk.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    st.info("💡 Your CSV can contain additional columns such as CourseId, rating, date, or other metadata.")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read this CSV: {e}")
            df = None

        if df is not None and not df.empty:
            review_col = detect_review_column(df)
            course_col = detect_course_column(df, review_col)

            st.success(f"Detected review column: **{review_col}**")

            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", len(df))
            m2.metric("Columns", len(df.columns))
            m3.metric("Review Column", review_col)

            st.markdown("### Select Model for Analysis")
            selected_model = st.radio(
                "Model", ["Logistic Regression", "DistilBERT", "Combined"],
                horizontal=True, label_visibility="collapsed", key="csv_model", index=2
            )

            if st.button("🔍 Analyze CSV", type="primary", use_container_width=True):
                with st.spinner("Analyzing reviews..."):
                    work_df = df.copy()
                    texts = work_df[review_col].astype(str).tolist()

                    preds, confs = batch_predict(texts, engine=selected_model)
                    work_df["Sentiment"] = preds
                    work_df["Confidence"] = np.round(confs, 4)

                    st.session_state["csv_result"] = {
                        "df": work_df,
                        "review_col": review_col,
                        "course_col": course_col,
                    }
                st.success(f"Analysis completed for {len(work_df)} reviews.")

    result = st.session_state.get("csv_result")
    if result is not None:
        work_df = result["df"]
        review_col = result["review_col"]
        course_col = result["course_col"]

        st.markdown('<p class="section-header">📊 Overall Sentiment Analysis</p>', unsafe_allow_html=True)
        counts = work_df["Sentiment"].value_counts().reindex(SENT_ORDER, fill_value=0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", int(counts["POSITIVE"]))
        c2.metric("Neutral", int(counts["NEUTRAL"]))
        c3.metric("Negative", int(counts["NEGATIVE"]))

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(["Positive", "Neutral", "Negative"], counts.values,
                color=[SENT_COLORS[s] for s in SENT_ORDER])
        ax1.set_title("Overall Sentiment Distribution")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        st.markdown('<p class="section-header">🔗 Aspect-Based Sentiment Analysis</p>', unsafe_allow_html=True)
        st.caption("Shows which course-related aspects are discussed in the feedback and whether students feel positively, neutrally, or negatively about them.")

        @st.cache_data(show_spinner=False)
        def build_aspect_table(texts, courses):
            rows = []
            for i, text in enumerate(texts):
                mentions = extract_aspect_mentions(text)
                for aspect, sentence in mentions.items():
                    rows.append({
                        "RowIndex": i,
                        "Course": courses[i] if courses is not None else "N/A",
                        "Aspect": aspect,
                        "Review": sentence,
                        "FullReview": text,
                    })
            adf = pd.DataFrame(rows)
            if not adf.empty:
                preds, confs = batch_predict(adf["Review"].tolist(), engine="Logistic Regression")
                adf["Sentiment"] = preds
                adf["Confidence"] = np.round(confs, 4)
            return adf

        courses_list = work_df[course_col].astype(str).tolist() if course_col else None
        texts_ = work_df[review_col].astype(str).tolist()
        aspect_df = build_aspect_table(tuple(texts_), tuple(courses_list) if courses_list else None)

        if aspect_df.empty:
            st.warning("No course-related aspects were detected in this dataset.")
        else:
            overall_pivot = aspect_df.groupby(["Aspect", "Sentiment"]).size().unstack(fill_value=0)
            for s in SENT_ORDER:
                if s not in overall_pivot.columns:
                    overall_pivot[s] = 0
            overall_pivot = overall_pivot.reindex(ASPECT_NAMES).fillna(0).astype(int)
            overall_pivot = overall_pivot[SENT_ORDER]

            overall_summary = overall_pivot.copy()
            overall_summary["Reviews"] = overall_summary.sum(axis=1)
            overall_summary["Overall Sentiment"] = overall_summary.apply(
                lambda r: majority_sentiment(r["POSITIVE"], r["NEUTRAL"], r["NEGATIVE"]), axis=1)
            overall_summary = overall_summary.reset_index().rename(
                columns={"POSITIVE": "Positive", "NEUTRAL": "Neutral", "NEGATIVE": "Negative"})
            overall_summary = overall_summary[overall_summary["Reviews"] > 0]
            overall_summary = overall_summary[["Aspect", "Reviews", "Positive", "Neutral", "Negative", "Overall Sentiment"]]

            st.dataframe(overall_summary, use_container_width=True, hide_index=True)

            plot_pivot = overall_pivot.loc[overall_summary["Aspect"]]
            x = np.arange(len(plot_pivot))
            width = 0.25
            fig2, ax2 = plt.subplots(figsize=(11, 5))
            ax2.bar(x - width, plot_pivot["POSITIVE"], width, label="Positive", color=SENT_COLORS["POSITIVE"])
            ax2.bar(x, plot_pivot["NEUTRAL"], width, label="Neutral", color=SENT_COLORS["NEUTRAL"])
            ax2.bar(x + width, plot_pivot["NEGATIVE"], width, label="Negative", color=SENT_COLORS["NEGATIVE"])
            ax2.set_xticks(x)
            ax2.set_xticklabels(plot_pivot.index, rotation=45, ha="right")
            ax2.set_xlabel("Aspect")
            ax2.set_ylabel("Count")
            ax2.set_title("Aspect Sentiment Analysis")
            ax2.legend(title="Sentiment")
            fig2.tight_layout()
            st.pyplot(fig2)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("⬇️ Download Overall Aspect Analysis", df_to_csv_bytes(overall_summary),
                                    "overall_aspect_analysis.csv", "text/csv", use_container_width=True)
            with dl2:
                st.download_button("🖼️ Download Aspect Graph", fig_to_png_bytes(fig2),
                                    "aspect_graph.png", "image/png", use_container_width=True)

            st.markdown('<p class="section-header">🎓 Course-Wise Aspect Analysis</p>', unsafe_allow_html=True)
            st.caption("Aspect sentiment is grouped using the Course Name or Course ID available in the uploaded CSV.")

            if not course_col:
                st.info("No course/course-ID column was detected in this CSV, so course-wise breakdown isn't available. Add a column like `Course`, `CourseId` or `Course Name` to enable it.")
            else:
                course_pivot = aspect_df.groupby(["Course", "Aspect", "Sentiment"]).size().unstack(fill_value=0)
                for s in SENT_ORDER:
                    if s not in course_pivot.columns:
                        course_pivot[s] = 0
                course_pivot = course_pivot[SENT_ORDER].reset_index()
                course_pivot["Reviews"] = course_pivot[SENT_ORDER].sum(axis=1)
                course_pivot["Overall Sentiment"] = course_pivot.apply(
                    lambda r: majority_sentiment(r["POSITIVE"], r["NEUTRAL"], r["NEGATIVE"]), axis=1)
                course_summary = course_pivot.rename(
                    columns={"POSITIVE": "Positive", "NEUTRAL": "Neutral", "NEGATIVE": "Negative"})
                course_summary = course_summary[["Course", "Aspect", "Reviews", "Positive", "Neutral", "Negative", "Overall Sentiment"]]

                st.download_button("⬇️ Download All Course-Wise Aspect Analysis", df_to_csv_bytes(course_summary),
                                    "course_wise_aspect_analysis.csv", "text/csv", use_container_width=True)

                course_options = sorted(course_summary["Course"].unique().tolist())
                selected_course = st.selectbox("Select Course", course_options)

                course_view = course_summary[course_summary["Course"] == selected_course]
                st.dataframe(course_view, use_container_width=True, hide_index=True)

                fig3, ax3 = plt.subplots(figsize=(10, 4.5))
                bar_colors = [SENT_COLORS[s] for s in course_view["Overall Sentiment"]]
                ax3.bar(course_view["Aspect"], course_view["Reviews"], color=bar_colors)
                ax3.set_title(f"Aspect Sentiment for {selected_course}")
                ax3.set_xlabel("Aspect")
                ax3.set_ylabel("Count")
                plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
                fig3.tight_layout()
                st.pyplot(fig3)

                dl3, dl4 = st.columns(2)
                with dl3:
                    st.download_button("⬇️ Download Selected Course Analysis", df_to_csv_bytes(course_view),
                                        f"{selected_course}_aspect_analysis.csv", "text/csv", use_container_width=True)
                with dl4:
                    st.download_button("🖼️ Download Course Graph", fig_to_png_bytes(fig3),
                                        f"{selected_course}_aspect_graph.png", "image/png", use_container_width=True)

            st.markdown('<p class="section-header">📋 Detailed Aspect Results</p>', unsafe_allow_html=True)
            detailed = aspect_df[["Course", "Aspect", "Sentiment", "Confidence", "Review"]] if course_col else aspect_df[["Aspect", "Sentiment", "Confidence", "Review"]]
            st.dataframe(detailed, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download Detailed Aspect Results", df_to_csv_bytes(detailed),
                                "detailed_aspect_results.csv", "text/csv", use_container_width=True)

        st.markdown('<p class="section-header">📋 Analysis Results</p>', unsafe_allow_html=True)
        TRANSLATE_LIMIT = 200
        preview_df = work_df.head(TRANSLATE_LIMIT).copy()

        lang_labels, translations = [], []
        for text in preview_df[review_col].astype(str).tolist():
            code, name = detect_language(text)
            lang_labels.append(name)
            translations.append(translate_to_english(text, code))

        preview_df["Language"] = lang_labels
        preview_df["English Translation"] = translations

        display_cols = ([course_col] if course_col else []) + [review_col, "Language", "English Translation", "Sentiment", "Confidence"]
        st.dataframe(preview_df[display_cols], use_container_width=True, hide_index=True)
        if len(work_df) > TRANSLATE_LIMIT:
            st.caption(f"Showing translations for the first {TRANSLATE_LIMIT} of {len(work_df)} rows to keep things fast. Full sentiment results for all rows are in the download below.")

        full_export_cols = ([course_col] if course_col else []) + [review_col, "Sentiment", "Confidence"]
        st.download_button("⬇️ Download Results", df_to_csv_bytes(work_df[full_export_cols]),
                            "analysis_results.csv", "text/csv", use_container_width=True)

    render_footer()

# =============================================================
# VIEW: ASPECT ANALYSIS (single review, standalone)
# =============================================================
elif app_mode == "Aspect Analysis":
    render_hero()
    st.markdown('<p class="main-header">🔗 Aspect Analysis</p>', unsafe_allow_html=True)
    st.caption("Identify important course-related aspects and their sentiment.")

    sample_col2, text_col2 = st.columns(2)
    with sample_col2:
        st.selectbox("Select a sample feedback (optional)", list(SAMPLE_FEEDBACKS.keys()),
                     key="aspect_sample_choice", on_change=_apply_sample_feedback_to("aspect_sample_choice", "aspect_text"))
    with text_col2:
        text = st.text_area("Enter course feedback", key="aspect_text",
                             placeholder="Example: The instructor was excellent, but the assignments were difficult.")
    if st.button("🔎 Analyze Aspects", type="primary"):
        if not text.strip():
            st.warning("Please enter some review text first.")
        else:
            lang_code, lang_name = detect_language(text)
            translated = translate_to_english(text, lang_code)
            mentions = extract_aspect_mentions(translated)
            if mentions:
                st.success(f"Detected {len(mentions)} aspect(s) in this review.")
            render_aspect_cards(mentions)

    render_footer()

# =============================================================
# VIEW: EXPLAINABLE AI (SHAP-style feature contributions)
# =============================================================
elif app_mode == "Explainable AI (SHAP)":
    render_hero()
    st.markdown('<p class="main-header">💡 Explainable AI (SHAP)</p>', unsafe_allow_html=True)
    st.caption("Understand which words influence the sentiment prediction (based on the Logistic Regression model's TF-IDF weights).")

    sample_col3, text_col3 = st.columns(2)
    with sample_col3:
        st.selectbox("Select a sample feedback (optional)", list(SAMPLE_FEEDBACKS.keys()),
                     key="shap_sample_choice", on_change=_apply_sample_feedback_to("shap_sample_choice", "shap_text"))
    with text_col3:
        text = st.text_area("Enter course feedback", key="shap_text",
                             placeholder="Enter a review to generate word-level explanations.")
    if st.button("🎨 Generate Explanation", type="primary"):
        if not text.strip():
            st.warning("Please enter some review text first.")
        elif not MODEL_READY:
            st.error("The sentiment model isn't loaded, so an explanation can't be generated right now.")
        else:
            pred_label, conf = predict_sentiment(text, model_type="Logistic Regression")
            st.markdown(f"### Predicted Sentiment: **{pred_label}**")
            render_explainable_section(text, pred_label, show_header=False)
            st.info("ℹ️ SHAP shows which words and tokens contributed to the model's prediction, helping you understand the reasoning behind each sentiment.")

    render_footer()

# =============================================================
# VIEW: MODEL PERFORMANCE (real evaluation data — no live computation)
# =============================================================
elif app_mode == "Model Performance":
    render_hero()
    st.markdown('<p class="main-header">📈 Model Performance</p>', unsafe_allow_html=True)
    st.caption("Confusion matrices and classification reports from the real, held-out test-set evaluation of both models. These numbers are pre-computed — not calculated live.")

    metrics = load_metrics_json()
    baseline = metrics.get("baseline")
    distil = metrics.get("distilbert")

    if not baseline and not distil:
        st.warning("Metrics files (`metrics.json` / `distilbert_metrics.json`) weren't found alongside the app, so this dashboard can't be shown.")
    else:
        # ---------------- Headline metric comparison ----------------
        st.markdown('<p class="section-header">📊 Accuracy & Macro F1 Comparison</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        if baseline:
            c1.metric("Baseline Accuracy", f"{baseline['accuracy']*100:.1f}%")
            c2.metric("Baseline Macro F1", f"{baseline['classification_report']['macro avg']['f1-score']*100:.1f}%")
        if distil:
            c3.metric("DistilBERT Accuracy", f"{distil['test_results']['test_accuracy']*100:.1f}%")
            c4.metric("DistilBERT Macro F1", f"{distil['test_results']['test_f1_macro']*100:.1f}%")

        if baseline and distil:
            fig_cmp = go.Figure(data=[
                go.Bar(name="Baseline (TF-IDF + LR)",
                       x=["Accuracy", "Macro F1"],
                       y=[baseline["accuracy"], baseline["classification_report"]["macro avg"]["f1-score"]],
                       marker_color="#94A3B8"),
                go.Bar(name="Multilingual DistilBERT",
                       x=["Accuracy", "Macro F1"],
                       y=[distil["test_results"]["test_accuracy"], distil["test_results"]["test_f1_macro"]],
                       marker_color="#3B82F6"),
            ])
            fig_cmp.update_layout(barmode="group", yaxis_tickformat=".0%", height=380,
                                   margin=dict(t=20, b=20))
            st.plotly_chart(fig_cmp, use_container_width=True)

        # ---------------- Confusion matrices ----------------
        st.markdown('<p class="section-header">🔢 Confusion Matrices</p>', unsafe_allow_html=True)
        cm_col1, cm_col2 = st.columns(2)

        def plot_confusion_matrix(cm, labels, title):
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=labels, y=labels, colorscale="Blues",
                text=cm, texttemplate="%{text}", showscale=False,
            ))
            fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual",
                               yaxis_autorange="reversed", height=380, margin=dict(t=40, b=20))
            return fig

        with cm_col1:
            if baseline:
                st.plotly_chart(plot_confusion_matrix(baseline["confusion_matrix"], baseline["labels"],
                                                       "Baseline (TF-IDF + Logistic Regression)"),
                                 use_container_width=True)
        with cm_col2:
            if distil:
                labels_cap = [l.upper() for l in distil["class_names"]]
                st.plotly_chart(plot_confusion_matrix(distil["confusion_matrix"], labels_cap,
                                                       "Multilingual DistilBERT"),
                                 use_container_width=True)

        # ---------------- Classification reports ----------------
        st.markdown('<p class="section-header">📋 Classification Reports</p>', unsafe_allow_html=True)
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            if baseline:
                st.markdown("**Baseline (TF-IDF + Logistic Regression)**")
                rows = []
                for cls in baseline["labels"]:
                    r = baseline["classification_report"][cls]
                    rows.append({"Class": cls, "Precision": round(r["precision"], 3),
                                 "Recall": round(r["recall"], 3), "F1": round(r["f1-score"], 3),
                                 "Support": int(r["support"])})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with rep_col2:
            if distil:
                st.markdown("**Multilingual DistilBERT**")
                rows = []
                for cls in distil["class_names"]:
                    r = distil["per_class_metrics"][cls]
                    rows.append({"Class": cls.upper(), "Precision": round(r["precision"], 3),
                                 "Recall": round(r["recall"], 3), "F1": round(r["f1"], 3),
                                 "Support": int(r["support"])})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ---------------- Class distribution pie chart ----------------
        st.markdown('<p class="section-header">🥧 Test-Set Class Distribution</p>', unsafe_allow_html=True)
        st.caption("Real class balance in the held-out test set used to evaluate the baseline model — most course reviews are positive.")
        if baseline:
            pie_labels = baseline["labels"]
            pie_values = [baseline["classification_report"][c]["support"] for c in pie_labels]
            fig_pie = go.Figure(data=[go.Pie(
                labels=pie_labels, values=pie_values, hole=0.4,
                marker=dict(colors=[SENT_COLORS[l] for l in pie_labels]),
            )])
            fig_pie.update_layout(height=380, margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.info("ℹ️ These evaluation numbers come from the project's real training run (see `metrics.json` / `distilbert_metrics.json`) — they are not recalculated each time the app runs.")

    render_footer()

# =============================================================
# VIEW: ABOUT
# =============================================================
elif app_mode == "About":
    render_hero()
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.write(
        "Course Feedback Sentiment Analysis is an AI-powered educational analytics application "
        "designed to transform student feedback into clear and actionable insights."
    )
    st.write(
        "The application analyzes individual reviews as well as complete CSV datasets, identifies "
        "important course-related aspects, determines their sentiment, and provides model-level "
        "explanations to help understand the prediction."
    )

    st.markdown('<p class="section-header">✨ What this application provides</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    provides = [
        ("💬 Single Review", ["Analyze one student review", "Identify overall sentiment",
                              "Classify as Positive, Neutral, or Negative"]),
        ("📄 CSV Analysis", ["Analyze complete feedback datasets", "Detect a suitable review column",
                             "Perform sentiment analysis at scale"]),
        ("🔗 Aspect Analysis", ["Identify important course-related aspects", "Determine sentiment for each aspect",
                                "Covers instructor, content, assignments, difficulty, and platform etc."]),
    ]
    for col, (title, bullets) in zip(cols, provides):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                for b in bullets:
                    st.markdown(f"- {b}")
    cols2 = st.columns(2)
    provides2 = [
        ("💡 Explainable AI", ["Explain sentiment predictions", "Identify influential words",
                               "Show word-level model contributions"]),
        ("⬇️ Downloadable Results", ["Export analysis results", "Download detailed reports and CSV results",
                                     "Save Explainable AI graphs as PNG"]),
    ]
    for col, (title, bullets) in zip(cols2, provides2):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                for b in bullets:
                    st.markdown(f"- {b}")

    st.markdown('<p class="section-header">🤖 Machine Learning Models</p>', unsafe_allow_html=True)
    mcols = st.columns(3)
    models_info = [
        ("🔵 TF-IDF + Logistic Regression", ["TF-IDF text feature extraction", "Logistic Regression classification",
                                             "3 sentiment classes", "Word-level feature contributions",
                                             "Supports Explainable AI"]),
        ("🔵 Fine-tuned Multilingual DistilBERT", ["Multilingual DistilBERT transformer", "Fine-tuned for course feedback",
                                                    "Captures contextual relationships", "3 sentiment classes",
                                                    "Student feedback analysis"]),
        ("🔵 Combined Model", ["Combines Logistic Regression + DistilBERT", "Uses both models for prediction",
                               "Averages probability outputs", "Traditional + transformer approaches",
                               "Recommended for final presentation"]),
    ]
    for col, (title, bullets) in zip(mcols, models_info):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                for b in bullets:
                    st.markdown(f"- {b}")

    st.markdown('<p class="section-header">🎯 Sentiment and Aspect Analysis</p>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### 🎯 Sentiment Detection")
        st.write("Each review is classified into one of three sentiment categories:")
        st.markdown("- 🟢 Positive\n- 🟡 Neutral\n- 🔴 Negative")

    st.markdown('<p class="section-header">🔗 Course Aspects</p>', unsafe_allow_html=True)
    st.write("The application can identify aspects such as:")
    st.markdown("".join(f'<span class="pill">{a}</span>' for a in ASPECT_NAMES), unsafe_allow_html=True)

    st.markdown('<p class="section-header">⚙️ How the Application Works</p>', unsafe_allow_html=True)
    steps = [
        ("Enter or upload feedback", "Analyze a single review or upload a CSV file containing multiple course reviews."),
        ("Select an AI model", "Choose Logistic Regression, DistilBERT, or the recommended Combined model."),
        ("Generate sentiment predictions", "The selected model determines the overall sentiment and confidence."),
        ("Discover course aspects", "The application identifies relevant aspects and evaluates their individual sentiment."),
        ("Understand the prediction", "Explainable AI highlights words that contributed to the model prediction."),
        ("Download the results", "Reports, CSV results, and Explainable AI graphs can be exported for further use."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"**{i}️⃣ {title}:** {desc}")

    st.markdown('<p class="section-header">📈 Why This Application Is Useful</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="highlight-box">'
        '<b>🎓 Turning student feedback into actionable insights.</b><br><br>'
        'Instead of manually reviewing large volumes of course feedback, educators and course teams can '
        'quickly identify overall sentiment, discover specific areas that students appreciate or struggle with, '
        'and understand the factors influencing model predictions.'
        '</div>', unsafe_allow_html=True
    )

    st.markdown("")
    checklist = ["💬 Single Review Analysis", "📄 CSV File Analysis", "🔗 Aspect Analysis",
                 "💡 Explainable AI", "📊 Model Confidence", "⬇️ Downloadable Results", "🖥️ CPU Compatible"]
    with st.container(border=True):
        for item in checklist:
            st.markdown(f'<div class="check-row">✅ {item}</div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="highlight-box"><b>Created By Afsah Arshad</b></div>', unsafe_allow_html=True)

    render_footer()
