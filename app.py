import io
import re
import os
import json
import html as html_lib
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    TRANSLATOR_AVAILABLE = True
except Exception:
    TRANSLATOR_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

DISTILBERT_REPO = "Afsah-2027/coursera-multilingual-distilbert"

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

.orig-text-left { text-align: left !important; white-space: pre-wrap; direction: ltr; unicode-bidi: plaintext; }

section[data-testid="stSidebar"] {
    background-color: #0B1B38;
}
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
section[data-testid="stSidebar"] * { color: #E5E7EB; }

.sidebar-logo {
    width: 44px; height: 44px;
    display:flex; align-items:center; justify-content:center;
    margin-bottom: 0.6rem;
}
.sidebar-logo svg { width: 40px; height: 40px; }
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
    justify-content: flex-start !important;
}
/* The icon + label sit in an inner flex container that Streamlit centers by
   default — force it to hug the left edge so nav buttons read left-aligned. */
section[data-testid="stSidebar"] div.stButton > button > div {
    width: 100%;
    justify-content: flex-start !important;
    text-align: left !important;
}
section[data-testid="stSidebar"] div.stButton > button p {
    text-align: left !important;
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


# MyMemoryTranslator validates the source language strictly against its own code
# table (case-sensitive, e.g. "zh-CN" not "zh-cn"), so langdetect's raw codes must
# be mapped to MyMemory's exact codes or every call raises LanguageNotSupportedException
# and silently falls through to "no translation" — this was the root cause of
# translations not appearing on some deployments.
_MYMEMORY_LANG_MAP = {
    "ur": "ur-PK", "zh-cn": "zh-CN", "zh": "zh-CN", "zh-tw": "zh-TW",
    "ko": "ko-KR", "ru": "ru-RU", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
    "hi": "hi-IN", "ar": "ar-SA", "pt": "pt-PT", "ja": "ja-JP", "it": "it-IT",
    "tr": "tr-TR", "id": "id-ID", "vi": "vi-VN", "bn": "bn-IN", "fa": "fa-IR",
    "nl": "nl-NL",
}
_MYMEMORY_TARGET_EN = "en-GB"

# MyMemory's free tier rejects requests over ~500 characters, so long text is
# translated in chunks and stitched back together.
_MYMEMORY_CHUNK_LIMIT = 450


def _chunk_text(text, limit):
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    chunks, current = [], ""
    for part in parts:
        if len(current) + len(part) + 1 > limit:
            if current:
                chunks.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


def _translate_mymemory(text, lang_code):
    src = _MYMEMORY_LANG_MAP.get(lang_code, "auto")
    if len(text) <= _MYMEMORY_CHUNK_LIMIT:
        return MyMemoryTranslator(source=src, target=_MYMEMORY_TARGET_EN).translate(text)
    translator = MyMemoryTranslator(source=src, target=_MYMEMORY_TARGET_EN)
    return " ".join(translator.translate(c) for c in _chunk_text(text, _MYMEMORY_CHUNK_LIMIT))


def _translate_google_direct(text):
    """Last-resort fallback: hit Google's public (unofficial) translate endpoint
    directly over HTTPS, bypassing deep_translator entirely. This works from hosts
    where deep_translator's GoogleTranslator is blocked for other reasons (e.g. a
    stale/rotated endpoint) but plain outbound HTTPS to Google still succeeds."""
    resp = requests.get(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": text},
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(seg[0] for seg in data[0] if seg[0])


def _translate_one(text, lang_code, target="en"):
    """Try three independent translation providers in order. Returns None
    (never raises) if all of them fail."""
    try:
        result = GoogleTranslator(source="auto", target=target).translate(text)
        if result and result.strip():
            return result
    except Exception:
        pass

    try:
        result = _translate_mymemory(text, lang_code)
        if result and result.strip():
            return result
    except Exception:
        pass

    try:
        result = _translate_google_direct(text)
        if result and result.strip():
            return result
    except Exception:
        pass

    return None


def translate_to_english(text, lang_code):
    """Best-effort translation, trying three independent providers (Google Translate,
    MyMemory, then a direct Google endpoint call) before finally giving up and
    returning the original text. This function never raises.

    Note: NLLB-200 (a ~2.4GB local model) was tried here previously, but loading it
    alongside DistilBERT reliably exceeded the free hosting tier's memory limit and
    hard-crashed the app (an OS-level OOM kill, which can't be caught by try/except).
    It has been removed for stability."""
    if not text or not text.strip():
        return text
    if lang_code in ("en", "unknown"):
        return text

    if TRANSLATOR_AVAILABLE:
        result = _translate_one(text, lang_code)
        if result and result.strip():
            return result

    # Last resort — never raise, just show the original text.
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

# Actionable suggestions shown for every sentiment — POSITIVE notes reinforce what's
# working, NEGATIVE/NEUTRAL suggest an improvement.
ASPECT_SUGGESTIONS = {
    "Course Content": {
        "POSITIVE": "Content is landing well — keep it as the model for future modules.",
        "NEGATIVE": "Review and refresh the course material — add more examples and simplify dense sections.",
        "NEUTRAL": "Consider adding supplementary examples or reading material to deepen the content.",
    },
    "Instructor": {
        "POSITIVE": "Instructor delivery is a strength — capture what's working for onboarding other instructors.",
        "NEGATIVE": "Provide the instructor with feedback on pacing and clarity of explanations.",
        "NEUTRAL": "Encourage more interactive Q&A or office hours to strengthen the instructor connection.",
    },
    "Assignments": {
        "POSITIVE": "Assignments are hitting the mark — reuse this format for upcoming modules.",
        "NEGATIVE": "Re-calibrate assignment difficulty and provide clearer instructions or worked examples.",
        "NEUTRAL": "Add a few practice assignments with varying difficulty to build confidence gradually.",
    },
    "Quizzes & Assessments": {
        "POSITIVE": "Assessments feel fair and well-calibrated — no changes needed here.",
        "NEGATIVE": "Review quiz difficulty and grading transparency; consider adding practice quizzes.",
        "NEUTRAL": "Offer optional practice assessments so students can self-check before graded ones.",
    },
    "Difficulty": {
        "POSITIVE": "Pacing and difficulty are well-tuned for this audience — maintain the current level.",
        "NEGATIVE": "Add a beginner-friendly on-ramp or prerequisite guide to ease the learning curve.",
        "NEUTRAL": "Offer optional advanced and beginner tracks so pacing fits more skill levels.",
    },
    "Learning Experience": {
        "POSITIVE": "Students report strong learning outcomes — highlight this course as a benchmark.",
        "NEGATIVE": "Gather specific feedback on what hindered learning and iterate on the weakest module.",
        "NEUTRAL": "Add more interactive or hands-on elements to make learning more engaging.",
    },
    "Course Structure": {
        "POSITIVE": "The module sequence is working well — no restructuring needed.",
        "NEGATIVE": "Reorganize modules into a clearer, more logical sequence with defined milestones.",
        "NEUTRAL": "Add a visual course roadmap so students can see how modules connect.",
    },
    "Platform": {
        "POSITIVE": "Platform experience is smooth — good baseline to maintain during future updates.",
        "NEGATIVE": "Investigate platform usability issues — navigation, load times, or broken links.",
        "NEUTRAL": "Collect specific UX feedback to identify small friction points in the platform.",
    },
    "Video & Audio": {
        "POSITIVE": "Recording quality is a plus — keep the same setup for future recordings.",
        "NEGATIVE": "Improve recording audio quality and consider re-recording unclear video segments.",
        "NEUTRAL": "Add captions or transcripts to improve accessibility and clarity.",
    },
    "Certificates": {
        "POSITIVE": "Certificate process is clear and valued — no changes needed.",
        "NEGATIVE": "Clarify certificate requirements and streamline the issuance process.",
        "NEUTRAL": "Highlight the certificate's value and how it can be used professionally.",
    },
    "Duration": {
        "POSITIVE": "Course length feels right to students — keep the current pacing.",
        "NEGATIVE": "Reassess course length — consider splitting into shorter, focused modules.",
        "NEUTRAL": "Offer a suggested pacing guide so students can plan their time better.",
    },
    "Value": {
        "POSITIVE": "Students feel they're getting good value — a strong point to highlight in marketing.",
        "NEGATIVE": "Reassess pricing relative to content depth, or add more bonus resources.",
        "NEUTRAL": "Communicate the course's value proposition more clearly upfront.",
    },
    "Practical Application": {
        "POSITIVE": "Hands-on application is resonating — keep expanding this type of exercise.",
        "NEGATIVE": "Add more real-world projects or case studies to bridge theory and practice.",
        "NEUTRAL": "Include at least one capstone-style project to reinforce practical skills.",
    },
    "Relevance": {
        "POSITIVE": "Content feels current and industry-relevant — keep it updated on this cadence.",
        "NEGATIVE": "Update course content to reflect current industry practices and tools.",
        "NEUTRAL": "Periodically refresh examples to keep the material feeling current.",
    },
    "Overall Experience": {
        "POSITIVE": "Overall experience is strong — a good candidate for a testimonial or case study.",
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


def render_aspect_cards(mentions_dict, engine="Logistic Regression"):
    """mentions_dict: {aspect: sentence}. Renders a sentiment summary strip, then a
    3-col card grid with sentiment, confidence, the detected sentence, and a suggestion.
    Uses the same model `engine` as the overall sentiment prediction, so a review that
    scores Neutral overall (e.g. via the Combined model) doesn't show aspects scored by a
    different, single model that may disagree with the overall verdict."""
    if not mentions_dict:
        st.info("No specific course aspects (Content, Instructor, Difficulty, etc.) were detected in this review.")
        return None
    items = list(mentions_dict.items())
    preds, confs = batch_predict([s for _, s in items], engine=engine)

    # Summary strip: how many aspects landed Positive / Neutral / Negative. This is
    # what actually explains an overall-Neutral verdict — a mixed spread across
    # aspects, rather than every aspect independently being "neutral".
    pos_n = sum(1 for p in preds if p == "POSITIVE")
    neu_n = sum(1 for p in preds if p == "NEUTRAL")
    neg_n = sum(1 for p in preds if p == "NEGATIVE")
    total = len(preds)
    m1, m2, m3 = st.columns(3)
    m1.metric("🟢 Positive aspects", f"{pos_n}/{total}")
    m2.metric("🟡 Neutral aspects", f"{neu_n}/{total}")
    m3.metric("🔴 Negative aspects", f"{neg_n}/{total}")
    if total > 1:
        bar_html = (
            '<div style="display:flex;width:100%;height:10px;border-radius:6px;overflow:hidden;margin:0.4rem 0 1rem 0;">'
            f'<div style="width:{pos_n/total*100:.1f}%;background:#22C55E;"></div>'
            f'<div style="width:{neu_n/total*100:.1f}%;background:#F59E0B;"></div>'
            f'<div style="width:{neg_n/total*100:.1f}%;background:#DC2626;"></div>'
            '</div>'
        )
        st.markdown(bar_html, unsafe_allow_html=True)

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
                    snippet = sentence if len(sentence) <= 140 else sentence[:137].rstrip() + "…"
                    st.caption(f"📝 “{snippet}”")
                    suggestion = get_aspect_suggestion(aspect, pred)
                    if suggestion:
                        label = "Note" if pred == "POSITIVE" else "Suggestion"
                        st.markdown(f"💡 *{label}: {suggestion}*")
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
    """SHAP-style word contribution chart, matching the reference design exactly:
    a single header/caption followed by a blue horizontal bar chart of the top
    positive-contributing words. Returns the figure (or None)."""
    if show_header:
        st.markdown('<p class="section-header">💡 Explainable AI (SHAP)</p>', unsafe_allow_html=True)
    st.caption("Words that influenced the prediction")
    contrib_df = get_word_contributions(text, pred_label)
    if contrib_df.empty:
        st.info("None of the words in this review were recognized by the model's vocabulary.")
        return None

    positive_only = contrib_df[contrib_df["Contribution"] > 0]
    if positive_only.empty:
        st.info("None of the words in this review pushed toward this prediction.")
        return None
    top = positive_only.head(5).sort_values("Contribution")

    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.5 * len(top))))
    ax.barh(top["Word"], top["Contribution"], color="#1E7FD8")
    ax.set_title("Top Contributing Words")
    ax.set_xlabel("SHAP Value")
    fig.tight_layout()
    st.pyplot(fig)

    return fig


# =============================================================
# ONE-TIME TOAST
# =============================================================
st.toast("Turning student feedback into actionable insights", icon="🎓")

# =============================================================
# SIDEBAR NAVIGATION
# =============================================================
NAV_ITEMS = [
    ("Single Review Analysis", ":material/chat:"),
    ("CSV Analysis", ":material/description:"),
    ("Aspect Analysis", ":material/link:"),
    ("Explainable AI (SHAP)", ":material/lightbulb:"),
    ("About", ":material/info:"),
]

if "nav" not in st.session_state:
    st.session_state.nav = "Single Review Analysis"

with st.sidebar:
    GRAD_CAP_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3B82F6">'
                     '<path d="M12 3 1 9l11 6 9-4.91V17h2V9L12 3z"/>'
                     '<path d="M5 13.18v4L12 21l7-3.82v-4L12 17l-7-3.82z"/></svg>')
    st.markdown(f'<div class="sidebar-logo">{GRAD_CAP_SVG}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Course Feedback<br>Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">AI-Powered Insights for Better Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-author">Created By Afsah Arshad</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    for name, icon in NAV_ITEMS:
        is_active = st.session_state.nav == name
        if st.button(name, key=f"nav_{name}", use_container_width=True, icon=icon,
                     type="primary" if is_active else "secondary"):
            st.session_state.nav = name
            st.rerun()

# Left-align original (possibly RTL-script) feedback text wherever it's shown,
# so Urdu/Arabic text doesn't auto-flip to right-aligned next to its translation.
st.markdown("""
<style>
[data-testid="stDataFrame"] div[role="gridcell"] { direction: ltr !important; text-align: left !important; unicode-bidi: plaintext !important; }
</style>
""", unsafe_allow_html=True)

app_mode = st.session_state.nav

# =============================================================
# SAMPLE FEEDBACKS & MODEL DESCRIPTIONS (Single Review)
# =============================================================
SAMPLE_CSV_DATASETS = {
    "-- None (upload your own) --": None,
    "Course Feedback by Course Name (30 rows)": "sample_course_feedbacks_by_course_name.csv",
    "Course Feedback with Details (30 rows)": "sample_course_feedbacks_with_details.csv",
    "Multilingual Course Feedback Test (45 rows)": "sample_multilingual_course_feedback_test.csv",
    "Course Feedback with Sentiment Labels (60 rows)": "sample_book1.csv",
    "Simple Course Feedbacks (30 rows)": "sample_course_feedbacks.csv",
    "Simple Test Reviews (30 rows)": "sample_test.csv",
    "Full Original Dataset (123,240 rows)": "sample_full_dataset.csv",
}

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

                if lang_code != "en":
                    with st.container(border=True):
                        st.markdown("#### 🌐 English Translation")
                        st.markdown(f"Detected language: **{lang_name}**")
                        st.markdown("**Original Feedback**")
                        st.markdown(f'<div class="orig-text-left">{html_lib.escape(user_review)}</div>',
                                    unsafe_allow_html=True)
                        st.markdown("**English Translation**")
                        st.write(translated)
                    st.markdown("")

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

                st.markdown('<p class="section-header">🔗 Aspect Analysis</p>', unsafe_allow_html=True)
                st.caption("Key aspects detected in the feedback and their sentiment")
                mentions = extract_aspect_mentions(translated)
                render_aspect_cards(mentions, engine=selected_model)

                shap_fig = render_explainable_section(translated, sentiment)

                lr_conf, distil_conf, combined_conf, distil_available = get_engine_confidences(translated)
                report_txt = (
                    f"Review: {user_review}\n\nLanguage: {lang_name}\nTranslation: {translated}\n\n"
                    f"Sentiment: {sentiment} (confidence {conf:.2f})\n\n"
                    f"Model Confidence -> Logistic Regression: {lr_conf:.2f}, DistilBERT: {distil_conf:.2f}, Combined: {combined_conf:.2f}\n"
                )
                result_df = pd.DataFrame([{"Review": user_review, "Sentiment": sentiment, "Confidence": round(conf, 4)}])

                mc_col, qa_col = st.columns(2)
                with mc_col:
                    with st.container(border=True):
                        st.markdown("#### 📊 Model Confidence")
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
                with qa_col:
                    with st.container(border=True):
                        st.markdown("#### ⚡ Quick Actions")
                        st.download_button("📄 Download Detailed Report", report_txt.encode("utf-8"),
                                            "detailed_report.txt", "text/plain", use_container_width=True)
                        st.download_button("⬇️ Download Results", df_to_csv_bytes(result_df),
                                            "single_review_result.csv", "text/csv", use_container_width=True)
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

    sample_col, upload_col = st.columns(2)
    with sample_col:
        sample_choice = st.selectbox("Load a sample dataset (optional)", list(SAMPLE_CSV_DATASETS.keys()), key="csv_sample_choice")
    with upload_col:
        uploaded_file = st.file_uploader("Or upload your own CSV file", type=["csv"])
    st.info("💡 Your CSV can contain additional columns such as CourseId, rating, date, or other metadata.")

    df = None
    sample_file = SAMPLE_CSV_DATASETS.get(sample_choice)
    if sample_file:
        try:
            df = pd.read_csv(sample_file)
            st.caption(f"📂 Loaded bundled sample: **{sample_choice}**")
        except Exception as e:
            st.error(f"Couldn't load the sample dataset: {e}")
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read this CSV: {e}")

    if True:
        if df is not None and not df.empty:
            if len(df) > 5000:
                st.caption("⚡ This is a large dataset — Logistic Regression will run fastest. DistilBERT/Combined may take a while.")
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
                lang_code, _ = detect_language(text)
                translated_text = cached_translate(text, lang_code)
                mentions = extract_aspect_mentions(translated_text)
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

        ASPECT_ANALYSIS_ROW_CAP = 5000
        if len(texts_) > ASPECT_ANALYSIS_ROW_CAP:
            st.caption(f"⚡ Aspect analysis is computed on the first {ASPECT_ANALYSIS_ROW_CAP:,} of {len(texts_):,} rows to keep things fast. Overall sentiment above still covers all rows.")
            texts_for_aspects = texts_[:ASPECT_ANALYSIS_ROW_CAP]
            courses_for_aspects = courses_list[:ASPECT_ANALYSIS_ROW_CAP] if courses_list else None
        else:
            texts_for_aspects = texts_
            courses_for_aspects = courses_list

        aspect_df = build_aspect_table(tuple(texts_for_aspects), tuple(courses_for_aspects) if courses_for_aspects else None)

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
