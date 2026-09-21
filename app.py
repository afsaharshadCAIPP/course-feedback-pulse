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
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except Exception:
    TRANSLATOR_AVAILABLE = False

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

.insight-box {
    background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
    padding:1rem 1.2rem;
}

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
    width: 100%; text-align: left; border-radius: 10px; border: none;
    background-color: #14213F; color: #E5E7EB !important;
    padding: 0.65rem 0.9rem; margin-bottom: 0.5rem; font-weight: 600; font-size: 0.95rem;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #1D2E52; color: #FFFFFF !important;
}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background-color: #3B82F6 !important; color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model...")
def load_models():
    models = {}
    try:
        if os.path.exists("coursera_tfidf_logistic_model.pkl"):
            models["model"] = joblib.load("coursera_tfidf_logistic_model.pkl")
        if os.path.exists("coursera_tfidf_vectorizer.pkl"):
            models["vectorizer"] = joblib.load("coursera_tfidf_vectorizer.pkl")
    except Exception:
        pass
    return models

loaded_models = load_models()
MODEL_READY = "model" in loaded_models and "vectorizer" in loaded_models

def normalize_label(lbl):
    return str(lbl).strip().upper()

def align_probs(probs, classes):
    probs = np.asarray(probs)
    out = np.zeros((probs.shape[0], len(SENT_ORDER)))
    for j, cls in enumerate(SENT_ORDER):
        if cls in classes:
            out[:, j] = probs[:, classes.index(cls)]
    sums = out.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return out / sums

def batch_predict_lr(texts):
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

def batch_predict(texts):
    lr_probs, lr_classes = batch_predict_lr(texts)
    final = align_probs(lr_probs, lr_classes)
    idx = final.argmax(axis=1)
    labels = np.array([SENT_ORDER[i] for i in idx])
    conf = final.max(axis=1)
    return labels, conf

def get_full_probs(text):
    if not text.strip():
        return {"POSITIVE": 1 / 3, "NEUTRAL": 1 / 3, "NEGATIVE": 1 / 3}
    lr_probs, lr_classes = batch_predict_lr([text])
    final = align_probs(lr_probs, lr_classes)[0]
    return {cls: float(final[i]) for i, cls in enumerate(SENT_ORDER)}

def get_word_contributions(text, pred_label):
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

LANG_NAMES = {
    "en": "English", "ur": "Urdu", "zh-cn": "Chinese", "zh": "Chinese", "ko": "Korean",
    "ru": "Russian", "es": "Spanish", "fr": "French", "de": "German", "hi": "Hindi",
    "ar": "Arabic", "pt": "Portuguese", "ja": "Japanese", "it": "Italian", "tr": "Turkish",
}

def detect_language(text):
    try:
        code = detect(text)
        return code, LANG_NAMES.get(code, code.upper())
    except Exception:
        return "en", "English"

def translate_to_english(text, lang_code):
    if not text or not text.strip() or lang_code == "en":
        return text
    if TRANSLATOR_AVAILABLE:
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception:
            pass
    return text

ASPECT_KEYWORDS = {
    "Course Content": ["content", "material", "topics", "curriculum", "concepts", "syllabus"],
    "Instructor": ["instructor", "teacher", "professor", "lecturer", "teaching", "taught", "explained"],
    "Assignments": ["assignment", "homework", "exercise", "project", "task"],
    "Quizzes & Assessments": ["quiz", "test", "exam", "assessment", "grading", "grade"],
    "Difficulty": ["difficult", "easy", "hard", "challenging", "complex", "simple"],
    "Learning Experience": ["learn", "learning", "experience", "understand", "helpful", "useful", "skill"],
    "Course Structure": ["structure", "organized", "sequence", "module", "section"],
    "Platform": ["platform", "website", "app", "interface", "portal", "dashboard"],
}
ASPECT_NAMES = list(ASPECT_KEYWORDS.keys())
ASPECT_ICONS = {
    "Course Content": "📖", "Instructor": "👤", "Assignments": "📋",
    "Quizzes & Assessments": "📝", "Difficulty": "🎯", "Learning Experience": "🎓",
    "Course Structure": "🏗️", "Platform": "💻"
}

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

def detect_review_column(df):
    for cand in ["feedback", "review", "reviews", "text", "comment", "comments", "description"]:
        for c in df.columns:
            if cand in c.lower():
                return c
    obj_cols = list(df.select_dtypes(include="object").columns)
    return obj_cols[0] if obj_cols else df.columns[0]

def detect_course_column(df, review_col):
    for cand in ["course name", "coursename", "course_name", "course title", "coursetitle", "course_title", "course"]:
        for c in df.columns:
            if cand in c.lower():
                return c
    for c in df.columns:
        if c != review_col and df[c].dtype == object:
            return c
    return None

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")

def render_hero():
    title_col, badge_col = st.columns([3, 1])
    with title_col:
        st.markdown('<p class="main-header">🎓 Course Feedback Sentiment Analysis</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Analyze student feedback, discover key aspects, and understand sentiment.</p>', unsafe_allow_html=True)
    with badge_col:
        st.markdown(
            '<div class="info-badge-card">'
            '<div class="badge-title">🎓 AI Education Analytics</div>'
            '<div class="badge-body">Sentiment • Aspects • Explainability</div>'
            '</div>', unsafe_allow_html=True)
    st.markdown("---")

def render_footer():
    st.markdown('<div class="app-footer">🎓 Course Feedback Sentiment Analysis • Created By Afsah Arshad</div>', unsafe_allow_html=True)

NAV_ITEMS = [
    ("Single Review Analysis", "💬"),
    ("CSV Analysis", "📄"),
    ("About", "ℹ️"),
]

if "nav" not in st.session_state:
    st.session_state.nav = "Single Review Analysis"

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎓</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Course Feedback<br>Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-author">Created By Afsah Arshad</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    for name, icon in NAV_ITEMS:
        is_active = st.session_state.nav == name
        if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.nav = name
            st.rerun()

app_mode = st.session_state.nav

SAMPLE_FEEDBACKS = {
    "Choose a feedback...": "",
    "Excellent course": "This course is excellent and very easy to follow. I learned a lot.",
    "Good but difficult": "The instructor was good and the content was useful, but some assignments were difficult.",
    "Multilingual (Urdu)": "کورس بہت اچھا تھا لیکن اسائنمنٹس تھوڑے مشکل تھے۔",
    "Multilingual (Chinese)": "这门课程非常棒，内容清晰，而且我学到了很多东西。",
}

def _apply_sample():
    choice = st.session_state.get("sample_choice")
    if choice and SAMPLE_FEEDBACKS.get(choice):
        st.session_state["single_review_text"] = SAMPLE_FEEDBACKS[choice]

if app_mode == "Single Review Analysis":
    render_hero()
    st.markdown('<p class="section-header">💬 Single Review Analysis</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Select sample feedback", list(SAMPLE_FEEDBACKS.keys()), key="sample_choice", on_change=_apply_sample)
    with c2:
        user_review = st.text_area("Review text", key="single_review_text", value=st.session_state.get("single_review_text", "The instructor was good and the content was useful, but some assignments were difficult."))

    if st.button("▶ Analyze Review", type="primary"):
        if not user_review.strip():
            st.warning("Please enter review text.")
        else:
            lang_code, lang_name = detect_language(user_review)
            translated = translate_to_english(user_review, lang_code)
            probs = get_full_probs(translated)
            sentiment = max(probs, key=probs.get)
            conf = probs[sentiment]

            with st.container(border=True):
                st.markdown(f"### Prediction: {SENT_EMOJI[sentiment]} **{sentiment.capitalize()}** (Confidence: {conf:.2f})")
                for cls in SENT_ORDER:
                    st.markdown(f"{cls}: {probs[cls]*100:.0f}%")
                    st.progress(float(probs[cls]))

            if lang_code != "en":
                st.info(f"🌍 Detected language: **{lang_name}** | Translation: {translated}")

            st.markdown("### Aspect Analysis")
            mentions = extract_aspect_mentions(translated)
            if mentions:
                for asp, sent in mentions.items():
                    st.markdown(f"- **{ASPECT_ICONS.get(asp, '🔹')} {asp}**: {sent}")
            else:
                st.info("No specific aspects detected.")

    render_footer()

elif app_mode == "CSV Analysis":
    render_hero()
    st.markdown('<p class="section-header">📄 Batch CSV Analysis</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            review_col = detect_review_column(df)
            course_col = detect_course_column(df, review_col)
            st.success(f"Review column: **{review_col}** | Course column: **{course_col if course_col else 'None'}**")

            if st.button("Run Analysis", type="primary"):
                work_df = df.copy()
                texts = work_df[review_col].astype(str).tolist()
                preds, confs = batch_predict(texts)
                work_df["Sentiment"] = preds
                work_df["Confidence"] = np.round(confs, 4)
                st.dataframe(work_df, use_container_width=True)
                st.download_button("Download CSV", df_to_csv_bytes(work_df), "results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    render_footer()

elif app_mode == "About":
    render_hero()
    st.markdown("### About This App")
    render_footer()
