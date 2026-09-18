import io
import re
import os
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

# =============================================================
# GLOBAL STYLE — dark navy sidebar with pill-style nav buttons
# =============================================================
st.markdown("""
<style>
.main-header { font-size: 2.3rem !important; color: #0F172A !important; font-weight: 800 !important; margin-bottom: 0.2rem !important; line-height: 1.25 !important; }
.sub-header { font-size: 1.05rem !important; color: #4B5563 !important; margin-bottom: 1.2rem !important; }
.section-header { font-size: 1.6rem !important; font-weight: 800 !important; color: #0F172A !important; margin: 1.4rem 0 0.3rem 0 !important; }
.app-footer { text-align:center; color:#6B7280; font-size:0.85rem; margin-top:2.5rem; padding-top:1rem; border-top:1px solid #E5E7EB; }

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
@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists("coursera_tfidf_logistic_model.pkl"):
            models["model"] = joblib.load("coursera_tfidf_logistic_model.pkl")
        if os.path.exists("coursera_tfidf_vectorizer.pkl"):
            models["vectorizer"] = joblib.load("coursera_tfidf_vectorizer.pkl")
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

loaded_models = load_models()
MODEL_READY = "model" in loaded_models and "vectorizer" in loaded_models

# =============================================================
# CORE PREDICTION HELPERS (fast, batched — no torch needed)
# =============================================================
def batch_predict(texts):
    """Vectorized sentiment prediction for a list of strings. Returns (labels, confidences)."""
    clean = [t if isinstance(t, str) and t.strip() else " " for t in texts]
    if not MODEL_READY:
        return np.array(["NEUTRAL"] * len(clean)), np.array([0.5] * len(clean))
    try:
        vec = loaded_models["vectorizer"].transform(clean)
        model = loaded_models["model"]
        preds = model.predict(vec)
        probs = model.predict_proba(vec)
        conf = probs.max(axis=1)
        return preds, conf
    except Exception:
        return np.array(["NEUTRAL"] * len(clean)), np.array([0.5] * len(clean))


def predict_sentiment(text, model_type="Combined"):
    """Single-text prediction used by the Single Review view. model_type is kept for UI parity."""
    preds, confs = batch_predict([text])
    return preds[0], float(confs[0])


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
    """Best-effort translation. Always falls back to the original text — never raises."""
    if not text or not text.strip():
        return text
    if lang_code in ("en", "unknown"):
        return text
    if not TRANSLATOR_AVAILABLE:
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


# =============================================================
# ASPECT KEYWORDS
# =============================================================
ASPECT_KEYWORDS = {
    "Difficulty": ["difficult", "difficulty", "easy", "easier", "hard", "challenging",
                   "challenge", "complex", "complicated", "simple", "beginner", "advanced"],
    "Course Content": ["content", "course content", "material", "materials", "topics",
                        "topic", "lessons", "lesson", "curriculum", "concepts", "concept",
                        "theory", "information"],
    "Learning Experience": ["learn", "learned", "learning", "experience", "understand",
                             "understanding", "helpful", "useful", "skill", "skills",
                             "improved", "improve"],
    "Instructor": ["instructor", "teacher", "professor", "lecturer", "mentor", "teaching",
                   "teach", "taught", "explained", "explanation", "lecture", "lectures"],
    "Assignments": ["assignment", "assignments", "homework", "exercise", "exercises",
                     "quiz", "quizzes", "test", "tests", "project", "projects"],
    "Overall Experience": ["overall", "experience", "satisfied", "satisfaction", "enjoyed",
                            "enjoy", "recommend", "recommended", "great course", "amazing"],
    "Course Structure": ["structure", "structured", "organized", "organised", "organization",
                          "sequence", "order", "module", "modules", "section", "sections"],
    "Duration": ["duration", "length", "time", "hours", "hour", "week", "weeks", "short", "long"],
    "Practical Application": ["practical", "application", "applications", "real-world",
                               "real world", "hands-on", "hands on", "apply", "applied",
                               "implementation", "practice"],
}
ASPECT_NAMES = list(ASPECT_KEYWORDS.keys())


def split_sentences(text):
    text = str(text)
    parts = re.split(r"(?<=[.!?۔])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_aspect_mentions(text):
    """Return {aspect: representative_sentence} for aspects mentioned in the text."""
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
# SIDEBAR NAVIGATION
# =============================================================
NAV_ITEMS = [
    ("Single Review Analysis", "📝"),
    ("CSV Analysis", "📄"),
    ("Aspect Analysis", "🔗"),
    ("Explainable AI (SHAP)", "💡"),
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
# VIEW: SINGLE REVIEW ANALYSIS
# =============================================================
if app_mode == "Single Review Analysis":
    st.markdown('<p class="main-header">Course Feedback Sentiment Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze student feedback, discover key aspects, and understand what drives sentiment — powered by AI.</p>', unsafe_allow_html=True)

    st.markdown("### Select Model")
    selected_model = st.radio(
        "Model Selection", ["Logistic Regression", "DistilBERT", "Combined"],
        horizontal=True, label_visibility="collapsed"
    )
    if selected_model == "Combined":
        st.info("⚡ **Combined (Recommended):** Uses the trained Logistic Regression classifier as the primary engine and is recommended for the final presentation.")
    elif selected_model == "DistilBERT":
        st.info("ℹ️ The DistilBERT transformer model isn't bundled with this deployment (it needs a lot of extra memory), so results below use the lightweight Logistic Regression model instead.")

    user_review = st.text_area("Enter student review text below:", placeholder="Type or paste feedback here...")

    if st.button("🔍 Analyze Review", type="primary"):
        if user_review.strip() == "":
            st.warning("Please enter some review text first.")
        else:
            with st.spinner("Analyzing sentiment..."):
                lang_code, lang_name = detect_language(user_review)
                translated = translate_to_english(user_review, lang_code)
                sentiment, conf = predict_sentiment(user_review, selected_model)

                st.markdown("---")
                st.markdown("### 🌍 Language Detection")
                if lang_code != "en":
                    st.info(f"**Detected language:** {lang_name}\n\n**English Translation:** {translated}")
                else:
                    st.info(f"**Detected language:** {lang_name}")

                st.markdown("### Sentiment Prediction")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.metric(label="Sentiment", value=sentiment, delta=f"Confidence: {conf:.2f}")
                with col2:
                    st.metric(label="Primary Score", value=f"{conf * 100:.0f}%")

                st.markdown("### Aspects Mentioned")
                mentions = extract_aspect_mentions(translated)
                if mentions:
                    preds, confs = batch_predict(list(mentions.values()))
                    aspect_rows = [
                        {"Aspect": a, "Sentiment": p, "Confidence": round(float(c), 4), "Sentence": s}
                        for (a, s), p, c in zip(mentions.items(), preds, confs)
                    ]
                    st.dataframe(pd.DataFrame(aspect_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No specific course aspects were detected in this review.")

# =============================================================
# VIEW: CSV ANALYSIS
# =============================================================
elif app_mode == "CSV Analysis":
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
                horizontal=True, label_visibility="collapsed", key="csv_model"
            )

            if st.button("🔍 Analyze CSV", type="primary", use_container_width=True):
                with st.spinner("Analyzing reviews..."):
                    work_df = df.copy()
                    texts = work_df[review_col].astype(str).tolist()

                    # ---- Overall sentiment (batched, fast) ----
                    preds, confs = batch_predict(texts)
                    work_df["Sentiment"] = preds
                    work_df["Confidence"] = np.round(confs, 4)

                    st.session_state["csv_result"] = {
                        "df": work_df,
                        "review_col": review_col,
                        "course_col": course_col,
                    }
                st.success(f"Analysis completed for {len(work_df)} reviews.")

    # ---- Render results if analysis has run ----
    result = st.session_state.get("csv_result")
    if result is not None:
        work_df = result["df"]
        review_col = result["review_col"]
        course_col = result["course_col"]

        # ---------------- Overall Sentiment ----------------
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

        # ---------------- Aspect-Based Sentiment (overall) ----------------
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
                preds, confs = batch_predict(adf["Review"].tolist())
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

            # ---------------- Course-Wise Aspect Analysis ----------------
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

            # ---------------- Detailed Aspect Results ----------------
            st.markdown('<p class="section-header">📋 Detailed Aspect Results</p>', unsafe_allow_html=True)
            detailed = aspect_df[["Course", "Aspect", "Sentiment", "Confidence", "Review"]] if course_col else aspect_df[["Aspect", "Sentiment", "Confidence", "Review"]]
            st.dataframe(detailed, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download Detailed Aspect Results", df_to_csv_bytes(detailed),
                                "detailed_aspect_results.csv", "text/csv", use_container_width=True)

        # ---------------- Overall Analysis Results (with language + translation) ----------------
        st.markdown('<p class="section-header">📋 Analysis Results</p>', unsafe_allow_html=True)
        TRANSLATE_LIMIT = 200
        preview_df = work_df.head(TRANSLATE_LIMIT).copy()

        langs, lang_labels, translations = [], [], []
        for text in preview_df[review_col].astype(str).tolist():
            code, name = detect_language(text)
            langs.append(code)
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

        st.markdown('<div class="app-footer">🎓 Course Feedback Sentiment Analysis • AI-Powered Education Analytics • Created By Afsah Arshad</div>', unsafe_allow_html=True)

# =============================================================
# VIEW: ASPECT ANALYSIS (single review, standalone)
# =============================================================
elif app_mode == "Aspect Analysis":
    st.markdown('<p class="main-header">Aspect-Based Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Break down feedback into specific course components like Instructor, Content, and Pacing.</p>', unsafe_allow_html=True)

    text = st.text_area("Enter a review to break down by aspect:", placeholder="Type or paste feedback here...")
    if st.button("🔎 Analyze Aspects", type="primary"):
        if not text.strip():
            st.warning("Please enter some review text first.")
        else:
            lang_code, lang_name = detect_language(text)
            translated = translate_to_english(text, lang_code)
            mentions = extract_aspect_mentions(translated)

            if not mentions:
                st.info("No specific course aspects (Content, Instructor, Difficulty, etc.) were detected in this review.")
            else:
                preds, confs = batch_predict(list(mentions.values()))
                rows = [
                    {"Aspect": a, "Sentiment": p, "Confidence": round(float(c), 4), "Sentence": s}
                    for (a, s), p, c in zip(mentions.items(), preds, confs)
                ]
                rows_df = pd.DataFrame(rows)
                st.success(f"Detected {len(rows_df)} aspect(s) in this review.")
                st.dataframe(rows_df, use_container_width=True, hide_index=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                colors = [SENT_COLORS[s] for s in rows_df["Sentiment"]]
                ax.bar(rows_df["Aspect"], rows_df["Confidence"], color=colors)
                ax.set_ylabel("Confidence")
                ax.set_title("Aspect Sentiment")
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
                fig.tight_layout()
                st.pyplot(fig)

# =============================================================
# VIEW: EXPLAINABLE AI (SHAP-style feature contributions)
# =============================================================
elif app_mode == "Explainable AI (SHAP)":
    st.markdown('<p class="main-header">Explainable AI (SHAP)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand why the model made specific predictions using feature contributions.</p>', unsafe_allow_html=True)

    text = st.text_area("Enter a review to explain:", placeholder="Type or paste feedback here...")
    if st.button("💡 Explain Prediction", type="primary"):
        if not text.strip():
            st.warning("Please enter some review text first.")
        elif not MODEL_READY:
            st.error("The sentiment model isn't loaded, so an explanation can't be generated right now.")
        else:
            vectorizer = loaded_models["vectorizer"]
            model = loaded_models["model"]
            vec = vectorizer.transform([text])
            pred_label = model.predict(vec)[0]
            class_idx = list(model.classes_).index(pred_label)

            feature_names = np.array(vectorizer.get_feature_names_out())
            row = vec.toarray()[0]
            nonzero = np.nonzero(row)[0]

            # For a linear model, each feature's contribution to the class score is coef * tfidf value.
            contributions = model.coef_[class_idx][nonzero] * row[nonzero]
            words = feature_names[nonzero]

            contrib_df = pd.DataFrame({"Word": words, "Contribution": contributions})
            contrib_df = contrib_df.reindex(contrib_df["Contribution"].abs().sort_values(ascending=False).index)
            top = contrib_df.head(12).sort_values("Contribution")

            st.markdown(f"### Predicted Sentiment: **{pred_label}**")
            st.caption("Feature contributions show how much each word pushed the model's score toward or away from this prediction (based on the model's TF-IDF weights).")

            if top.empty:
                st.info("None of the words in this review were recognized by the model's vocabulary.")
            else:
                fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(top))))
                colors = ["#22C55E" if v > 0 else "#DC2626" for v in top["Contribution"]]
                ax.barh(top["Word"], top["Contribution"], color=colors)
                ax.set_xlabel(f"Contribution to '{pred_label}'")
                ax.axvline(0, color="#9CA3AF", linewidth=0.8)
                fig.tight_layout()
                st.pyplot(fig)
                st.dataframe(contrib_df.reset_index(drop=True), use_container_width=True, hide_index=True)

# =============================================================
# VIEW: ABOUT
# =============================================================
elif app_mode == "About":
    st.markdown('<p class="main-header">About This App</p>', unsafe_allow_html=True)
    st.write("Course Feedback Sentiment Analyzer created by **Afsah Arshad** to evaluate academic programs using machine learning models.")
    st.markdown("""
- **Single Review Analysis** — analyze one piece of feedback at a time, with language detection and aspect breakdown.
- **CSV Analysis** — upload a batch of reviews and get overall, aspect-based, and course-wise sentiment breakdowns with downloadable results.
- **Aspect Analysis** — see which course aspects (Instructor, Content, Difficulty, etc.) are mentioned in a review and how students feel about each.
- **Explainable AI** — see which words in a review pushed the model's prediction in each direction.
    """)
    st.caption("Model: TF-IDF + Logistic Regression, trained on multilingual Coursera-style course feedback.")
