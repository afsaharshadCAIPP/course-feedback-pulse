# ============================================================
# COURSE FEEDBACK SENTIMENT ANALYSIS
# Modern Education AI Dashboard
#
# Created By Afsah Arshad
#
# Features:
#   1. Single Review Analysis
#   2. CSV File Analysis
#   3. Aspect Analysis
#   4. Explainable AI
#   5. Model Confidence
#   6. Downloadable Results
#
# Models:
#   - TF-IDF + Logistic Regression
#   - Fine-tuned Multilingual DistilBERT
#   - Combined Model
#
# CPU Compatible
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import re
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import torch
import joblib

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Course Feedback Sentiment Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "Course Feedback Sentiment Analysis"

APP_SUBTITLE = (
    "Analyze student feedback, discover key aspects, "
    "and understand what drives sentiment — powered by AI."
)

CREATED_BY = "Created By Afsah Arshad"

DEVICE = torch.device("cpu")

MAX_LENGTH = 128

# Local Hugging Face translation model.
# The model is downloaded once from Hugging Face and then
# cached locally by Transformers. No Google Translate/API
# request is used for translation.
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"
TRANSLATION_TARGET_LANGUAGE = "eng_Latn"
TRANSLATION_MAX_INPUT_LENGTH = 512
TRANSLATION_MAX_OUTPUT_LENGTH = 128


# ============================================================
# MODEL PATHS
# ============================================================

DISTILBERT_MODEL_DIR = (
    "./coursera_multilingual_distilbert"
)

TFIDF_MODEL_FILE = (
    "./coursera_tfidf_logistic_model.pkl"
)

TFIDF_VECTORIZER_FILE = (
    "./coursera_tfidf_vectorizer.pkl"
)


# ============================================================
# SENTIMENT LABELS
# ============================================================

LABEL_NAMES = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE"
}

DISPLAY_LABELS = {
    "NEGATIVE": "Negative",
    "NEUTRAL": "Neutral",
    "POSITIVE": "Positive"
}


# ============================================================
# SAMPLE FEEDBACK
# ============================================================

SAMPLE_FEEDBACK = {

    "Excellent course":
        "This course was excellent. "
        "The instructor explained everything clearly "
        "and the assignments were very useful.",

    "Very helpful":
        "The course was very helpful and easy to understand. "
        "I learned many useful concepts.",

    "Good but difficult":
        "The instructor was good and the content was useful, "
        "but some assignments were difficult.",

    "Average experience":
        "The course was okay. Some topics were useful, "
        "but the overall experience was average.",

    "Poor experience":
        "The course was difficult to follow. "
        "The assignments were confusing and "
        "the explanations were not clear.",

    "Excellent instructor":
        "The instructor was excellent and explained "
        "difficult topics in a very clear and engaging way.",

    "Difficult assignments":
        "The assignments were too difficult and required "
        "much more time than expected.",

    "Good course structure":
        "The course structure was very organized "
        "and the lessons were arranged in a logical sequence."
}


# ============================================================
# ASPECT KEYWORDS
# ============================================================

ASPECT_KEYWORDS = {

    "Course Content": [
        "content", "topic", "topics", "lesson", "lessons", "material", "materials",
        "curriculum", "concept", "concepts", "subject", "subjects", "course content",
        "course material", "learning material", "study material", "lecture", "lectures",
        "theory", "theories", "case study", "case studies"
    ],

    "Instructor": [
        "instructor", "teacher", "professor", "lecturer", "trainer", "educator",
        "teaching", "teach", "taught", "explanation", "explained", "explain",
        "instruction", "instructions", "guidance", "feedback", "presentation",
        "presenter", "teaching style", "teaching method", "instructor style",
        "instructor feedback"
    ],

    "Assignments": [
        "assignment", "assignments", "homework", "exercise", "exercises", "task",
        "tasks", "project", "projects", "submission", "submissions", "peer review",
        "peer reviewed", "practical assignment", "graded assignment"
    ],

    "Quizzes & Assessments": [
        "quiz", "quizzes", "test", "tests", "exam", "exams", "assessment",
        "assessments", "graded quiz", "graded quizzes", "graded test", "graded tests",
        "final exam", "final assessment", "knowledge check", "knowledge checks",
        "evaluation", "evaluations"
    ],

    "Difficulty": [
        "difficult", "difficulty", "hard", "easy", "challenging", "confusing",
        "complex", "simple", "complicated", "advanced", "beginner", "basic",
        "struggle", "struggled", "struggling", "manageable", "overwhelming",
        "overwhelmed", "straightforward", "too difficult", "too easy",
        "easy to understand", "hard to understand"
    ],

    "Learning Experience": [
        "learning", "learn", "learned", "learning experience", "understand",
        "understanding", "skills", "skill", "educational", "insight", "insights",
        "improve", "improvement", "progress", "learned a lot", "learn something",
        "gained knowledge", "gained skills", "new skills", "new knowledge"
    ],

    "Course Structure": [
        "structure", "structured", "organized", "organised", "organization",
        "organisation", "sequence", "module", "modules", "section", "sections",
        "chapter", "chapters", "unit", "units", "course flow", "course structure",
        "course organization", "course organisation", "well organized",
        "well organised", "poorly organized", "poorly organised"
    ],

    "Platform": [
        "platform", "website", "interface", "user interface", "UI", "app",
        "application", "navigation", "navigate", "loading", "load", "buffering",
        "playback", "technical", "technical issue", "technical issues", "technology",
        "software", "bug", "bugs", "error", "errors", "login", "access", "accessibility"
    ],

    "Video & Audio": [
        "video", "videos", "lecture video", "lecture videos", "video quality",
        "video resolution", "video clarity", "audio", "audio quality", "sound quality",
        "sound", "voice quality", "voice", "subtitles", "subtitle", "captions",
        "caption", "transcript", "transcripts", "video playback", "audio clarity"
    ],

    "Certificates": [
        "certificate", "certificates", "certification", "certifications",
        "credential", "credentials", "completion certificate", "course certificate",
        "certificate of completion", "digital certificate", "certification process"
    ],

    "Duration": [
        "duration", "course duration", "course length", "length", "hour", "hours",
        "week", "weeks", "day", "days", "time commitment", "time required",
        "study time", "learning time", "completion time", "too long", "too short",
        "time consuming", "time-consuming", "pace", "pacing"
    ],

    "Value": [
        "value", "worth", "price", "cost", "money", "benefit", "benefits",
        "valuable", "worthwhile", "affordable", "expensive", "cheap", "pricing",
        "course fee", "fee", "value for money", "worth the money", "worth the price",
        "return on investment", "ROI"
    ],

    "Practical Application": [
        "practical", "practical application", "real world", "real-world", "real life",
        "real-life", "hands on", "hands-on", "hands on experience", "industry example",
        "industry examples", "real world example", "real world examples",
        "apply knowledge", "apply skills", "applying knowledge", "applying skills",
        "practical skills", "practical knowledge"
    ],

    "Relevance": [
        "relevant", "relevance", "up to date", "up-to-date", "current information",
        "modern", "outdated", "obsolete", "industry relevant", "industry relevance",
        "job relevant", "career relevant", "relevant to my work",
        "relevant to my career"
    ],

    "Overall Experience": [
        "overall experience", "course experience", "overall course experience",
        "overall learning experience", "student experience", "my experience",
        "my overall experience", "experience with the course",
        "experience of the course", "general experience", "overall impression",
        "general impression", "overall satisfaction", "course satisfaction",
        "overall feeling", "general feeling", "overall opinion", "general opinion",
        "opinion about the course", "opinion of the course", "enjoyed the course",
        "enjoyment of the course", "satisfied with the course",
        "dissatisfied with the course", "happy with the course",
        "unhappy with the course", "liked the course", "loved the course",
        "disliked the course", "recommend the course", "recommend this course",
        "would recommend", "not recommend", "course was", "course felt",
        "felt about the course"
    ]

}


# ============================================================
# ASPECT ICONS
# ============================================================

ASPECT_ICONS = {

    "Course Content": "📖",
    "Instructor": "👤",
    "Assignments": "📄",
    "Difficulty": "🎯",
    "Learning Experience": "💡",
    "Course Structure": "🗂️",
    "Platform": "🖥️",
    "Certificates": "🏷️",
    "Duration": "⏱️",
    "Value": "💎"
}


# ============================================================
# SESSION STATE
# ============================================================

if "active_page" not in st.session_state:
    st.session_state.active_page = "Single Review Analysis"

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

if "single_result" not in st.session_state:
    st.session_state.single_result = None

if "csv_results" not in st.session_state:
    st.session_state.csv_results = None

if "csv_model" not in st.session_state:
    st.session_state.csv_model = "Combined"

if "csv_review_column" not in st.session_state:
    st.session_state.csv_review_column = None


# ============================================================
# CUSTOM COMPACT DIVIDER
# ============================================================

def compact_divider():
    st.markdown(
        '<hr class="compact-divider">',
        unsafe_allow_html=True
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .key-insight-text {
        white-space: normal;
        overflow: visible;
        width: 100%;
        font-size: 16px;
        line-height: 1.5;
        padding-bottom: 8px;
    }

    .capability-heading {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 1.05rem;
        font-weight: 600;
        line-height: 1.3;
        margin-bottom: 1rem;
    }

    .model-card-title {
        white-space: normal;
        overflow: visible;
        word-break: normal;
        overflow-wrap: break-word;
        font-size: 1.08rem;
        font-weight: 700;
        line-height: 1.3;
        min-height: 2.8rem;
        margin-bottom: 0.65rem;
    }

    .sentiment-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 16px;
    }

    .sentiment-circle {
        width: 12px;
        height: 12px;
        min-width: 12px;
        border-radius: 50%;
        display: inline-block;
    }

    .positive-circle {
        background-color: #22c55e;
    }

    .negative-circle {
        background-color: #ef4444;
    }

    .neutral-circle {
        background-color: #facc15;
    }

    /* ============================================================
       MAIN HEADER
       ============================================================ */

    /* ============================================================
       MAIN HEADER
       Native Streamlit container styling — no raw HTML markup.
       ============================================================ */

    .st-key-main_header {
        background: #eaf3ff;
        border: 1px solid #d7e6f8;
        border-radius: 14px;
        padding: 24px 28px 22px 28px;
        margin: 0;
    }

    .st-key-main_header [data-testid="stHorizontalBlock"] {
        align-items: flex-start;
        gap: 24px;
    }

    .st-key-main_header [data-testid="stColumn"]:first-child
    [data-testid="stHorizontalBlock"] {
        align-items: flex-start;
    }

    .st-key-main_header [data-testid="stColumn"]:first-child
    [data-testid="stColumn"]:first-child {
        display: flex;
        align-items: flex-start;
        justify-content: center;
    }

    .st-key-main_header [data-testid="stColumn"]:first-child {
        min-width: 0;
    }

    .st-key-main_header h2 {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 2.85rem !important;
        line-height: 1.08 !important;
        letter-spacing: -0.025em !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    .st-key-header_info {
        border: 1.5px solid #123f78 !important;
        border-radius: 10px !important;
        background: transparent !important;
        padding: 12px 22px 16px 16px !important;
        box-sizing: border-box;
        min-height: 0 !important;
    }

    .st-key-header_info p {
        margin: 0 !important;
        padding: 0 !important;
        color: #123f78 !important;
    }

    .st-key-header_info [data-testid="stMarkdownContainer"] {
        padding: 0 !important;
    }

    .header-info-title {
        color: #123f78 !important;
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
        margin: 0 0 5px 0 !important;
    }

    .header-info-text {
        color: #123f78 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        line-height: 1.45 !important;
        margin: 0 !important;
    }

    .main-header-title {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 2.85rem !important;
        line-height: 1.08 !important;
        letter-spacing: -0.025em;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    .main-header-subtitle {
        margin: 10px 0 0 0 !important;
        padding: 0 !important;
        font-size: 1.08rem !important;
        line-height: 1.5 !important;
        color: #667085 !important;
    }

    @media (max-width: 900px) {
        .st-key-main_header {
            padding: 20px;
        }

        .st-key-main_header [data-testid="stHorizontalBlock"] {
            gap: 14px;
        }

        .main-header-title {
            font-size: 2.35rem !important;
        }
    }

    /* ============================================================
       COMPACT DIVIDERS
       Use a custom rule instead of Streamlit's default divider
       so the vertical spacing is fully controlled.
       ============================================================ */

    .compact-divider {
        border: 0 !important;
        border-top: 1px solid #d1d5db !important;
        margin: 0.55rem 0 !important;
        padding: 0 !important;
        height: 0 !important;
        display: block !important;
    }

    /* Keep page headings close to the divider above them. */
    div[data-testid="stHeading"] h1,
    h1 {
        margin-top: 0.35rem !important;
    }

    /* Keep the header-to-divider transition compact. */
    .main-header-divider {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* ============================================================
       BUTTON + SIDEBAR THEME
       ============================================================ */

    /* Make all main-page buttons and download buttons blue */
    div.stButton > button,
    div.stDownloadButton > button,
    button[kind="secondary"],
    button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover,
    button[kind="secondary"]:hover,
    button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        border-color: #1e40af !important;
    }

    div.stButton > button:focus,
    div.stDownloadButton > button:focus,
    button[kind="secondary"]:focus,
    button[kind="primary"]:focus {
        color: #ffffff !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25) !important;
    }

    /* ============================================================
       BUTTON ICONS
       Keep all Streamlit button icons solid white
       ============================================================ */

    div.stButton > button [data-testid="stIconMaterial"],
    div.stDownloadButton > button [data-testid="stIconMaterial"],
    div.stButton > button .material-symbols-rounded,
    div.stButton > button .material-symbols-outlined,
    div.stDownloadButton > button .material-symbols-rounded,
    div.stDownloadButton > button .material-symbols-outlined {
        color: #ffffff !important;
        fill: #ffffff !important;
        font-variation-settings:
            "FILL" 1,
            "wght" 700,
            "GRAD" 0,
            "opsz" 24 !important;
    }

    /* Navy/dark-blue sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b1f3a !important;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #0b1f3a !important;
    }

    /* Move the entire sidebar content slightly upward */
    section[data-testid="stSidebar"] > div {
        padding-top: 0.75rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0.75rem !important;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #dbeafe !important;
    }

    /* Sidebar navigation: keep the entire icon + text group left aligned */
    section[data-testid="stSidebar"] div.stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
    }

    section[data-testid="stSidebar"] div.stButton > button > div {
        width: auto !important;
        flex: 0 0 auto !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }

    section[data-testid="stSidebar"] div.stButton > button p {
        margin: 0 !important;
        text-align: left !important;
    }

    /* Sidebar button icons: solid white */
    section[data-testid="stSidebar"] div.stButton > button [data-testid="stIconMaterial"],
    section[data-testid="stSidebar"] div.stButton > button .material-symbols-rounded,
    section[data-testid="stSidebar"] div.stButton > button .material-symbols-outlined {
        color: #ffffff !important;
        fill: #ffffff !important;
        font-variation-settings:
            "FILL" 1,
            "wght" 700,
            "GRAD" 0,
            "opsz" 24 !important;
    }

    /* Sidebar navigation buttons */
    /* Default: transparent with dark-blue text */
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
        background-color: rgba(255, 255, 255, 0.10) !important;
        color: #0B1F3A !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Hover: keep transparent and highlight the border in blue */
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.10) !important;
        color: #0B1F3A !important;
        border: 1px solid #3B82F6 !important;
        box-shadow: none !important;
    }

    /* Selected: blue background with white text */
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    /* Keep selected button blue */
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:focus,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:active {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border-color: #3B82F6 !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }

    /* Sidebar brand icon: centered, slightly larger, and blue */
    .sidebar-brand-icon {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        width: 100%;
        margin: 0 0 10px 0;
        padding-left: 4px;
        padding: 0;
        line-height: 1;
    }

    .sidebar-brand-icon svg {
        width: 46px;
        height: 46px;
        display: block;
        fill: #3B82F6 !important;
    }

    /* Reduce spacing between sidebar heading and subtitle */
    .sidebar-brand-title {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.25 !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }

    .sidebar-brand-subtitle {
        margin: 10px 0 0 0 !important;
        padding: 0 !important;
        line-height: 1.35 !important;
        font-size: 1rem !important;
        color: #dbeafe !important;
    }

</style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# MULTILINGUAL INPUT + LOCAL TRANSLATION
# ============================================================

# NLLB uses FLORES-200 language codes. This mapping covers
# common languages likely to appear in course feedback.
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "ar": "arb_Arab",
    "fa": "pes_Arab",
    "ur": "urd_Arab",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "ro": "ron_Latn",
    "cs": "ces_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "hu": "hun_Latn"
}


LANGUAGE_DISPLAY_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "tr": "Turkish",
    "pl": "Polish",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ar": "Arabic",
    "fa": "Persian",
    "ur": "Urdu",
    "hi": "Hindi",
    "bn": "Bengali",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "ro": "Romanian",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "hu": "Hungarian"
}


def detect_feedback_language(text):
    """
    Detect the language code used to select the NLLB source
    language. langdetect is used only for identification;
    translation itself is performed locally by Hugging Face.
    """
    text = clean_text(text)

    if not text:
        return "unknown"

    # Fast character/script checks for languages that are
    # particularly useful for multilingual course feedback.
    if re.search(r"[\u0600-\u06FF]", text):
        # Urdu/Persian/Arabic share Arabic script. langdetect
        # below can distinguish them when available.
        pass
    elif re.search(r"[\u0900-\u097F]", text):
        return "hi"
    elif re.search(r"[\u0980-\u09FF]", text):
        return "bn"
    elif re.search(r"[\u4E00-\u9FFF]", text):
        return "zh"
    elif re.search(r"[\u3040-\u30FF]", text):
        return "ja"
    elif re.search(r"[\uAC00-\uD7AF]", text):
        return "ko"
    elif re.search(r"[\u0400-\u04FF]", text):
        return "ru"

    try:
        from langdetect import detect

        detected = detect(text).lower()

        if detected in NLLB_LANGUAGE_CODES:
            return detected

    except Exception:
        pass

    # Safe fallback for Arabic-script feedback when a language
    # detector is unavailable.
    if re.search(r"[\u0600-\u06FF]", text):
        return "ur"

    return "unknown"


def language_display_name(code):
    code = str(code).lower()

    return LANGUAGE_DISPLAY_NAMES.get(
        code,
        code.upper() if code != "unknown" else "Unknown"
    )


@st.cache_resource(show_spinner=False)
def load_translation_model():
    """
    Load the local NLLB translation model once per Streamlit
    process. The first run downloads the model from Hugging Face;
    subsequent runs reuse the local Hugging Face cache.
    """
    translation_tokenizer = AutoTokenizer.from_pretrained(
        TRANSLATION_MODEL_NAME
    )

    translation_model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL_NAME
        )
    )

    translation_model.to(DEVICE)
    translation_model.eval()

    return (
        translation_tokenizer,
        translation_model
    )


def _get_nllb_language_id(tokenizer, language_code):
    """
    Support both current and older Transformers tokenizer
    implementations.
    """
    if hasattr(tokenizer, "lang_code_to_id"):

        language_id = tokenizer.lang_code_to_id.get(
            language_code
        )

        if language_id is not None:
            return language_id

    return tokenizer.convert_tokens_to_ids(
        language_code
    )


def translate_texts_to_english(texts, source_language):
    """
    Translate a list of texts from one detected language to
    English in a single local Hugging Face generation call.
    """
    cleaned_texts = [
        clean_text(text)
        for text in texts
    ]

    if not cleaned_texts:
        return []

    if source_language == "en":
        return cleaned_texts

    source_code = NLLB_LANGUAGE_CODES.get(
        source_language
    )

    if source_code is None:
        raise ValueError(
            f"The detected language '{source_language}' "
            "is not currently mapped to an NLLB language code."
        )

    translation_tokenizer, translation_model = (
        load_translation_model()
    )

    translation_tokenizer.src_lang = source_code

    inputs = translation_tokenizer(
        cleaned_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=TRANSLATION_MAX_INPUT_LENGTH
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    target_language_id = _get_nllb_language_id(
        translation_tokenizer,
        TRANSLATION_TARGET_LANGUAGE
    )

    with torch.no_grad():

        generated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=target_language_id,
            max_length=TRANSLATION_MAX_OUTPUT_LENGTH
        )

    return [
        clean_text(value)
        for value in translation_tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
    ]


def translate_feedback_batch(texts):
    """
    Detect and translate a collection of reviews.

    English reviews are returned unchanged. Non-English reviews
    are grouped by source language and translated in batches.
    This is substantially more suitable for CSV analysis than
    sending one external API request per review.
    """
    cleaned_texts = [
        clean_text(text)
        for text in texts
    ]

    results = [
        {
            "original": text,
            "translated": text,
            "language": "unknown",
            "language_name": "Unknown",
            "was_translated": False
        }
        for text in cleaned_texts
    ]

    language_groups = {}

    for index, text in enumerate(cleaned_texts):

        if not text:
            continue

        language = detect_feedback_language(
            text
        )

        results[index]["language"] = language
        results[index]["language_name"] = (
            language_display_name(language)
        )

        if language == "en":

            results[index]["translated"] = text

        elif language in NLLB_LANGUAGE_CODES:

            language_groups.setdefault(
                language,
                []
            ).append(index)

        else:
            raise ValueError(
                "The language of one or more reviews "
                "could not be mapped to a supported NLLB "
                "language. Please enter feedback in a "
                "supported language or use English."
            )

    for language, indices in language_groups.items():

        translated_texts = translate_texts_to_english(
            [
                cleaned_texts[index]
                for index in indices
            ],
            language
        )

        for index, translated in zip(
            indices,
            translated_texts
        ):

            results[index]["translated"] = (
                translated
            )
            results[index]["was_translated"] = True

    return results


def prepare_feedback_for_analysis(text):
    """
    Prepare one review for the existing English-based
    sentiment, aspect, and SHAP components.
    """
    text = clean_text(text)

    if not text:
        return (
            "",
            {
                "original": "",
                "translated": "",
                "language": "unknown",
                "language_name": "Unknown",
                "was_translated": False
            }
        )

    result = translate_feedback_batch(
        [text]
    )[0]

    return (
        result["translated"],
        result
    )


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_into_sentences(text):

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?。！？])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# NORMALIZE MODEL LABEL
# ============================================================

def normalize_label(value):

    if isinstance(
        value,
        (int, np.integer)
    ):

        return LABEL_NAMES.get(
            int(value),
            "NEUTRAL"
        )

    if isinstance(
        value,
        float
    ):

        if value.is_integer():

            return LABEL_NAMES.get(
                int(value),
                "NEUTRAL"
            )

    text = str(value).strip().upper()

    if text in [
        "0",
        "NEGATIVE",
        "NEG"
    ]:

        return "NEGATIVE"

    if text in [
        "1",
        "NEUTRAL",
        "NEU"
    ]:

        return "NEUTRAL"

    if text in [
        "2",
        "POSITIVE",
        "POS"
    ]:

        return "POSITIVE"

    return text


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():

    if not os.path.isdir(
        DISTILBERT_MODEL_DIR
    ):

        raise FileNotFoundError(
            "DistilBERT model folder not found:\n"
            f"{DISTILBERT_MODEL_DIR}"
        )

    if not os.path.isfile(
        TFIDF_MODEL_FILE
    ):

        raise FileNotFoundError(
            "Logistic Regression model not found:\n"
            f"{TFIDF_MODEL_FILE}"
        )

    if not os.path.isfile(
        TFIDF_VECTORIZER_FILE
    ):

        raise FileNotFoundError(
            "TF-IDF vectorizer not found:\n"
            f"{TFIDF_VECTORIZER_FILE}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        DISTILBERT_MODEL_DIR
    )

    distilbert_model = (
        AutoModelForSequenceClassification.from_pretrained(
            DISTILBERT_MODEL_DIR
        )
    )

    distilbert_model.to(
        DEVICE
    )

    distilbert_model.eval()

    tfidf_model = joblib.load(
        TFIDF_MODEL_FILE
    )

    tfidf_vectorizer = joblib.load(
        TFIDF_VECTORIZER_FILE
    )

    return (
        tokenizer,
        distilbert_model,
        tfidf_model,
        tfidf_vectorizer
    )


# ============================================================
# LOAD MODELS
# ============================================================

try:

    (
        tokenizer,
        distilbert_model,
        tfidf_model,
        tfidf_vectorizer
    ) = load_models()

except Exception as error:

    st.error(
        "Unable to load the trained models."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# DISTILBERT PREDICTION
# ============================================================

def predict_distilbert(text):

    text = clean_text(text)

    if not text:
        return None

    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = distilbert_model(
            **inputs
        )

        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=-1
            )[0]
            .cpu()
            .numpy()
        )

    if len(probabilities) < 3:

        raise ValueError(
            "The DistilBERT model does not appear "
            "to be a 3-class sentiment model."
        )

    probabilities = probabilities[:3]

    prediction_index = int(
        np.argmax(
            probabilities
        )
    )

    label = LABEL_NAMES.get(
        prediction_index,
        "NEUTRAL"
    )

    return {

        "label": label,

        "confidence": float(
            probabilities[
                prediction_index
            ]
        ),

        "probabilities": {

            "NEGATIVE": float(
                probabilities[0]
            ),

            "NEUTRAL": float(
                probabilities[1]
            ),

            "POSITIVE": float(
                probabilities[2]
            )
        }
    }


# ============================================================
# LOGISTIC REGRESSION PREDICTION
# ============================================================

def predict_logistic(text):

    text = clean_text(text)

    if not text:
        return None

    vector = (
        tfidf_vectorizer
        .transform([text])
    )

    raw_probabilities = (
        tfidf_model
        .predict_proba(vector)[0]
    )

    probabilities = {

        "NEGATIVE": 0.0,
        "NEUTRAL": 0.0,
        "POSITIVE": 0.0
    }

    for class_value, probability in zip(
        tfidf_model.classes_,
        raw_probabilities
    ):

        label = normalize_label(
            class_value
        )

        if label in probabilities:

            probabilities[label] = float(
                probability
            )

    label = max(
        probabilities,
        key=probabilities.get
    )

    return {

        "label": label,

        "confidence": probabilities[
            label
        ],

        "probabilities": probabilities
    }


# ============================================================
# COMBINED PREDICTION
# ============================================================

def predict_combined(text):

    logistic = predict_logistic(
        text
    )

    distilbert = predict_distilbert(
        text
    )

    probabilities = {

        label: (
            logistic["probabilities"][label]
            +
            distilbert["probabilities"][label]
        ) / 2

        for label in [
            "NEGATIVE",
            "NEUTRAL",
            "POSITIVE"
        ]
    }

    label = max(
        probabilities,
        key=probabilities.get
    )

    return {

        "label": label,

        "confidence": probabilities[
            label
        ],

        "probabilities": probabilities,

        "logistic": logistic,

        "distilbert": distilbert
    }


# ============================================================
# GENERAL PREDICTION
# ============================================================

def predict_sentiment(
    text,
    model_name
):

    if model_name == "Logistic Regression":

        return predict_logistic(
            text
        )

    if model_name == "DistilBERT":

        return predict_distilbert(
            text
        )

    return predict_combined(
        text
    )


# ============================================================
# ASPECT DETECTION
# ============================================================

def find_aspects(sentence):

    sentence_lower = (
        sentence.lower()
    )

    detected = []

    for aspect, keywords in (
        ASPECT_KEYWORDS.items()
    ):

        for keyword in keywords:

            pattern = (
                r"\b"
                +
                re.escape(
                    keyword.lower()
                )
                +
                r"\b"
            )

            if re.search(
                pattern,
                sentence_lower
            ):

                detected.append(
                    aspect
                )

                break

    return detected


# ============================================================
# ASPECT-SPECIFIC CLAUSE SPLITTING
# ============================================================

def split_into_clauses(sentence):

    sentence = clean_text(
        sentence
    )

    if not sentence:
        return []

    clauses = re.split(
        r"\s+(?:but|however|although|though|while|whereas|yet|except|"
        r"and yet|on the other hand)\s+|"
        r"\s*;\s*|"
        r"\s*,\s+(?=(?:the|this|that|these|those|my|our|your|"
        r"some|many|most|an|a)\b)",
        sentence,
        flags=re.IGNORECASE
    )

    cleaned = []

    for clause in clauses:

        clause = clean_text(
            clause
        )

        if clause:
            cleaned.append(
                clause
            )

    return cleaned


# ============================================================
# FIND THE RELEVANT CLAUSE FOR AN ASPECT
# ============================================================

def get_aspect_clauses(
    sentence,
    aspect
):

    clauses = split_into_clauses(
        sentence
    )

    keywords = ASPECT_KEYWORDS[
        aspect
    ]

    relevant = []

    for clause in clauses:

        clause_lower = clause.lower()

        found = False

        for keyword in keywords:

            pattern = (
                r"\b"
                +
                re.escape(
                    keyword.lower()
                )
                +
                r"\b"
            )

            if re.search(
                pattern,
                clause_lower
            ):

                found = True
                break

        if found:

            relevant.append(
                clause
            )

    if not relevant:

        return [
            sentence
        ]

    return relevant


# ============================================================
# EXTRACT ASPECT-SPECIFIC TEXT
# ============================================================

def extract_aspect_sentences(text):

    result = {
        aspect: []
        for aspect in ASPECT_KEYWORDS
    }

    sentences = split_into_sentences(
        text
    )

    for sentence in sentences:

        aspects = find_aspects(
            sentence
        )

        for aspect in aspects:

            clauses = get_aspect_clauses(
                sentence,
                aspect
            )

            for clause in clauses:

                if clause not in result[aspect]:

                    result[aspect].append(
                        clause
                    )

    return result


# ============================================================
# LOCAL ASPECT SENTIMENT LEXICON
# ============================================================

POSITIVE_SENTIMENT_WORDS = {

    "excellent",
    "amazing",
    "awesome",
    "great",
    "good",
    "helpful",
    "useful",
    "valuable",
    "effective",
    "clear",
    "clearly",
    "easy",
    "easier",
    "simple",
    "organized",
    "well-organized",
    "engaging",
    "interesting",
    "enjoyable",
    "enjoyed",
    "informative",
    "relevant",
    "practical",
    "beneficial",
    "fantastic",
    "wonderful",
    "perfect",
    "strong",
    "positive",
    "satisfied",
    "satisfying",
    "understandable",
    "accessible",
    "logical",
    "smooth",
    "convenient",
    "impressive",
    "motivating",
    "insightful",
    "worthwhile"
}


NEGATIVE_SENTIMENT_WORDS = {

    "bad",
    "poor",
    "terrible",
    "awful",
    "horrible",
    "worst",
    "useless",
    "confusing",
    "confused",
    "difficult",
    "hard",
    "challenging",
    "complex",
    "unclear",
    "disorganized",
    "unorganized",
    "boring",
    "frustrating",
    "frustrated",
    "annoying",
    "annoyed",
    "disappointing",
    "disappointed",
    "weak",
    "negative",
    "expensive",
    "waste",
    "wasted",
    "slow",
    "tedious",
    "stressful",
    "overwhelming",
    "unhelpful",
    "irrelevant",
    "unnecessary",
    "complicated",
    "poorly",
    "late",
    "long",
    "dissatisfied",
    "difficulties",
    "problem",
    "problems",
    "issue",
    "issues"
}


# ============================================================
# NEGATION WORDS
# ============================================================

NEGATION_WORDS = {

    "not",
    "no",
    "never",
    "neither",
    "nor",
    "isn't",
    "wasn't",
    "weren't",
    "aren't",
    "don't",
    "doesn't",
    "didn't",
    "can't",
    "cannot",
    "couldn't",
    "won't",
    "wouldn't",
    "hardly",
    "barely",
    "without"
}


# ============================================================
# INTENSIFIER WORDS
# ============================================================

INTENSIFIER_WORDS = {

    "very": 1.35,
    "really": 1.30,
    "extremely": 1.55,
    "highly": 1.45,
    "especially": 1.30,
    "quite": 1.20,
    "so": 1.25,
    "too": 1.20,
    "absolutely": 1.50,
    "completely": 1.45,
    "incredibly": 1.50
}


# ============================================================
# ASPECT SENTIMENT WORD EXTRACTION
# ============================================================

def get_aspect_context(
    clause,
    aspect,
    window=7
):

    words = re.findall(
        r"\b[\w'-]+\b",
        clause.lower()
    )

    if not words:
        return clause

    aspect_positions = []

    keywords = ASPECT_KEYWORDS[
        aspect
    ]

    for index, word in enumerate(
        words
    ):

        for keyword in keywords:

            keyword_words = (
                keyword.lower().split()
            )

            if (
                len(keyword_words) == 1
                and word == keyword_words[0]
            ):

                aspect_positions.append(
                    index
                )

    if not aspect_positions:

        return clause

    start = max(
        0,
        min(aspect_positions) - window
    )

    end = min(
        len(words),
        max(aspect_positions) + window + 1
    )

    return " ".join(
        words[start:end]
    )


# ============================================================
# LOCAL ASPECT SENTIMENT
# ============================================================

def calculate_local_aspect_sentiment(
    clause,
    aspect
):

    context = get_aspect_context(
        clause,
        aspect,
        window=7
    )

    words = re.findall(
        r"\b[\w'-]+\b",
        context.lower()
    )

    positive_score = 0.0
    negative_score = 0.0

    positive_hits = []
    negative_hits = []

    for index, word in enumerate(
        words
    ):

        sentiment = None

        if word in POSITIVE_SENTIMENT_WORDS:

            sentiment = "POSITIVE"

        elif word in NEGATIVE_SENTIMENT_WORDS:

            sentiment = "NEGATIVE"

        if sentiment is None:
            continue

        negation_found = False

        start = max(
            0,
            index - 3
        )

        previous_words = words[
            start:index
        ]

        for previous_word in previous_words:

            if previous_word in NEGATION_WORDS:

                negation_found = True
                break

        intensity = 1.0

        if index > 0:

            previous_word = words[
                index - 1
            ]

            intensity = INTENSIFIER_WORDS.get(
                previous_word,
                1.0
            )

        if negation_found:

            if sentiment == "POSITIVE":

                negative_score += (
                    intensity
                )

                negative_hits.append(
                    word
                )

            else:

                positive_score += (
                    intensity
                )

                positive_hits.append(
                    word
                )

        else:

            if sentiment == "POSITIVE":

                positive_score += (
                    intensity
                )

                positive_hits.append(
                    word
                )

            else:

                negative_score += (
                    intensity
                )

                negative_hits.append(
                    word
                )

    total = (
        positive_score
        +
        negative_score
    )

    if total == 0:

        return {

            "label": "NEUTRAL",

            "confidence": 0.0,

            "probabilities": {

                "NEGATIVE": 0.0,

                "NEUTRAL": 1.0,

                "POSITIVE": 0.0
            },

            "evidence": [],

            "context": context
        }

    if positive_score > negative_score:

        local_label = "POSITIVE"

        margin = (
            positive_score
            -
            negative_score
        )

    elif negative_score > positive_score:

        local_label = "NEGATIVE"

        margin = (
            negative_score
            -
            positive_score
        )

    else:

        local_label = "NEUTRAL"

        margin = 0.0

    local_confidence = min(
        0.95,
        0.55
        +
        (
            margin
            /
            max(total, 1.0)
        )
        *
        0.40
    )

    if positive_score > negative_score:

        positive_probability = (
            0.50
            +
            0.50
            *
            (
                positive_score
                /
                max(total, 1.0)
            )
        )

        negative_probability = (
            0.50
            *
            (
                negative_score
                /
                max(total, 1.0)
            )
        )

    elif negative_score > positive_score:

        negative_probability = (
            0.50
            +
            0.50
            *
            (
                negative_score
                /
                max(total, 1.0)
            )
        )

        positive_probability = (
            0.50
            *
            (
                positive_score
                /
                max(total, 1.0)
            )
        )

    else:

        positive_probability = 0.25
        negative_probability = 0.25

    neutral_probability = max(
        0.0,
        1.0
        -
        positive_probability
        -
        negative_probability
    )

    probabilities = {

        "NEGATIVE":
            float(negative_probability),

        "NEUTRAL":
            float(neutral_probability),

        "POSITIVE":
            float(positive_probability)
    }

    evidence = (
        positive_hits
        +
        negative_hits
    )

    return {

        "label": local_label,

        "confidence": float(
            local_confidence
        ),

        "probabilities": probabilities,

        "evidence": evidence,

        "context": context
    }


# ============================================================
# COMBINE LOCAL ASPECT SENTIMENT + DISTILBERT
# ============================================================

def combine_aspect_predictions(
    local_prediction,
    distilbert_prediction
):

    local_probabilities = (
        local_prediction[
            "probabilities"
        ]
    )

    model_probabilities = (
        distilbert_prediction[
            "probabilities"
        ]
    )

    has_local_evidence = (
        len(
            local_prediction[
                "evidence"
            ]
        ) > 0
    )

    if has_local_evidence:

        local_weight = 0.70
        model_weight = 0.30

    else:

        local_weight = 0.15
        model_weight = 0.85

    probabilities = {

        label: (
            local_weight
            *
            local_probabilities[label]
            +
            model_weight
            *
            model_probabilities[label]
        )

        for label in [
            "NEGATIVE",
            "NEUTRAL",
            "POSITIVE"
        ]
    }

    total = sum(
        probabilities.values()
    )

    if total > 0:

        probabilities = {

            label:
                value / total

            for label, value
            in probabilities.items()
        }

    label = max(
        probabilities,
        key=probabilities.get
    )

    confidence = probabilities[
        label
    ]

    return {

        "label": label,

        "confidence": float(
            confidence
        ),

        "probabilities": probabilities
    }


# ============================================================
# ASPECT SENTIMENT ANALYSIS
# ============================================================

def calculate_aspect_results(text):

    aspect_clauses = (
        extract_aspect_sentences(
            text
        )
    )

    results = []

    for aspect, clauses in (
        aspect_clauses.items()
    ):

        if not clauses:
            continue

        aspect_predictions = []

        for clause in clauses:

            local_prediction = (
                calculate_local_aspect_sentiment(
                    clause,
                    aspect
                )
            )

            distilbert_prediction = (
                predict_distilbert(
                    clause
                )
            )

            combined_prediction = (
                combine_aspect_predictions(
                    local_prediction,
                    distilbert_prediction
                )
            )

            aspect_predictions.append({

                "prediction":
                    combined_prediction,

                "sentence":
                    clause,

                "local_evidence":
                    local_prediction[
                        "evidence"
                    ]
            })

        if not aspect_predictions:
            continue

        averaged_probabilities = {

            "NEGATIVE": 0.0,

            "NEUTRAL": 0.0,

            "POSITIVE": 0.0
        }

        for item in aspect_predictions:

            probabilities = (
                item["prediction"][
                    "probabilities"
                ]
            )

            for label in averaged_probabilities:

                averaged_probabilities[
                    label
                ] += probabilities[
                    label
                ]

        number_of_predictions = (
            len(aspect_predictions)
        )

        for label in averaged_probabilities:

            averaged_probabilities[
                label
            ] /= number_of_predictions

        final_label = max(
            averaged_probabilities,
            key=averaged_probabilities.get
        )

        final_confidence = (
            averaged_probabilities[
                final_label
            ]
        )

        unique_sentences = []

        for item in aspect_predictions:

            sentence = item[
                "sentence"
            ]

            if sentence not in unique_sentences:

                unique_sentences.append(
                    sentence
                )

        supporting_text = " ".join(
            unique_sentences
        )

        results.append({

            "aspect": aspect,

            "sentiment": final_label,

            "confidence": float(
                final_confidence
            ),

            "sentence": supporting_text
        })

    return results


# ============================================================
# SENTIMENT DISPLAY HELPERS
# ============================================================

def sentiment_icon(label):

    if label == "POSITIVE":
        return "😊"

    if label == "NEGATIVE":
        return "☹️"

    return "😐"


def display_label(label):

    return DISPLAY_LABELS.get(
        label,
        label.title()
    )


# ============================================================
# SENTIMENT CIRCLE HELPER
# ============================================================

def sentiment_circle_html(label):

    if label == "POSITIVE":

        circle_class = "positive-circle"

    elif label == "NEGATIVE":

        circle_class = "negative-circle"

    else:

        circle_class = "neutral-circle"

    return (
        f'<span class="sentiment-circle '
        f'{circle_class}"></span>'
    )


# ============================================================
# KEY INSIGHT
# ============================================================

def create_key_insight(
    prediction
):

    label = prediction[
        "label"
    ]

    confidence = prediction[
        "confidence"
    ]

    if label == "POSITIVE":

        if confidence >= 0.80:

            return (
                "The model predicts a positive "
                "sentiment with high confidence. "
                "The feedback is generally favorable "
                "with only minor negative points."
            )

        return (
            "The feedback is generally positive, "
            "although the model has identified "
            "some uncertainty."
        )

    if label == "NEGATIVE":

        if confidence >= 0.80:

            return (
                "The model predicts a negative "
                "sentiment with high confidence. "
                "The feedback contains areas that "
                "may require attention or improvement."
            )

        return (
            "The feedback contains negative elements, "
            "although the overall prediction has "
            "some uncertainty."
        )

    return (
        "The model predicts a neutral sentiment. "
        "The feedback contains a mixture of "
        "positive and negative observations."
    )


# ============================================================
# HEADER
# ============================================================

def render_header():
    with st.container(key="main_header"):
        header_left, header_right = st.columns([4.6, 1.6], gap="large")

        with header_left:
            icon_col, title_col = st.columns([1.05, 4.0], gap="medium")

            with icon_col:
                # Original colourful graduation-cap icon.
                st.markdown(
                    """
                    <svg width="104" height="104" viewBox="0 0 64 64"
                         xmlns="http://www.w3.org/2000/svg"
                         aria-hidden="true"
                         style="display:block; margin-top:4px;">
                        <path fill="#5426A6"
                              d="M4 24.5 32 10l28 14.5-28 14.5L4 24.5Z"/>
                        <path fill="#5426A6"
                              d="M14 31v11.5C14 49.4 22.1 55 32 55s18-5.6 18-12.5V31l-5 2.6v8.9c0 3.8-5.9 7.5-13 7.5s-13-3.7-13-7.5v-8.9L14 31Z"/>
                        <path fill="#FFFFFF"
                              d="M19 32.2 32 39l13-6.8v8.7c0 3.8-5.9 7.5-13 7.5s-13-3.7-13-7.5v-8.7Z"/>
                        <path fill="#5426A6"
                              d="M56 27v13h-5V29.6L56 27Z"/>
                        <circle fill="#F6A623" cx="53.5" cy="44.5" r="2.5"/>
                    </svg>
                    """,
                    unsafe_allow_html=True
                )

            with title_col:
                st.markdown("## Course Feedback<br>Sentiment Analysis", unsafe_allow_html=True)
                st.markdown(APP_SUBTITLE)

        with header_right:
            with st.container(border=True, key="header_info"):
                st.markdown("**AI Education Analytics**")
                st.markdown("Sentiment&nbsp;&nbsp;&nbsp; Aspects&nbsp;&nbsp;&nbsp; Explainability", unsafe_allow_html=True)

    compact_divider()


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        # Centered blue graduation-cap icon
        st.markdown(
            '''
            <div class="sidebar-brand-icon">
                <svg viewBox="0 0 64 64" aria-hidden="true">
                    <path d="M4 24.5 32 10l28 14.5-28 14.5L4 24.5Z"/>
                    <path d="M14 31v11.5C14 49.4 22.1 55 32 55s18-5.6 18-12.5V31l-5 2.6v8.9c0 3.8-5.9 7.5-13 7.5s-13-3.7-13-7.5v-8.9L14 31Z"/>
                    <path d="M56 27v13h-5V29.6L56 27Z"/>
                    <circle cx="53.5" cy="44.5" r="2.5"/>
                </svg>
            </div>
            <div class="sidebar-brand-title">
                Course Feedback<br>
                Sentiment Analysis
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sidebar-brand-subtitle">AI-Powered Insights for Better Learning</div>',
            unsafe_allow_html=True
        )

        st.caption(
            CREATED_BY
        )

        pages = [

            (
                ":material/chat:",
                "Single Review Analysis"
            ),

            (
                ":material/description:",
                "CSV Analysis"
            ),

            (
                ":material/link:",
                "Aspect Analysis"
            ),

            (
                ":material/lightbulb:",
                "Explainable AI (SHAP)"
            ),

            (
                ":material/info:",
                "About"
            )
        ]

        for icon, page in pages:

            is_active = (
                st.session_state.active_page
                == page
            )

            if st.button(

                page,

                icon=icon,

                key=f"nav_{page}",

                use_container_width=True,

                type=(
                    "primary"
                    if is_active
                    else "secondary"
                )
            ):

                st.session_state.active_page = (
                    page
                )

                st.rerun()

        compact_divider()

        st.caption(
            "📈 Turning student feedback "
            "into actionable insights."
        )


# ============================================================
# REVIEW INPUT AREA
# ============================================================

def render_review_input():

    st.subheader(
        "  Single Review"
    )

    left, right = st.columns(
        [1.15, 2.0]
    )

    with left:

        st.caption(
            "Select a sample feedback (optional)"
        )

        sample = st.selectbox(

            "Feedback example",

            [
                "Choose a feedback..."
            ]
            +
            list(
                SAMPLE_FEEDBACK.keys()
            ),

            label_visibility="collapsed"
        )

        if sample != "Choose a feedback...":

            st.session_state.review_text = (
                SAMPLE_FEEDBACK[sample]
            )

    with right:

        st.caption(
            "Type your own review"
        )

        review = st.text_area(

            "Course feedback",

            value=(
                st.session_state.review_text
            ),

            height=110,

            max_chars=1000,

            placeholder=(
                "Enter your course feedback here..."
            ),

            label_visibility="collapsed"
        )

        st.session_state.review_text = review

        st.caption(
            f"{len(review)}/1000"
        )

    compact_divider()

    model_col, recommendation_col, button_col = (
        st.columns(
            [1.15, 1.35, 1]
        )
    )

    with model_col:

        selected_model = st.radio(

            "**Select Model**",

            [
                "Logistic Regression",
                "DistilBERT",
                "Combined"
            ],

            index=2
        )

    with recommendation_col:

        st.info(
            "🔵 **Combined (Recommended)**\n\n"
            "Uses both Logistic Regression "
            "and multilingual DistilBERT.\n"
            "Averages the probability outputs "
            "of the two trained classifiers and "
            "is recommended for the final presentation."
        )

    with button_col:

        st.write("")
        st.write("")

        analyze = st.button(

            "Analyze Review",

            icon=":material/search:",

            type="primary",

            use_container_width=True
        )

    if analyze:

        if not clean_text(review):

            st.warning(
                "Please enter a course review first."
            )

            return

        with st.spinner(
            "Translating and analyzing review..."
        ):

            analysis_text, translation = (
                prepare_feedback_for_analysis(
                    review
                )
            )

            prediction = predict_sentiment(
                analysis_text,
                selected_model
            )

            aspects = (
                calculate_aspect_results(
                    analysis_text
                )
            )

        st.session_state.single_result = {

            "text": review,

            "analysis_text": analysis_text,

            "translation": translation,

            "model": selected_model,

            "prediction": prediction,

            "aspects": aspects
        }


# ============================================================
# SENTIMENT PREDICTION CARD
# ============================================================

def render_sentiment_card(
    result
):

    prediction = result[
        "prediction"
    ]

    probabilities = prediction[
        "probabilities"
    ]

    label = prediction[
        "label"
    ]

    left, right = st.columns(
        [1.2, 1.5]
    )

    with left:

        st.markdown(
            "### Sentiment Prediction"
        )

        st.markdown(
            f"## {sentiment_icon(label)}  "
            f"{display_label(label)}"
        )

        st.write(
            "Confidence: "
            f"**{prediction['confidence']:.2f}**"
        )

    with right:

        rows = [

            (
                "Positive",
                "POSITIVE"
            ),

            (
                "Neutral",
                "NEUTRAL"
            ),

            (
                "Negative",
                "NEGATIVE"
            )
        ]

        for name, key in rows:

            bar_col, value_col = (
                st.columns(
                    [4, 0.7]
                )
            )

            with bar_col:

                st.write(
                    name
                )

                st.progress(
                    probabilities[key]
                )

            with value_col:

                st.write(
                    f"{probabilities[key]:.0%}"
                )


# ============================================================
# ASPECT CARDS
# ============================================================

def render_aspects(
    aspects
):

    st.subheader(
        "🔗  Aspect Analysis"
    )

    st.caption(
        "Key aspects detected in the feedback "
        "and their sentiment"
    )

    if not aspects:

        st.info(
            "No specific course aspects "
            "were detected."
        )

        return

    display_aspects = aspects[:8]

    for start in range(
        0,
        len(display_aspects),
        3
    ):

        row = display_aspects[
            start:start + 3
        ]

        columns = st.columns(
            3
        )

        for index, item in enumerate(
            row
        ):

            with columns[index]:

                with st.container(
                    border=True
                ):

                    aspect = item[
                        "aspect"
                    ]

                    sentiment = item[
                        "sentiment"
                    ]

                    confidence = item[
                        "confidence"
                    ]

                    icon = ASPECT_ICONS.get(
                        aspect,
                        "🔹"
                    )

                    st.write(
                        f"**{icon}  {aspect}**"
                    )

                    circle = (
                        sentiment_circle_html(
                            sentiment
                        )
                    )

                    sentiment_text = (
                        display_label(
                            sentiment
                        )
                    )

                    st.markdown(
                        f"""
                        <div class="sentiment-row">
                            {circle}
                            <span>{sentiment_text}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        confidence
                    )

                    st.caption(
                        f"Confidence: {confidence:.4f}"
                    )


# ============================================================
# MODEL CONFIDENCE
# ============================================================

def render_model_confidence(
    text
):

    st.subheader(
        "📊  Model Confidence"
    )

    logistic = predict_logistic(
        text
    )

    distilbert = predict_distilbert(
        text
    )

    combined = predict_combined(
        text
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(
            "Logistic Regression",
            f"{logistic['confidence']:.2f}"
        )

    with col2:

        st.metric(
            "DistilBERT",
            f"{distilbert['confidence']:.2f}"
        )

    with col3:

        st.metric(
            "Combined",
            f"{combined['confidence']:.2f}",
            delta="Best"
        )


# ============================================================
# WORD IMPORTANCE
# ============================================================

def calculate_word_importance(
    text
):

    text = clean_text(
        text
    )

    if not text:

        return pd.DataFrame(
            columns=[
                "word",
                "value"
            ]
        )

    vector = (
        tfidf_vectorizer
        .transform(
            [text]
        )
    )

    feature_names = (
        tfidf_vectorizer
        .get_feature_names_out()
    )

    row = vector.toarray()[0]

    classes = [

        normalize_label(
            value
        )

        for value in (
            tfidf_model.classes_
        )
    ]

    prediction = predict_logistic(
        text
    )

    target_label = prediction[
        "label"
    ]

    if target_label in classes:

        target_index = classes.index(
            target_label
        )

    else:

        target_index = 0

    coefficient_row = (
        tfidf_model
        .coef_[target_index]
    )

    contributions = (
        row * coefficient_row
    )

    nonzero = np.where(
        row != 0
    )[0]

    data = []

    for index in nonzero:

        data.append({

            "word": feature_names[index],

            "value": float(
                contributions[index]
            )
        })

    if not data:

        return pd.DataFrame(
            columns=[
                "word",
                "value"
            ]
        )

    result = pd.DataFrame(
        data
    )

    result["absolute"] = (
        result["value"]
        .abs()
    )

    result = (
        result
        .sort_values(
            "absolute",
            ascending=False
        )
        .head(12)
        .sort_values(
            "value"
        )
    )

    return result[
        [
            "word",
            "value"
        ]
    ]


# ============================================================
# CREATE EXPLAINABLE AI GRAPH
# ============================================================

def create_explanation_figure(
    text
):

    data = calculate_word_importance(
        text
    )

    if data.empty:

        return None

    fig, ax = plt.subplots(
        figsize=(
            7,
            4.2
        )
    )

    ax.barh(
        data["word"],
        data["value"]
    )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "SHAP Value"
    )

    ax.set_title(
        "Top Contributing Words"
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    plt.tight_layout()

    return fig


# ============================================================
# EXPLAINABLE AI CHART
# ============================================================

def render_explanation_chart(
    text
):

    fig = create_explanation_figure(
        text
    )

    if fig is None:

        st.info(
            "No TF-IDF words were available "
            "for a word-level explanation."
        )

        return

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# CREATE EXPLAINABLE AI PNG
# ============================================================

def create_explanation_png(
    text
):

    fig = create_explanation_figure(
        text
    )

    if fig is None:

        return None

    image_buffer = BytesIO()

    fig.savefig(
        image_buffer,
        format="png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    image_buffer.seek(
        0
    )

    return image_buffer.getvalue()


# ============================================================
# REVIEW REPORT
# ============================================================

def create_review_report(
    result
):

    prediction = result[
        "prediction"
    ]

    lines = [

        "COURSE FEEDBACK SENTIMENT ANALYSIS",

        "Created By Afsah Arshad",

        "=" * 50,

        f"Model: {result['model']}",

        f"Sentiment: {prediction['label']}",

        (
            f"Confidence: "
            f"{prediction['confidence']:.4f}"
        ),

        "",

        "REVIEW",

        result["text"],

        "",

        "LANGUAGE",

        result.get(
            "translation",
            {}
        ).get(
            "language_name",
            "English"
        ),

        "",

        "ENGLISH TEXT USED FOR ANALYSIS",

        result.get(
            "analysis_text",
            result["text"]
        ),

        "",

        "ASPECTS"
    ]

    for item in result[
        "aspects"
    ]:

        lines.append(

            f"- {item['aspect']}: "
            f"{item['sentiment']} "
            f"({item['confidence']:.2f})"
        )

    return "\n".join(
        lines
    )


# ============================================================
# REVIEW CSV
# ============================================================

def create_result_csv(
    result
):

    prediction = result[
        "prediction"
    ]

    dataframe = pd.DataFrame([{

        "Review":
            result["text"],

        "Language":
            result.get(
                "translation",
                {}
            ).get(
                "language_name",
                "English"
            ),

        "English Translation":
            result.get(
                "analysis_text",
                result["text"]
            ),

        "Model":
            result["model"],

        "Sentiment":
            prediction["label"],

        "Confidence":
            prediction["confidence"],

        "Negative Probability":
            prediction[
                "probabilities"
            ]["NEGATIVE"],

        "Neutral Probability":
            prediction[
                "probabilities"
            ]["NEUTRAL"],

        "Positive Probability":
            prediction[
                "probabilities"
            ]["POSITIVE"]
    }])

    return dataframe.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


# ============================================================
# SINGLE REVIEW PAGE
# ============================================================

def render_single_review():

    st.title(
        "💬 Single Review Analysis"
    )

    st.caption(
        "Analyze one course review using "
        "the trained AI models."
    )

    render_review_input()

    result = (
        st.session_state.single_result
    )

    if result is None:

        st.info(
            "Enter a review and select "
            "**Analyze Review** to see the analysis."
        )

        return

    compact_divider()

    st.subheader(
        "📋  Single Review Analysis"
    )

    translation = result.get(
        "translation"
    )

    if translation and translation.get(
        "was_translated"
    ):

        with st.container(
            border=True
        ):

            st.subheader(
                "🌐  English Translation"
            )

            st.caption(
                "Detected language: "
                f"**{translation['language_name']}**"
            )

            st.markdown(
                "**Original Feedback**"
            )

            st.write(
                result["text"]
            )

            st.markdown(
                "**English Translation**"
            )

            st.write(
                translation["translated"]
            )

        compact_divider()

    # --------------------------------------------------------
    # SENTIMENT PREDICTION
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        render_sentiment_card(
            result
        )

    st.write("")

    # --------------------------------------------------------
    # KEY INSIGHT - FULL WIDTH
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "💡  Key Insight"
        )

        insight = create_key_insight(
            result["prediction"]
        )

        st.markdown(
            f'<div class="key-insight-text">{insight}</div>',
            unsafe_allow_html=True
        )

    compact_divider()

    # --------------------------------------------------------
    # ASPECT ANALYSIS
    # MAXIMUM 3 CARDS PER ROW
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        render_aspects(
            result["aspects"]
        )

    compact_divider()

    # --------------------------------------------------------
    # EXPLAINABLE AI - NEXT ROW
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "💡  Explainable AI (SHAP)"
        )

        st.caption(
            "Words that influenced the prediction"
        )

        render_explanation_chart(
            result.get(
                "analysis_text",
                result["text"]
            )
        )

    compact_divider()

    # --------------------------------------------------------
    # MODEL CONFIDENCE + QUICK ACTIONS
    # --------------------------------------------------------

    confidence_col, actions_col = (
        st.columns(
            [1.55, 1]
        )
    )

    with confidence_col:

        with st.container(
            border=True
        ):

            render_model_confidence(
                result.get(
                    "analysis_text",
                    result["text"]
                )
            )

    with actions_col:

        with st.container(
            border=True
        ):

            st.subheader(
                "⚡  Quick Actions"
            )

            report = create_review_report(
                result
            )

            # ------------------------------------------------
            # DOWNLOAD DETAILED REPORT
            # ------------------------------------------------

            st.download_button(

                "Download Detailed Report",

                icon=":material/description:",

                data=report,

                file_name=(
                    "review_analysis.txt"
                ),

                mime="text/plain",

                use_container_width=True
            )

            # ------------------------------------------------
            # DOWNLOAD CSV RESULTS
            # ------------------------------------------------

            st.download_button(

                "Download Results",
        icon=":material/download:",

                data=create_result_csv(
                    result
                ),

                file_name=(
                    "review_result.csv"
                ),

                mime="text/csv",

                use_container_width=True
            )

            # ------------------------------------------------
            # DOWNLOAD EXPLAINABLE AI GRAPH
            # ------------------------------------------------

            explanation_png = (
                create_explanation_png(
                    result.get(
                        "analysis_text",
                        result["text"]
                    )
                )
            )

            if explanation_png is not None:

                st.download_button(

                    "Download Explainable AI Graph",

                    icon=":material/bar_chart:",

                    data=explanation_png,

                    file_name=(
                        "explainable_ai_graph.png"
                    ),

                    mime="image/png",

                    use_container_width=True
                )

    st.info(
        "ℹ️ SHAP shows which words and tokens "
        "contributed to the model's prediction, "
        "helping you understand the reasoning "
        "behind each sentiment."
    )


# ============================================================
# CSV REVIEW COLUMN DETECTION
# ============================================================

def detect_review_column(
    dataframe
):

    preferred_columns = [

        "Review",
        "review",
        "Reviews",
        "reviews",
        "Feedback",
        "feedback",
        "Comment",
        "comment",
        "Text",
        "text"
    ]

    for column in preferred_columns:

        if column in dataframe.columns:

            return column

    for column in dataframe.columns:

        if dataframe[
            column
        ].dtype == "object":

            values = (
                dataframe[column]
                .dropna()
                .astype(str)
            )

            if len(values) == 0:
                continue

            average_length = (
                values
                .str
                .len()
                .mean()
            )

            if average_length > 20:

                return column

    return None


# ============================================================
# CSV COURSE COLUMN DETECTION
# ============================================================

def detect_course_columns(
    dataframe
):

    course_id_columns = [
        "Course ID",
        "CourseId",
        "course_id",
        "courseid",
        "Course_ID",
        "course id",
        "Course Code",
        "CourseCode",
        "course_code",
        "course code",
        "ID"
    ]

    course_name_columns = [
        "Course Name",
        "CourseName",
        "course_name",
        "coursename",
        "Course",
        "course",
        "Course Title",
        "CourseTitle",
        "course_title",
        "course title",
        "Title",
        "title"
    ]

    course_id_column = None
    course_name_column = None

    for column in course_id_columns:

        if column in dataframe.columns:

            course_id_column = column
            break

    for column in course_name_columns:

        if column in dataframe.columns:

            if column != course_id_column:

                course_name_column = column
                break

    return (
        course_id_column,
        course_name_column
    )


# ============================================================
# COURSE LABEL
# ============================================================

def get_course_label(
    row,
    course_id_column,
    course_name_column
):

    course_id = ""

    course_name = ""

    if course_id_column is not None:

        value = row.get(
            course_id_column,
            ""
        )

        if pd.notna(value):

            course_id = clean_text(
                value
            )

    if course_name_column is not None:

        value = row.get(
            course_name_column,
            ""
        )

        if pd.notna(value):

            course_name = clean_text(
                value
            )

    if course_name and course_id:

        return (
            f"{course_name} "
            f"({course_id})"
        )

    if course_name:

        return course_name

    if course_id:

        return course_id

    return "Course Not Specified"


# ============================================================
# FORMAT ASPECT RESULTS
# ============================================================

def format_aspect_results(
    aspect_results
):

    if not aspect_results:

        return ""

    formatted = []

    for item in aspect_results:

        formatted.append(
            f"{item['aspect']} = "
            f"{item['sentiment']} = "
            f"{item['confidence']:.4f}"
        )

    return " || ".join(
        formatted
    )


# ============================================================
# CSV ANALYSIS
# ============================================================

def analyze_csv(
    dataframe,
    model_name
):

    review_column = (
        detect_review_column(
            dataframe
        )
    )

    if review_column is None:

        return (
            None,
            None
        )

    (
        course_id_column,
        course_name_column
    ) = detect_course_columns(
        dataframe
    )

    # Prepare all reviews first so non-English feedback is
    # translated locally in language batches rather than by
    # repeatedly calling an external translation service.
    review_indices = []
    review_texts = []

    for index, (_, source_row) in enumerate(
        dataframe.iterrows()
    ):

        review = clean_text(
            source_row[
                review_column
            ]
        )

        if review:

            review_indices.append(index)
            review_texts.append(review)

    try:

        translation_results = (
            translate_feedback_batch(
                review_texts
            )
        )

    except Exception as error:

        st.error(
            "Multilingual translation could not be completed."
        )

        st.code(
            str(error)
        )

        return (
            None,
            review_column
        )

    translation_by_index = {
        index: translation
        for index, translation in zip(
            review_indices,
            translation_results
        )
    }

    rows = []

    total = len(
        dataframe
    )

    progress = st.progress(
        0
    )

    for index, (_, source_row) in enumerate(
        dataframe.iterrows()
    ):

        review = clean_text(
            source_row[
                review_column
            ]
        )

        if review:

            try:

                translation = (
                    translation_by_index[index]
                )

                analysis_text = (
                    translation["translated"]
                )

                prediction = (
                    predict_sentiment(
                        analysis_text,
                        model_name
                    )
                )

                aspect_results = (
                    calculate_aspect_results(
                        analysis_text
                    )
                )

                rows.append({

                    "Course":
                        get_course_label(
                            source_row,
                            course_id_column,
                            course_name_column
                        ),

                    "Review":
                        review,

                    "Language":
                        translation[
                            "language_name"
                        ],

                    "English Translation":
                        analysis_text,

                    "Sentiment":
                        prediction[
                            "label"
                        ],

                    "Confidence":
                        prediction[
                            "confidence"
                        ],

                    "Negative":
                        prediction[
                            "probabilities"
                        ]["NEGATIVE"],

                    "Neutral":
                        prediction[
                            "probabilities"
                        ]["NEUTRAL"],

                    "Positive":
                        prediction[
                            "probabilities"
                        ]["POSITIVE"],

                    "Detected Aspects":
                        ", ".join(
                            item["aspect"]
                            for item in aspect_results
                        ),

                    "Aspect Analysis":
                        format_aspect_results(
                            aspect_results
                        )
                })

            except Exception:

                pass

        if (
            index % 5 == 0
            or index == total - 1
        ):

            progress.progress(
                (index + 1)
                /
                max(total, 1)
            )

    progress.empty()

    if not rows:

        return (
            None,
            review_column
        )

    return (
        pd.DataFrame(
            rows
        ),
        review_column
    )


# ============================================================
# BUILD ASPECT DATAFRAME FROM CSV RESULTS
# ============================================================

def build_csv_aspect_dataframe(
    results
):

    aspect_rows = []

    for _, row in results.iterrows():

        course = row.get(
            "Course",
            "Course Not Specified"
        )

        review = row.get(
            "Review",
            ""
        )

        aspect_text = row.get(
            "Aspect Analysis",
            ""
        )

        if not isinstance(
            aspect_text,
            str
        ) or not aspect_text.strip():

            continue

        entries = aspect_text.split(
            " || "
        )

        for entry in entries:

            parts = entry.split(
                " = "
            )

            if len(parts) != 3:

                continue

            aspect = parts[0].strip()

            sentiment = parts[1].strip()

            try:

                confidence = float(
                    parts[2].strip()
                )

            except Exception:

                confidence = 0.0

            aspect_rows.append({

                "Course":
                    course,

                "Review":
                    review,

                "Aspect":
                    aspect,

                "Sentiment":
                    sentiment,

                "Confidence":
                    confidence
            })

    if not aspect_rows:

        return pd.DataFrame(
            columns=[
                "Course",
                "Review",
                "Aspect",
                "Sentiment",
                "Confidence"
            ]
        )

    return pd.DataFrame(
        aspect_rows
    )


# ============================================================
# CSV ASPECT SUMMARY
# ============================================================

def create_csv_aspect_summary(
    aspect_dataframe
):

    if aspect_dataframe.empty:

        return pd.DataFrame(
            columns=[
                "Aspect",
                "Reviews",
                "Positive",
                "Neutral",
                "Negative",
                "Overall Sentiment"
            ]
        )

    summary_rows = []

    for aspect, group in (
        aspect_dataframe
        .groupby("Aspect")
    ):

        counts = (
            group[
                "Sentiment"
            ]
            .value_counts()
        )

        positive = int(
            counts.get(
                "POSITIVE",
                0
            )
        )

        neutral = int(
            counts.get(
                "NEUTRAL",
                0
            )
        )

        negative = int(
            counts.get(
                "NEGATIVE",
                0
            )
        )

        if (
            positive >= neutral
            and positive >= negative
        ):

            overall = "POSITIVE"

        elif negative >= positive and negative >= neutral:

            overall = "NEGATIVE"

        else:

            overall = "NEUTRAL"

        summary_rows.append({

            "Aspect":
                aspect,

            "Reviews":
                len(group),

            "Positive":
                positive,

            "Neutral":
                neutral,

            "Negative":
                negative,

            "Overall Sentiment":
                overall
        })

    return (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "Reviews",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# COURSE-WISE ASPECT SUMMARY
# ============================================================

def create_course_aspect_summary(
    aspect_dataframe
):

    if aspect_dataframe.empty:

        return pd.DataFrame(
            columns=[
                "Course",
                "Aspect",
                "Reviews",
                "Positive",
                "Neutral",
                "Negative",
                "Overall Sentiment"
            ]
        )

    summary_rows = []

    grouped = (
        aspect_dataframe
        .groupby(
            [
                "Course",
                "Aspect"
            ]
        )
    )

    for (
        course,
        aspect
    ), group in grouped:

        counts = (
            group[
                "Sentiment"
            ]
            .value_counts()
        )

        positive = int(
            counts.get(
                "POSITIVE",
                0
            )
        )

        neutral = int(
            counts.get(
                "NEUTRAL",
                0
            )
        )

        negative = int(
            counts.get(
                "NEGATIVE",
                0
            )
        )

        if (
            positive >= neutral
            and positive >= negative
        ):

            overall = "POSITIVE"

        elif (
            negative >= positive
            and negative >= neutral
        ):

            overall = "NEGATIVE"

        else:

            overall = "NEUTRAL"

        summary_rows.append({

            "Course":
                course,

            "Aspect":
                aspect,

            "Reviews":
                len(group),

            "Positive":
                positive,

            "Neutral":
                neutral,

            "Negative":
                negative,

            "Overall Sentiment":
                overall
        })

    return (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "Course",
                "Reviews"
            ],
            ascending=[
                True,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CSV RESULTS
# ============================================================

def render_csv_results(
    results
):

    compact_divider()

    st.subheader(
        "📊 Overall Sentiment Analysis"
    )

    counts = (
        results[
            "Sentiment"
        ]
        .value_counts()
    )

    positive = int(
        counts.get(
            "POSITIVE",
            0
        )
    )

    neutral = int(
        counts.get(
            "NEUTRAL",
            0
        )
    )

    negative = int(
        counts.get(
            "NEGATIVE",
            0
        )
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:

        st.metric(
            "Positive",
            f"{positive:,}"
        )

    with c2:

        st.metric(
            "Neutral",
            f"{neutral:,}"
        )

    with c3:

        st.metric(
            "Negative",
            f"{negative:,}"
        )

    chart_data = pd.DataFrame({

        "Sentiment": [

            "Positive",
            "Neutral",
            "Negative"
        ],

        "Count": [

            positive,
            neutral,
            negative
        ]
    })

    # Sentiment colors: Positive = green, Neutral = yellow, Negative = red
    fig, ax = plt.subplots(figsize=(8, 4))

    sentiment_labels = [
        "Positive",
        "Neutral",
        "Negative"
    ]

    sentiment_values = [
        positive,
        neutral,
        negative
    ]

    sentiment_colors = [
        "green",
        "gold",
        "red"
    ]

    ax.bar(
        sentiment_labels,
        sentiment_values,
        color=sentiment_colors
    )

    ax.set_ylabel("Count")
    ax.set_title("Overall Sentiment Distribution")
    ax.grid(axis="y", alpha=0.2)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # --------------------------------------------------------
    # CSV ASPECT ANALYSIS
    # --------------------------------------------------------

    aspect_dataframe = (
        build_csv_aspect_dataframe(
            results
        )
    )

    if not aspect_dataframe.empty:

        compact_divider()

        st.subheader(
            "🔗 Aspect-Based Sentiment Analysis"
        )

        st.caption(
            "Shows which course-related aspects "
            "are discussed in the feedback and "
            "whether students feel positively, "
            "neutrally, or negatively about them."
        )

        aspect_summary = (
            create_csv_aspect_summary(
                aspect_dataframe
            )
        )

        st.dataframe(
            aspect_summary,
            use_container_width=True,
            hide_index=True
        )

        aspect_chart = (
            aspect_summary
            .set_index("Aspect")[
                [
                    "Positive",
                    "Neutral",
                    "Negative"
                ]
            ]
        )

        # Fixed sentiment colors for aspect chart
        fig, ax = plt.subplots(figsize=(10, 5))

        aspect_chart.plot(
            kind="bar",
            ax=ax,
            color=["green", "gold", "red"]
        )

        ax.set_xlabel("Aspect")
        ax.set_ylabel("Count")
        ax.set_title("Aspect Sentiment Distribution")
        ax.legend(title="Sentiment")
        ax.grid(axis="y", alpha=0.2)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        # Download the overall aspect analysis data and graph
        aspect_csv = aspect_summary.to_csv(
            index=False
        ).encode("utf-8")

        aspect_graph_buffer = BytesIO()
        fig.savefig(
            aspect_graph_buffer,
            format="png",
            dpi=300,
            bbox_inches="tight"
        )
        aspect_graph_buffer.seek(0)

        aspect_download_col1, aspect_download_col2 = st.columns(2)

        with aspect_download_col1:
            st.download_button(
                "Download Overall Aspect Analysis",
                icon=":material/download:",
                data=aspect_csv,
                file_name="overall_aspect_sentiment_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

        with aspect_download_col2:
            st.download_button(
                "Download Aspect Graph",
                icon=":material/image:",
                data=aspect_graph_buffer.getvalue(),
                file_name="overall_aspect_sentiment_graph.png",
                mime="image/png",
                use_container_width=True
            )

        plt.close(fig)

        compact_divider()

        st.subheader(
            "🎓 Course-Wise Aspect Analysis"
        )

        st.caption(
            "Aspect sentiment is grouped using "
            "the Course Name or Course ID available "
            "in the uploaded CSV."
        )

        course_summary = (
            create_course_aspect_summary(
                aspect_dataframe
            )
        )

        available_courses = sorted(
            course_summary[
                "Course"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # Download the complete course-wise aspect analysis
        st.download_button(
            "Download All Course-Wise Aspect Analysis",
            icon=":material/download:",
            data=course_summary.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="course_wise_aspect_sentiment_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

        if available_courses:

            selected_course = st.selectbox(
                "Select Course",
                available_courses
            )

            selected_course_summary = (
                course_summary[
                    course_summary[
                        "Course"
                    ].astype(str)
                    == str(selected_course)
                ]
            )

            st.dataframe(
                selected_course_summary,
                use_container_width=True,
                hide_index=True
            )

            course_chart = (
                selected_course_summary
                .set_index("Aspect")[
                    [
                        "Positive",
                        "Neutral",
                        "Negative"
                    ]
                ]
            )

            # Fixed sentiment colors for course-wise aspect chart
            fig, ax = plt.subplots(figsize=(10, 5))

            course_chart.plot(
                kind="bar",
                ax=ax,
                color=["green", "gold", "red"]
            )

            ax.set_xlabel("Aspect")
            ax.set_ylabel("Count")
            ax.set_title(
                f"Aspect Sentiment for {selected_course}"
            )
            ax.legend(title="Sentiment")
            ax.grid(axis="y", alpha=0.2)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            # Download the selected course analysis and graph
            selected_course_csv = selected_course_summary.to_csv(
                index=False
            ).encode("utf-8")

            course_graph_buffer = BytesIO()
            fig.savefig(
                course_graph_buffer,
                format="png",
                dpi=300,
                bbox_inches="tight"
            )
            course_graph_buffer.seek(0)

            course_download_col1, course_download_col2 = st.columns(2)

            with course_download_col1:
                st.download_button(
                    "Download Selected Course Analysis",
                    icon=":material/download:",
                    data=selected_course_csv,
                    file_name="selected_course_aspect_sentiment_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with course_download_col2:
                st.download_button(
                    "Download Course Graph",
                    icon=":material/image:",
                    data=course_graph_buffer.getvalue(),
                    file_name="selected_course_aspect_sentiment_graph.png",
                    mime="image/png",
                    use_container_width=True
                )

            plt.close(fig)

        compact_divider()

        st.subheader(
            "📋 Detailed Aspect Results"
        )

        detailed_aspects = (
            aspect_dataframe[
                [
                    "Course",
                    "Aspect",
                    "Sentiment",
                    "Confidence",
                    "Review"
                ]
            ]
            .copy()
        )

        st.dataframe(
            detailed_aspects,
            use_container_width=True,
            hide_index=True
        )

        # Download detailed aspect-level results
        st.download_button(
            "Download Detailed Aspect Results",
            icon=":material/download:",
            data=detailed_aspects.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="detailed_aspect_sentiment_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    compact_divider()

    st.subheader(
        "📋 Analysis Results"
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️  Download Results",
        data=results.to_csv(
            index=False
        ).encode(
            "utf-8"
        ),
        file_name=(
            "course_feedback_sentiment_results.csv"
        ),
        mime="text/csv",
        type="primary",
        use_container_width=True
    )


# ============================================================
# CSV PAGE
# ============================================================

def render_csv_analysis():

    st.title(
        "📄 CSV File Analysis"
    )

    st.caption(
        "Analyze sentiment distribution "
        "across a complete dataset."
    )

    with st.container(
        border=True
    ):

        st.write(
            "### 📁 Upload Course Feedback"
        )

        uploaded_file = (
            st.file_uploader(
                "Upload CSV File",
                type=["csv"],
                help=(
                    "CSV should contain a course "
                    "review or feedback text column."
                )
            )
        )

        st.info(
            "Your CSV can contain additional columns "
            "such as CourseId, rating, date, or "
            "other metadata."
        )

    if uploaded_file is None:

        return

    try:

        dataframe = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.error(
            "Could not read the CSV file."
        )

        st.code(
            str(error)
        )

        return

    review_column = (
        detect_review_column(
            dataframe
        )
    )

    if review_column is None:

        st.error(
            "No suitable review or feedback "
            "column was detected."
        )

        st.write(
            "Available columns:"
        )

        st.write(
            list(
                dataframe.columns
            )
        )

        return

    st.success(
        f"Detected review column: "
        f"**{review_column}**"
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:

        st.metric(
            "Rows",
            f"{len(dataframe):,}"
        )

    with c2:

        st.metric(
            "Columns",
            len(
                dataframe.columns
            )
        )

    with c3:

        st.metric(
            "Review Column",
            review_column
        )

    compact_divider()

    selected_model = st.radio(

        "Select Model for Analysis",

        [
            "Logistic Regression",
            "DistilBERT",
            "Combined"
        ],

        index=2,

        horizontal=True
    )

    if st.button(

        "Analyze CSV",

        icon=":material/search:",

        type="primary",

        use_container_width=True
    ):

        if (
            len(dataframe) > 10000
            and selected_model
            in {
                "DistilBERT",
                "Combined"
            }
        ):

            st.warning(
                f"This file contains "
                f"{len(dataframe):,} rows. "
                "DistilBERT analysis can take "
                "considerable time on a CPU."
            )

        with st.spinner(
            "Analyzing CSV file..."
        ):

            results, detected_column = (
                analyze_csv(
                    dataframe,
                    selected_model
                )
            )

        if results is None:

            st.error(
                "No valid reviews could be analyzed."
            )

            return

        st.session_state.csv_results = (
            results
        )

        st.session_state.csv_model = (
            selected_model
        )

        st.session_state.csv_review_column = (
            detected_column
        )

        st.success(
            f"Analysis completed for "
            f"{len(results):,} reviews."
        )

    results = (
        st.session_state.csv_results
    )

    if results is None:

        return

    render_csv_results(
        results
    )


# ============================================================
# ASPECT ANALYSIS PAGE
# ============================================================

def render_aspect_page():

    st.title(
        "🔗 Aspect Analysis"
    )

    st.caption(
        "Identify important course-related "
        "aspects and their sentiment."
    )

    text = st.text_area(

        "Enter course feedback",

        height=150,

        placeholder=(
            "Example: The instructor was excellent, "
            "but the assignments were difficult."
        )
    )

    if st.button(

        "Analyze Aspects",

        icon=":material/search:",

        type="primary",

        use_container_width=True
    ):

        if not clean_text(text):

            st.warning(
                "Please enter a review first."
            )

            return

        with st.spinner(
            "Translating and analyzing aspects..."
        ):

            analysis_text, translation = (
                prepare_feedback_for_analysis(
                    text
                )
            )

            results = (
                calculate_aspect_results(
                    analysis_text
                )
            )

        if translation.get(
            "was_translated"
        ):

            with st.container(
                border=True
            ):

                st.subheader(
                    "🌐  English Translation"
                )

                st.caption(
                    "Detected language: "
                    f"**{translation['language_name']}**"
                )

                st.markdown(
                    "**Original Feedback**"
                )

                st.write(
                    text
                )

                st.markdown(
                    "**English Translation**"
                )

                st.write(
                    analysis_text
                )

            compact_divider()

        if not results:

            st.info(
                "No specific course aspects "
                "were detected."
            )

            return

        st.subheader(
            "Detected Aspects"
        )

        for start in range(
            0,
            len(results),
            3
        ):

            row = results[
                start:start + 3
            ]

            columns = st.columns(
                3
            )

            for index, item in enumerate(
                row
            ):

                with columns[index]:

                    with st.container(
                        border=True
                    ):

                        icon = ASPECT_ICONS.get(
                            item["aspect"],
                            "🔹"
                        )

                        st.write(
                            f"**{icon} "
                            f"{item['aspect']}**"
                        )

                        circle = (
                            sentiment_circle_html(
                                item["sentiment"]
                            )
                        )

                        sentiment_text = (
                            display_label(
                                item["sentiment"]
                            )
                        )

                        st.markdown(
                            f"""
                            <div class="sentiment-row">
                                {circle}
                                <span>{sentiment_text}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.progress(
                            item["confidence"]
                        )

                        st.caption(
                            "Confidence: "
                            f"{item['confidence']:.4f}"
                        )

        compact_divider()

        st.subheader(
            "Aspect Evidence"
        )

        evidence = pd.DataFrame(
            results
        )[

            [
                "aspect",
                "sentiment",
                "confidence",
                "sentence"
            ]
        ]

        evidence.columns = [

            "Aspect",
            "Sentiment",
            "Confidence",
            "Supporting Text"
        ]

        st.dataframe(

            evidence,

            use_container_width=True,

            hide_index=True
        )


# ============================================================
# EXPLAINABLE AI PAGE
# ============================================================

def render_shap_page():

    st.title(
        "💡 Explainable AI (SHAP)"
    )

    st.caption(
        "Understand which words influence "
        "the sentiment prediction."
    )

    text = st.text_area(

        "Enter course feedback",

        height=150,

        placeholder=(
            "Enter a review to generate "
            "word-level explanations."
        )
    )

    if st.button(

        "Generate Explanation",

        icon=":material/psychology:",

        type="primary",

        use_container_width=True
    ):

        if not clean_text(text):

            st.warning(
                "Please enter a review first."
            )

            return

        with st.spinner(
            "Translating review and generating explanation..."
        ):

            analysis_text, translation = (
                prepare_feedback_for_analysis(
                    text
                )
            )

            prediction = predict_logistic(
                analysis_text
            )

        if translation.get(
            "was_translated"
        ):

            with st.container(
                border=True
            ):

                st.subheader(
                    "🌐  English Translation"
                )

                st.caption(
                    "Detected language: "
                    f"**{translation['language_name']}**"
                )

                st.markdown(
                    "**Original Feedback**"
                )

                st.write(
                    text
                )

                st.markdown(
                    "**English Translation**"
                )

                st.write(
                    analysis_text
                )

            compact_divider()

        c1, c2 = (
            st.columns(2)
        )

        with c1:

            st.metric(

                "Sentiment",

                display_label(
                    prediction["label"]
                )
            )

        with c2:

            st.metric(

                "Confidence",

                f"{prediction['confidence']:.2%}"
            )

        compact_divider()

        st.subheader(
            "Top Contributing Words"
        )

        render_explanation_chart(
            analysis_text
        )

        st.info(
            "ℹ️ Positive values push the prediction "
            "toward the selected sentiment class, "
            "while negative values push it in the "
            "opposite direction."
        )


# ============================================================
# ABOUT PAGE
# ============================================================

def render_about():

    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    st.subheader(
        "ℹ️ About This Project "
    )

    st.write(
        "Course Feedback Sentiment Analysis is an "
        "AI-powered educational analytics application "
        "designed to transform student feedback into "
        "clear and actionable insights."
    )

    st.write(
        "The application analyzes individual reviews "
        "as well as complete CSV datasets, identifies "
        "important course-related aspects, determines "
        "their sentiment, and provides model-level "
        "explanations to help understand the prediction."
    )

    compact_divider()

    # --------------------------------------------------------
    # APPLICATION CAPABILITIES
    # --------------------------------------------------------

    st.subheader(
        "✨ What this application provides"
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="capability-heading">💬 Single Review</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                • Analyze one student review<br>
                • Identify overall sentiment<br>
                • Classify as Positive, Neutral, or Negative
                """,
                unsafe_allow_html=True
            )

    with c2:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="capability-heading">📄 CSV Analysis</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                • Analyze complete feedback datasets<br>
                • Detect a suitable review column<br>
                • Perform sentiment analysis at scale
                """,
                unsafe_allow_html=True
            )

    with c3:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="capability-heading">🔗 Aspect Analysis</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                • Identify important course-related aspects<br>
                • Determine sentiment for each aspect<br>
                • Covers instructor, content, assignments, difficulty, and platform etc.
                """,
                unsafe_allow_html=True
            )

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="capability-heading">💡 Explainable AI</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                • Explain sentiment predictions<br>
                • Identify influential words<br>
                • Show word-level model contributions
                """,
                unsafe_allow_html=True
            )

    with c2:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="capability-heading">📥 Downloadable Results</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                • Export analysis results<br>
                • Download detailed reports and CSV results<br>
                • Save Explainable AI graphs as PNG
                """,
                unsafe_allow_html=True
            )

    compact_divider()

    # --------------------------------------------------------
    # MACHINE LEARNING MODELS
    # --------------------------------------------------------

    st.subheader(
        "🤖 Machine Learning Models"
    )

    model_1, model_2, model_3 = (
        st.columns(3)
    )

    with model_1:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="model-card-title">🔵 TF-IDF + Logistic Regression</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                • TF-IDF text feature extraction<br>
                • Logistic Regression classification<br>
                • 3 sentiment classes<br>
                • Word-level feature contributions<br>
                • Supports Explainable AI
                """,
                unsafe_allow_html=True
            )

    with model_2:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="model-card-title">🔵 Fine-tuned Multilingual DistilBERT</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                • Multilingual DistilBERT transformer<br>
                • Fine-tuned for course feedback<br>
                • Captures contextual relationships<br>
                • 3 sentiment classes<br>
                • Student feedback analysis
                """,
                unsafe_allow_html=True
            )

    with model_3:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="model-card-title">🔵 Combined Model</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                • Combines Logistic Regression + DistilBERT<br>
                • Uses both models for prediction<br>
                • Averages probability outputs<br>
                • Traditional + transformer approaches<br>
                • Recommended for final presentation
                """,
                unsafe_allow_html=True
            )

    compact_divider()

    # --------------------------------------------------------
    # SENTIMENT + ASPECT ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "🎯 Sentiment and Aspect Analysis"
    )

    # Sentiment Detection - first row
    with st.container(
        border=True
    ):
        st.write(
            "### 🎯 Sentiment Detection"
        )

        st.write(
            "Each review is classified into one of three sentiment categories:"
        )

        st.markdown(
            """
            <div style="display:flex; gap:10px; flex-wrap:wrap; padding-bottom:14px;">
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:7px 14px;">😊 Positive</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:7px 14px;">😐 Neutral</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:7px 14px;">☹️ Negative</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Course Aspects - next row
    with st.container(
        border=True
    ):
        st.write(
            "### 🔗 Course Aspects"
        )

        st.write(
            "The application can identify aspects such as:"
        )

        st.markdown(
            """
            <div style="display:flex; gap:8px; flex-wrap:wrap; line-height:1.8; padding-bottom:16px;">
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Course Content</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Instructor</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Assignments</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Quizzes &amp; Assessments</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Difficulty</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Learning Experience</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Course Structure</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Platform</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Video &amp; Audio</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Certificates</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Duration</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Value</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Practical Application</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Relevance</span>
                <span style="border:1px solid rgba(128,128,128,0.35); border-radius:8px; padding:5px 10px; display:inline-block;">Overall Experience</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    compact_divider()

    # --------------------------------------------------------
    # APPLICATION WORKFLOW
    # --------------------------------------------------------

    st.subheader(
        "⚙️ How the Application Works"
    )

    with st.container(
        border=True
    ):

        st.markdown(
            "**1️⃣ Enter or upload feedback:**  "
            "Analyze a single review or upload a CSV file containing multiple course reviews.",
            unsafe_allow_html=True
        )

        st.markdown(
            "**2️⃣ Translate when needed:**  "
            "Non-English feedback is detected and translated locally into English using a Hugging Face NLLB model before the existing analysis pipeline runs.",
            unsafe_allow_html=True
        )

        st.markdown(
            "**3️⃣ Select an AI model:**  "
            "Choose Logistic Regression, DistilBERT, or the recommended Combined model.",
            unsafe_allow_html=True
        )

        st.markdown(
            "**4️⃣ Generate sentiment predictions:**  "
            "The selected model determines the overall sentiment and confidence.",
            unsafe_allow_html=True
        )

        st.markdown(
            "**5️⃣ Discover course aspects:**  "
            "The application identifies relevant aspects and evaluates their individual sentiment.",
            unsafe_allow_html=True
        )

        st.markdown(
            "**6️⃣ Understand the prediction:**  "
            "Explainable AI highlights words that contributed to the model prediction.",
            unsafe_allow_html=True
        )

        st.markdown(
            "**7️⃣ Download the results:**  "
            "Reports, CSV results, and Explainable AI graphs can be exported for further use.",
            unsafe_allow_html=True
        )

    compact_divider()

    # --------------------------------------------------------
    # PROJECT VALUE
    # --------------------------------------------------------

    st.subheader(
        "📈 Why This Application Is Useful"
    )

    st.info(
        "🎓 **Turning student feedback into actionable insights.**\n\n"
        "Instead of manually reviewing large volumes of "
        "course feedback, educators and course teams can "
        "quickly identify overall sentiment, discover "
        "specific areas that students appreciate or "
        "struggle with, and understand the factors "
        "influencing model predictions."
    )

    compact_divider()

    # --------------------------------------------------------
    # PROJECT FEATURES
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "📊 Application Features"
        )

        st.write(
            "💬 Single Review Analysis"
        )

        st.write(
            "📄 CSV File Analysis"
        )

        st.write(
            "🌐 Multilingual Input & Local Translation"
        )

        st.write(
            "🔗 Aspect Analysis"
        )

        st.write(
            "💡 Explainable AI"
        )

        st.write(
            "📊 Model Confidence"
        )

        st.write(
            "📥 Downloadable Results"
        )

        st.write(
            "🖥️ CPU Compatible"
        )

    compact_divider()

    st.info(
        "**Created By Afsah Arshad**"
    )


# ============================================================
# FOOTER
# ============================================================

def render_footer():

    compact_divider()

    st.caption(
        "🎓 Course Feedback Sentiment Analysis  •  "
        "AI-Powered Education Analytics  •  "
        "Created By Afsah Arshad"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

render_sidebar()

render_header()

current_page = (
    st.session_state.active_page
)


if current_page == "Single Review Analysis":

    render_single_review()


elif current_page == "CSV Analysis":

    render_csv_analysis()


elif current_page == "Aspect Analysis":

    render_aspect_page()


elif current_page == "Explainable AI (SHAP)":

    render_shap_page()


elif current_page == "About":

    render_about()


else:

    render_single_review()


render_footer()