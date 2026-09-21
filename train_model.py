"""
Coursera Course Feedback Sentiment Analysis
--------------------------------------------

This script:

1. Loads the real Coursera course reviews dataset.
2. Removes missing and duplicate reviews.
3. Samples 2,000 reviews to determine the language distribution.
4. Automatically detects languages in the sample.
5. Reports the language distribution.
6. Detects language for the full dataset only when translation is needed.
7. Keeps English reviews unchanged.
8. Translates non-English reviews into English.
9. Saves the translated dataset.
10. Trains TF-IDF + Logistic Regression using English reviews.
11. Evaluates the model using a real 80/20 split.
12. Saves the trained model, vectorizer and metrics.

Sentiment labels:
    1-2 -> NEGATIVE
    3   -> NEUTRAL
    4-5 -> POSITIVE

No evaluation numbers are hardcoded.
"""

import json
import os
import joblib
import pandas as pd

from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,accuracy_score)


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "reviews_by_course.csv"

TRANSLATED_FILE = "reviews_by_course_english.csv"

MODEL_FILE = "coursera_tfidf_logistic_model.pkl"

VECTORIZER_FILE = "coursera_tfidf_vectorizer.pkl"

METRICS_FILE = "metrics.json"

# Number of reviews used for the initial language analysis
LANGUAGE_SAMPLE_SIZE = 2000

# Set this to True if you want to translate non-English reviews.
# If you only want to inspect the language distribution first,
# change this to False.
TRANSLATE_NON_ENGLISH = True

# Make language detection reproducible
DetectorFactory.seed = 42


# ============================================================
# Step 1: Load real dataset
# ============================================================

print("\n========================================")
print("STEP 1: LOADING DATASET")
print("========================================")

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)


# Keep only rows where Review and Label are available
df = df.dropna(
    subset=["Review", "Label"]
).reset_index(drop=True)

print(
    "Dataset shape after removing missing reviews:",
    df.shape
)


# ============================================================
# Step 2: Remove duplicate reviews
# ============================================================

print("\n========================================")
print("STEP 2: REMOVING DUPLICATES")
print("========================================")

before = len(df)

if "CourseId" in df.columns:

    df = df.drop_duplicates(
        subset=["CourseId", "Review", "Label"]
    ).reset_index(drop=True)

else:

    df = df.drop_duplicates(
        subset=["Review", "Label"]
    ).reset_index(drop=True)

after = len(df)

print("Duplicate rows removed:", before - after)


# ============================================================
# Step 3: Clean review text
# ============================================================

print("\n========================================")
print("STEP 3: CLEANING REVIEW TEXT")
print("========================================")

df["Review"] = (
    df["Review"]
    .astype(str)
    .str.strip()
)

df = df[
    df["Review"].str.len() > 0
].reset_index(drop=True)

print(
    "Dataset after text cleaning:",
    df.shape
)


# ============================================================
# Step 4: Check whether translated dataset already exists
# ============================================================

print("\n========================================")
print("STEP 4: CHECKING TRANSLATED DATASET")
print("========================================")


if os.path.exists(TRANSLATED_FILE):

    print(
        f"Found existing file: {TRANSLATED_FILE}"
    )

    try:

        translated_df = pd.read_csv(
            TRANSLATED_FILE
        )

        required_columns = [
            "Review",
            "Review_English",
            "language"
        ]

        if all(
            column in translated_df.columns
            for column in required_columns
        ):

            df = translated_df

            print(
                "Existing translated dataset loaded."
            )

            print(
                "Translation will NOT be repeated."
            )

            TRANSLATED_DATASET_EXISTS = True

        else:

            print(
                "Existing translated file does not "
                "contain the required columns."
            )

            print(
                "A new translation process will be started."
            )

            TRANSLATED_DATASET_EXISTS = False

    except Exception as e:

        print(
            "Could not load existing translated dataset:"
        )

        print(e)

        TRANSLATED_DATASET_EXISTS = False

else:

    print(
        "No existing translated dataset found."
    )

    TRANSLATED_DATASET_EXISTS = False


# ============================================================
# Step 5: Analyze language distribution using a sample
# ============================================================

print("\n========================================")
print("STEP 5: SAMPLE LANGUAGE ANALYSIS")
print("========================================")

print(
    f"Analyzing a sample of up to "
    f"{LANGUAGE_SAMPLE_SIZE} reviews..."
)


def detect_language(text):
    """
    Detect the language of a review.

    Returns:
        ISO language code such as:
        en = English
        es = Spanish
        fr = French
        de = German

    Returns 'unknown' if detection fails.
    """

    try:

        if not isinstance(text, str):
            return "unknown"

        text = text.strip()

        if not text:
            return "unknown"

        return detect(text)

    except Exception:

        return "unknown"


# Only analyze a sample first
sample_size = min(
    LANGUAGE_SAMPLE_SIZE,
    len(df)
)

sample_df = df.sample(
    n=sample_size,
    random_state=42
).copy()


sample_df["sample_language"] = (
    sample_df["Review"]
    .apply(detect_language)
)


language_counts = (
    sample_df["sample_language"]
    .value_counts()
)


print("\nLanguage distribution in sample:")

print(language_counts)


print("\nLanguage percentages:")

language_percentages = (
    sample_df["sample_language"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(language_percentages)


# ============================================================
# Step 6: Decide whether translation is required
# ============================================================

print("\n========================================")
print("STEP 6: LANGUAGE ANALYSIS RESULT")
print("========================================")


english_count = (
    sample_df["sample_language"] == "en"
).sum()


non_english_count = (
    sample_df["sample_language"] != "en"
).sum()


english_percentage = (
    english_count / len(sample_df)
) * 100


non_english_percentage = (
    non_english_count / len(sample_df)
) * 100


print(
    f"English reviews in sample: "
    f"{english_count}"
)

print(
    f"Non-English reviews in sample: "
    f"{non_english_count}"
)

print(
    f"English percentage: "
    f"{english_percentage:.2f}%"
)

print(
    f"Non-English percentage: "
    f"{non_english_percentage:.2f}%"
)


if non_english_count == 0:

    print(
        "\nNo non-English reviews were detected "
        "in the sample."
    )

    print(
        "Translation will not be performed."
    )

    TRANSLATE_NON_ENGLISH = False

else:

    print(
        "\nNon-English reviews were detected."
    )

    if TRANSLATE_NON_ENGLISH:

        print(
            "Translation of non-English reviews "
            "will be performed."
        )

    else:

        print(
            "Translation is currently disabled."
        )


# ============================================================
# Step 7: Translate non-English reviews
# ============================================================

if (
    TRANSLATE_NON_ENGLISH
    and not TRANSLATED_DATASET_EXISTS
):

    print("\n========================================")
    print("STEP 7: TRANSLATING NON-ENGLISH REVIEWS")
    print("========================================")

    print(
        "Language detection is now being performed "
        "on the full dataset."
    )

    print(
        "This step can take some time because "
        "the dataset is large."
    )

    # Detect language for the full dataset
    df["language"] = (
        df["Review"]
        .apply(detect_language)
    )

    print("\nFull dataset language distribution:")

    print(
        df["language"]
        .value_counts()
    )


    # Create translator
    translator = GoogleTranslator(
        source="auto",
        target="en"
    )


    # Cache translations so identical reviews
    # are translated only once.
    translation_cache = {}


    def translate_review(text, language):
        """
        Translate only non-English reviews.

        English reviews are returned unchanged.

        A cache is used so duplicate review text
        does not trigger multiple translation requests.
        """

        if not isinstance(text, str):
            return text

        text = text.strip()

        if not text:
            return text

        # English does not need translation
        if language == "en":
            return text

        # Check cache
        if text in translation_cache:
            return translation_cache[text]

        try:

            translated = translator.translate(text)

            if (
                translated is not None
                and isinstance(translated, str)
                and translated.strip()
            ):

                translation_cache[text] = (
                    translated.strip()
                )

                return translated.strip()

            # If translation returned nothing,
            # keep original text.
            translation_cache[text] = text

            return text

        except Exception as e:

            print(
                "\nTranslation failed."
            )

            print(
                "Language:",
                language
            )

            print(
                "Review:",
                text[:150]
            )

            print(
                "Error:",
                e
            )

            # Keep original review so
            # the data is not lost.
            translation_cache[text] = text

            return text


    # --------------------------------------------------------
    # Translate reviews
    # --------------------------------------------------------

    total_reviews = len(df)

    translated_counter = 0

    print(
        f"\nTotal reviews to process: "
        f"{total_reviews}"
    )


    translated_reviews = []


    for index, row in df.iterrows():

        text = row["Review"]

        language = row["language"]


        translated_text = translate_review(
            text,
            language
        )


        translated_reviews.append(
            translated_text
        )


        translated_counter += 1


        # Progress every 500 reviews
        if (
            translated_counter % 500 == 0
            or translated_counter == total_reviews
        ):

            print(
                f"Processed "
                f"{translated_counter}/"
                f"{total_reviews}"
            )


    df["Review_English"] = (
        translated_reviews
    )


    # --------------------------------------------------------
    # Save translated dataset immediately
    # --------------------------------------------------------

    df.to_csv(
        TRANSLATED_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        f"\nTranslated dataset saved as:"
        f"\n{TRANSLATED_FILE}"
    )


# ============================================================
# Step 8: If translation is disabled
# ============================================================

elif (
    not TRANSLATE_NON_ENGLISH
    and not TRANSLATED_DATASET_EXISTS
):

    print("\n========================================")
    print("STEP 7: TRANSLATION SKIPPED")
    print("========================================")

    print(
        "Using original reviews."
    )

    # Detect only the sample language
    # information is available.
    #
    # Since translation was skipped, use
    # original review text for modeling.

    df["language"] = "not_translated"

    df["Review_English"] = (
        df["Review"]
    )


# ============================================================
# Step 9: Verify English dataset
# ============================================================

print("\n========================================")
print("STEP 8: VERIFYING ENGLISH DATA")
print("========================================")


df["Review_English"] = (
    df["Review_English"]
    .astype(str)
    .str.strip()
)


# Remove empty translated reviews
df = df[
    df["Review_English"].str.len() > 0
].reset_index(drop=True)


print(
    "Final dataset shape:",
    df.shape
)


print("\nSample reviews:")

display_columns = [
    "Review",
    "Review_English"
]


if "language" in df.columns:

    display_columns.insert(
        1,
        "language"
    )


print(
    df[display_columns]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# Step 10: Convert ratings into sentiment labels
# ============================================================

print("\n========================================")
print("STEP 9: CREATING SENTIMENT LABELS")
print("========================================")


def rating_to_sentiment(rating):

    if rating <= 2:

        return "NEGATIVE"

    elif rating == 3:

        return "NEUTRAL"

    else:

        return "POSITIVE"


df["sentiment"] = (
    df["Label"]
    .apply(rating_to_sentiment)
)


print(
    "\nClass distribution:"
)

print(
    df["sentiment"]
    .value_counts()
)


# ============================================================
# Step 11: Train/Test split
# ============================================================

print("\n========================================")
print("STEP 10: TRAIN / TEST SPLIT")
print("========================================")


X_train, X_test, y_train, y_test = train_test_split(

    df["Review_English"],

    df["sentiment"],

    test_size=0.20,

    random_state=42,

    stratify=df["sentiment"]
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# Step 12: TF-IDF Vectorization
# ============================================================

print("\n========================================")
print("STEP 11: TF-IDF VECTORIZATION")
print("========================================")


vectorizer = TfidfVectorizer(

    stop_words="english",

    max_features=10000,

    ngram_range=(1, 2),

    min_df=2
)


X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)


print(
    "\nTF-IDF training matrix:",
    X_train_vec.shape
)

print(
    "TF-IDF testing matrix:",
    X_test_vec.shape
)


# ============================================================
# Step 13: Logistic Regression
# ============================================================

print("\n========================================")
print("STEP 12: TRAINING LOGISTIC REGRESSION")
print("========================================")


model = LogisticRegression(

    max_iter=1000,

    class_weight="balanced"
)


model.fit(
    X_train_vec,
    y_train
)


print(
    "Logistic Regression training completed."
)


# ============================================================
# Step 14: Predictions
# ============================================================

print("\n========================================")
print("STEP 13: MAKING PREDICTIONS")
print("========================================")


y_pred = model.predict(
    X_test_vec
)


# ============================================================
# Step 15: Evaluation
# ============================================================

print("\n========================================")
print("STEP 14: MODEL EVALUATION")
print("========================================")


labels = list(
    model.classes_
)


cm = confusion_matrix(

    y_test,

    y_pred,

    labels=labels

).tolist()


report = classification_report(

    y_test,

    y_pred,

    labels=labels,

    output_dict=True

)


acc = accuracy_score(

    y_test,

    y_pred

)


print(
    f"\nAccuracy: {acc:.4f}"
)


print(
    "\nConfusion Matrix:"
)

print(cm)


print(
    "\nClassification Report:"
)


for lbl in labels:

    r = report[lbl]

    print(

        f"{lbl}: "

        f"precision={r['precision']:.3f}, "

        f"recall={r['recall']:.3f}, "

        f"f1={r['f1-score']:.3f}, "

        f"support={int(r['support'])}"

    )


# ============================================================
# Step 16: Save evaluation metrics
# ============================================================

print("\n========================================")
print("STEP 15: SAVING METRICS")
print("========================================")


if "language" in df.columns:

    language_distribution = (
        df["language"]
        .value_counts()
        .to_dict()
    )

else:

    language_distribution = {}


metrics = {

    "dataset":
        "Coursera Course Reviews",

    "dataset_size_after_cleaning":
        len(df),

    "label_source":
        "Derived from Rating: "
        "1-2=NEGATIVE, "
        "3=NEUTRAL, "
        "4-5=POSITIVE",

    "language_processing": {

        "sample_size":
            sample_size,

        "language_detection":
            "Automatic language detection using langdetect",

        "translation":
            "Non-English reviews translated to English",

        "translation_library":
            "deep-translator",

        "translation_target":
            "English",

        "original_text_column":
            "Review",

        "translated_text_column":
            "Review_English"

    },

    "sample_language_distribution":
        language_counts.to_dict(),

    "full_language_distribution":
        language_distribution,

    "labels":
        labels,

    "confusion_matrix":
        cm,

    "classification_report":
        report,

    "accuracy":
        acc,

    "model":
        "TF-IDF + Logistic Regression",

    "train_test_split":
        "80/20",

    "random_state":
        42,

    "tfidf_parameters": {

        "max_features":
            10000,

        "ngram_range":
            "(1, 2)",

        "stop_words":
            "english",

        "min_df":
            2

    },

    "logistic_regression_parameters": {

        "max_iter":
            1000,

        "class_weight":
            "balanced"

    },

    "note":
        "Metrics were calculated from the real "
        "Coursera course review dataset. "
        "Language detection and translation were "
        "performed as preprocessing. "
        "No evaluation numbers are hardcoded."

}


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


print(
    f"Metrics saved to: {METRICS_FILE}"
)


# ============================================================
# Step 17: Save trained model and vectorizer
# ============================================================

print("\n========================================")
print("STEP 16: SAVING TRAINED MODEL")
print("========================================")


joblib.dump(

    model,

    MODEL_FILE

)


joblib.dump(

    vectorizer,

    VECTORIZER_FILE

)


print(
    f"Model saved as: {MODEL_FILE}"
)

print(
    f"Vectorizer saved as: {VECTORIZER_FILE}"
)


# ============================================================
# Final summary
# ============================================================

print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")


print(
    "\nFinal dataset size:",
    len(df)
)


print(
    "Accuracy:",
    f"{acc:.4f}"
)


print(
    "\nFiles:"
)


print(
    f"1. {TRANSLATED_FILE}"
)

print(
    f"2. {MODEL_FILE}"
)

print(
    f"3. {VECTORIZER_FILE}"
)

print(
    f"4. {METRICS_FILE}"
)


print(
    "\nDone!"
)