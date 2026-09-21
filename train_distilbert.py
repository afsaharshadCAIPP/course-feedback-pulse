# ============================================================
# FINE-TUNED MULTILINGUAL DISTILBERT
# Coursera Course Feedback Sentiment Analysis
#
# Combines the strongest parts of CODE 1 and CODE 2:
#
# 1. Loads Coursera reviews dataset
# 2. Cleans reviews and removes duplicates
# 3. Converts ratings into 3 sentiment classes
# 4. Uses 80/10/10 train/validation/test split
# 5. Uses class weights for imbalanced classes
# 6. Fine-tunes multilingual DistilBERT
# 7. Uses Macro F1 for model selection
# 8. Keeps TEST data completely separate
# 9. Evaluates with detailed metrics
# 10. Saves model, tokenizer and metrics
# 11. Includes prediction function
# ============================================================

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix)

from datasets import Dataset

from transformers import (AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments)


# ============================================================
# STEP 2: REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# STEP 3: CONFIGURATION
# ============================================================

DATASET_PATH = "reviews_by_course.csv"
MODEL_NAME = "distilbert-base-multilingual-cased"
OUTPUT_DIR = "./distilbert_results"
MODEL_OUTPUT_DIR = "./coursera_multilingual_distilbert"
METRICS_FILE = "./distilbert_metrics.json"
MAX_LENGTH = 128
NUM_LABELS = 3
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01


# ============================================================
# STEP 4: CHECK DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("STEP 1: DEVICE CHECK")
print("=" * 60)

print("Training device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("GPU not available.")
    print("Training will use CPU and may take a long time.")


# ============================================================
# STEP 5: LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print("Original dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# STEP 6: CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["Review", "Label"]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found in the dataset."
        )


# ============================================================
# STEP 7: CLEAN DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: CLEANING DATA")
print("=" * 60)

# Remove missing reviews
df = df.dropna(subset=["Review"]).copy()

# Convert reviews to strings
df["Review"] = df["Review"].astype(str)

# Remove empty reviews
df["Review"] = df["Review"].str.strip()

df = df[df["Review"] != ""]

# Remove missing labels
df = df.dropna(subset=["Label"]).copy()

# Remove duplicate reviews
before_duplicates = len(df)

df = df.drop_duplicates(subset=["Review"]).copy()
after_duplicates = len(df)

print("Duplicate reviews removed:",before_duplicates - after_duplicates)
print("Dataset shape after cleaning:", df.shape)


# ============================================================
# STEP 8: CONVERT RATINGS TO SENTIMENT LABELS
# ============================================================
#
# Rating:
# 1-2 = Negative
# 3   = Neutral
# 4-5 = Positive
#
# Numeric labels:
# 0 = Negative
# 1 = Neutral
# 2 = Positive
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: CREATING SENTIMENT LABELS")
print("=" * 60)

def label_sentiment(rating):

    rating = float(rating)

    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

df["label"] = df["Label"].apply(label_sentiment)

label_names = ["negative","neutral","positive"]


# ============================================================
# STEP 9: SHOW CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

class_counts = df["label"].value_counts().sort_index()

for label_id, count in class_counts.items():
    print(f"{label_names[label_id]:>10}: {count}")


# ============================================================
# STEP 10: TRAIN / VALIDATION / TEST SPLIT
# ============================================================
#
# 80% Training
# 10% Validation
# 10% Test
#
# IMPORTANT:
# The TEST set is never used during training.
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)


# First split:90% temporary data and 10% test data

train_val_df, test_df = train_test_split(
    df[["Review", "label"]],
    test_size=0.10,
    random_state=SEED,
    stratify=df["label"]
)

# Second split: 80% train, 10% validation and 10% / 90% = 11.11% of train_val

train_df, val_df = train_test_split(
    train_val_df,
    test_size=1 / 9,
    random_state=SEED,
    stratify=train_val_df["label"]
)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))

print(
    "\nApproximate split:",
    f"{len(train_df)/len(df)*100:.1f}% train,",
    f"{len(val_df)/len(df)*100:.1f}% validation,",
    f"{len(test_df)/len(df)*100:.1f}% test"
)


# ============================================================
# STEP 11: COMPUTE CLASS WEIGHTS
# ============================================================
# This helps the model pay more attention to minority classes.
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: COMPUTING CLASS WEIGHTS")
print("=" * 60)

classes = np.array([0, 1, 2])

class_weights = compute_class_weight(class_weight="balanced",classes=classes,y=train_df["label"].values)

print("Class weights (negative, neutral, positive):")

print(class_weights)

# Convert weights to PyTorch tensor
class_weights_tensor = torch.tensor(class_weights,dtype=torch.float).to(device)


# ============================================================
# STEP 12: CONVERT DATA TO HUGGING FACE DATASETS
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: CREATING HUGGING FACE DATASETS")
print("=" * 60)

train_ds = Dataset.from_pandas(
    train_df.rename(
        columns={"Review": "text"}
    ),
    preserve_index=False
)

val_ds = Dataset.from_pandas(
    val_df.rename(
        columns={"Review": "text"}
    ),
    preserve_index=False
)

test_ds = Dataset.from_pandas(
    test_df.rename(
        columns={"Review": "text"}
    ),
    preserve_index=False
)

print("Train dataset:", train_ds)
print("Validation dataset:", val_ds)
print("Test dataset:", test_ds)


# ============================================================
# STEP 13: LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded:")
print(MODEL_NAME)


# ============================================================
# STEP 14: TOKENIZATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: TOKENIZING DATA")
print("=" * 60)

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_ds = train_ds.map(tokenize_function,batched=True)

val_ds = val_ds.map(tokenize_function,batched=True)

test_ds = test_ds.map(tokenize_function,batched=True)

# Rename label column to labels
train_ds = train_ds.rename_column("label","labels")

val_ds = val_ds.rename_column("label","labels")

test_ds = test_ds.rename_column("label","labels")


# ============================================================
# STEP 15: SET PYTORCH FORMAT
# ============================================================

train_ds.set_format(type="torch",columns=["input_ids","attention_mask","labels"])

val_ds.set_format(type="torch",columns=["input_ids","attention_mask","labels"])

test_ds.set_format(type="torch",columns=["input_ids","attention_mask","labels"])

print("Tokenization completed.")


# ============================================================
# STEP 16: LOAD MULTILINGUAL DISTILBERT
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: LOADING MULTILINGUAL DISTILBERT")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=NUM_LABELS)

model.to(device)

print("Model loaded:")
print(MODEL_NAME)

print("Number of labels:", NUM_LABELS)


# ============================================================
# STEP 17: WEIGHTED TRAINER
# ============================================================
# Uses class-weighted CrossEntropyLoss.
# This is taken from the strongest part of CODE 2.
# ============================================================

class WeightedTrainer(Trainer):

    def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_function = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_function(logits,labels)

        if return_outputs:
            return loss, outputs

        return loss


# ============================================================
# STEP 18: METRICS FUNCTION
# ============================================================
# Macro F1 is used because it gives equal importance
# to negative, neutral and positive classes.
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits,axis=-1)
    precision, recall, f1, _ = (precision_recall_fscore_support(labels,predictions,average="macro",zero_division=0))
    accuracy = accuracy_score(labels,predictions)

    return {"accuracy": accuracy,"f1_macro": f1,"precision_macro": precision,"recall_macro": recall}


# ============================================================
# STEP 19: TRAINING ARGUMENTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: CONFIGURING TRAINING")
print("=" * 60)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=SEED
)


# ============================================================
# STEP 20: CREATE TRAINER
# ============================================================

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)


# ============================================================
# STEP 21: TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 12: FINE-TUNING DISTILBERT")
print("=" * 60)

print("Training device:", device)
print("\nTraining started...")
trainer.train()

print("\nTraining completed!")


# ============================================================
# STEP 22: VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 13: VALIDATION RESULTS")
print("=" * 60)

validation_results = trainer.evaluate(eval_dataset=val_ds)
print("\nValidation results:")

for key, value in validation_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# ============================================================
# STEP 23: FINAL TEST EVALUATION
# ============================================================
# IMPORTANT:
# This is the FIRST time the test set is used.
# The test set was NOT used to choose the best model.
# ============================================================

print("\n" + "=" * 60)
print("STEP 14: FINAL TEST EVALUATION")
print("=" * 60)

test_results = trainer.evaluate(eval_dataset=test_ds,metric_key_prefix="test")
print("\nTest results:")

for key, value in test_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# ============================================================
# STEP 24: PREDICTIONS ON TEST SET
# ============================================================

print("\n" + "=" * 60)
print("STEP 15: CLASSIFICATION REPORT")
print("=" * 60)

predictions_output = trainer.predict(test_ds)

y_pred = np.argmax(predictions_output.predictions,axis=-1)

y_true = predictions_output.label_ids

print("\nClassification Report:\n")

report = classification_report(y_true,y_pred,target_names=label_names,digits=4,zero_division=0)

print(report)


# ============================================================
# STEP 25: CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("STEP 16: CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_true,y_pred)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(pd.DataFrame(
        cm,
        index=[
            "Actual Negative",
            "Actual Neutral",
            "Actual Positive"
        ],
        columns=[
            "Predicted Negative",
            "Predicted Neutral",
            "Predicted Positive"
        ]
    )
)


# ============================================================
# STEP 26: DETAILED PER-CLASS METRICS
# ============================================================

print("\n" + "=" * 60)
print("STEP 17: PER-CLASS METRICS")
print("=" * 60)

precision, recall, f1, support = (precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0
    )
)

per_class_metrics = {}

for i, label_name in enumerate(label_names):

    per_class_metrics[label_name] = {
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1": float(f1[i]),
        "support": int(support[i])
    }

    print(f"\n{label_name.upper()}")
    print(f"Precision: {precision[i]:.4f}")
    print(f"Recall:    {recall[i]:.4f}")
    print(f"F1-score:  {f1[i]:.4f}")
    print(f"Support:   {support[i]}")


# ============================================================
# STEP 27: SAVE METRICS TO JSON
# ============================================================

print("\n" + "=" * 60)
print("STEP 18: SAVING METRICS")
print("=" * 60)

metrics_data = {
    "model": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "dataset_size": int(len(df)),
    "train_size": int(len(train_df)),
    "validation_size": int(len(val_df)),
    "test_size": int(len(test_df)),
    "class_names": label_names,
    "class_weights": [float(weight) for weight in class_weights],
    "validation_results": {
        key: float(value)
        if isinstance(value, (np.floating, float))
        else value
        for key, value in validation_results.items()
    },
    "test_results": {
        key: float(value)
        if isinstance(value, (np.floating, float))
        else value
        for key, value in test_results.items()
    },
    "per_class_metrics": per_class_metrics,
    "confusion_matrix": cm.tolist()
}


with open(METRICS_FILE,"w",encoding="utf-8") as f:
    json.dump(metrics_data,f,indent=4)

print("Metrics saved to:",METRICS_FILE)


# ============================================================
# STEP 28: SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 19: SAVING MODEL")
print("=" * 60)

trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

print("Model saved to:",MODEL_OUTPUT_DIR)


# ============================================================
# STEP 29: PREDICTION FUNCTION
# ============================================================
# Can be used later for new reviews.
# Supports multilingual text.
# ============================================================

def predict_sentiment(text):

    model.eval()

    inputs = tokenizer(text,return_tensors="pt",truncation=True,padding=True,max_length=MAX_LENGTH)

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits,dim=-1)
        prediction = torch.argmax(probabilities,dim=-1).item()

    confidence = probabilities[0,prediction].item()

    return {"sentiment": label_names[prediction],"confidence": confidence}


# ============================================================
# STEP 30: TEST PREDICTION FUNCTION
# ============================================================

print("\n" + "=" * 60)
print("STEP 20: TESTING PREDICTION FUNCTION")
print("=" * 60)

test_sentences = [
    "This course was amazing, I learned so much!",
    "Ce cours était décevant et mal organisé.",
    "这门课程很好，内容很实用",
    "The course was okay, nothing special.",
    "I really enjoyed this course."
]

for sentence in test_sentences:
    result = predict_sentiment(sentence)

    print("\nText:")
    print(sentence)

    print("Predicted sentiment:",result["sentiment"])

    print("Confidence:",f"{result['confidence']:.4f}")


# ============================================================
# STEP 31: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nModel:", MODEL_NAME)
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))
print("\nTest Accuracy:",f"{test_results.get('test_accuracy', 0):.4f}")
print("Test Macro F1:",f"{test_results.get('test_f1_macro', 0):.4f}")

print("\nModel saved to:",MODEL_OUTPUT_DIR)

print("Metrics saved to:",METRICS_FILE)

print("\nAll steps completed successfully!")
