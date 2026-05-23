"""
=============================================================================
GRADIO WEB APP — Multi-Class Cyberbullying Comment Classifier
=============================================================================
This app loads the 3 trained models (BERT, Logistic Regression, Linear SVM)
and provides a web interface for classifying comments.

Usage:
    1. Make sure 'exported_models/' folder exists in the same directory
    2. Install dependencies:  pip install gradio torch transformers joblib scikit-learn
    3. Run:  python gradio_app.py
=============================================================================
"""

import os
import re
import json
import numpy as np
import gradio as gr
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax as scipy_softmax

# =============================================================================
# CONFIGURATION
# =============================================================================
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_models")

# =============================================================================
# LOAD METADATA
# =============================================================================
print("Loading model metadata...")
with open(os.path.join(EXPORT_DIR, "model_metadata.json"), "r") as f:
    metadata = json.load(f)

LABEL_MAP = metadata["label_map"]
TARGET_NAMES = metadata["target_names"]
NUM_LABELS = metadata["num_labels"]
MODEL_NAME = metadata["bert_model_name"]
MAX_LEN = metadata["max_len"]

# Reverse label map: index -> name
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

print(f"  Labels: {TARGET_NAMES}")
print(f"  BERT base model: {MODEL_NAME}")
print(f"  Max sequence length: {MAX_LEN}")


# =============================================================================
# TEXT PREPROCESSING (must match training preprocessing exactly)
# =============================================================================

def clean_text_bert(text):
    """Minimal preprocessing for BERT (same as clean_text in training)."""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text_baseline(text):
    """More aggressive preprocessing for LR/SVM (same as clean_tweet in training)."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'rt\s+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =============================================================================
# LOAD MODELS
# =============================================================================

# --- BERT ---
print("Loading BERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

bert_tokenizer = AutoTokenizer.from_pretrained(os.path.join(EXPORT_DIR, "bert_tokenizer"))
bert_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)
bert_model.load_state_dict(
    torch.load(os.path.join(EXPORT_DIR, "bert_model.pt"), map_location=device)
)
bert_model.to(device)
bert_model.eval()
print("  BERT model loaded successfully!")

# --- Logistic Regression ---
print("Loading Logistic Regression model...")
lr_model = joblib.load(os.path.join(EXPORT_DIR, "logistic_regression.pkl"))
print("  Logistic Regression loaded successfully!")

# --- Linear SVM ---
print("Loading Linear SVM model...")
svm_model = joblib.load(os.path.join(EXPORT_DIR, "linear_svm.pkl"))
print("  Linear SVM loaded successfully!")

# --- TF-IDF Vectorizer ---
print("Loading TF-IDF Vectorizer...")
tfidf = joblib.load(os.path.join(EXPORT_DIR, "tfidf_vectorizer.pkl"))
print("  TF-IDF Vectorizer loaded successfully!")

print("\nAll models loaded. Ready for inference!\n")


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_bert(text):
    """Run BERT inference on a single text."""
    cleaned = clean_text_bert(text)
    encoding = bert_tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]

    return probs


def predict_lr(text):
    """Run Logistic Regression inference on a single text."""
    cleaned = clean_text_baseline(text)
    tfidf_vec = tfidf.transform([cleaned])
    probs = lr_model.predict_proba(tfidf_vec)[0]
    return probs


def predict_svm(text):
    """Run Linear SVM inference on a single text."""
    cleaned = clean_text_baseline(text)
    tfidf_vec = tfidf.transform([cleaned])
    raw_scores = svm_model.decision_function(tfidf_vec)
    probs = scipy_softmax(raw_scores, axis=1)[0]
    return probs


def format_label(label_name):
    """Make label names more readable."""
    return label_name.replace("_", " ").title()


def classify_comment(text):
    """
    Main classification function called by Gradio.
    Returns prediction results from all 3 models.
    """
    if not text or not text.strip():
        return (
            {},  # BERT
            {},  # LR
            {},  # SVM
            "Please enter a comment to classify."
        )

    # Get predictions from all 3 models
    bert_probs = predict_bert(text)
    lr_probs = predict_lr(text)
    svm_probs = predict_svm(text)

    # Format as {label: confidence} dicts for Gradio Label component
    bert_result = {format_label(TARGET_NAMES[i]): float(bert_probs[i]) for i in range(NUM_LABELS)}
    lr_result = {format_label(TARGET_NAMES[i]): float(lr_probs[i]) for i in range(NUM_LABELS)}
    svm_result = {format_label(TARGET_NAMES[i]): float(svm_probs[i]) for i in range(NUM_LABELS)}

    # Summary text
    bert_pred = TARGET_NAMES[np.argmax(bert_probs)]
    lr_pred = TARGET_NAMES[np.argmax(lr_probs)]
    svm_pred = TARGET_NAMES[np.argmax(svm_probs)]

    agreement = "✅ All 3 models agree!" if bert_pred == lr_pred == svm_pred else "⚠️ Models disagree — review individual predictions."

    summary = f"""### Prediction Summary

| Model | Predicted Class | Confidence |
|---|---|---|
| **BERT** | {format_label(bert_pred)} | {bert_probs[np.argmax(bert_probs)]:.1%} |
| **Logistic Regression** | {format_label(lr_pred)} | {lr_probs[np.argmax(lr_probs)]:.1%} |
| **Linear SVM** | {format_label(svm_pred)} | {svm_probs[np.argmax(svm_probs)]:.1%} |

{agreement}
"""

    return bert_result, lr_result, svm_result, summary


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# Example comments for users to try
examples = [
    # Not Cyberbullying
    ["I had a wonderful time at the park today, the weather was absolutely perfect!"],
    ["Can anyone recommend a good place to buy textbooks for the upcoming semester?"],
    # Religion
    ["All Muslims are violent terrorists who should be completely banned from entering our country."],
    ["Christians are just brainwashed fools who believe in a fake sky daddy and hate science."],
    # Age
    ["You act like a dumb 12 year old kid, log off the internet you literal child."],
    ["Stupid boomers are ruining the economy for the rest of us, go back to the nursing home."],
    # Gender
    ["Shut up you ugly bitch, nobody cares about your stupid opinion anyway."],
    ["She is such a dirty slut, I can't believe anyone would actually date her."],
    # Ethnicity
    ["You dirty niggers are destroying our neighborhoods, go back to Africa where you belong."],
    ["I'm sick of all these filthy wetbacks crossing the border and stealing our jobs."],
    # Other Cyberbullying
    ["You are an absolute loser, you should just go kill yourself and do us all a favor."],
    ["Nobody likes you, you're the ugliest person in the entire school and a total freak."],
]

# Build the Gradio UI
with gr.Blocks(
    title="Cyberbullying Comment Classifier",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="orange",
    ),
) as demo:

    gr.Markdown("""
    # 🛡️ Multi-Class Cyberbullying Comment Classifier
    ### WID3002 NLP Project — Toxic Comment Detection

    Type a comment below and see how **3 different models** classify it into one of 6 categories:
    `Not Cyberbullying` · `Religion` · `Age` · `Gender` · `Ethnicity` · `Other_Cyberbullying`
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter a comment",
                placeholder="Type or paste a comment here...",
                lines=3,
                max_lines=5,
            )
            classify_btn = gr.Button("🔍 Classify Comment", variant="primary", size="lg")

        with gr.Column(scale=3):
            summary_output = gr.Markdown(label="Summary")

    gr.Markdown("---")
    gr.Markdown("### Model Predictions (confidence scores)")

    with gr.Row():
        bert_output = gr.Label(label="🤖 BERT (Fine-tuned)", num_top_classes=6)
        lr_output = gr.Label(label="📊 Logistic Regression", num_top_classes=6)
        svm_output = gr.Label(label="📐 Linear SVM", num_top_classes=6)

    gr.Markdown("---")
    gr.Markdown("### Try these examples:")
    gr.Examples(
        examples=examples,
        inputs=text_input,
        outputs=[bert_output, lr_output, svm_output, summary_output],
        fn=classify_comment,
        cache_examples=False,
    )

    # Wire up the button
    classify_btn.click(
        fn=classify_comment,
        inputs=text_input,
        outputs=[bert_output, lr_output, svm_output, summary_output],
    )

    # Also trigger on Enter key
    text_input.submit(
        fn=classify_comment,
        inputs=text_input,
        outputs=[bert_output, lr_output, svm_output, summary_output],
    )

# =============================================================================
# LAUNCH
# =============================================================================
if __name__ == "__main__":
    print("Starting Gradio app...")
    demo.launch(
        share=False,      # Set to True to get a public link
        inbrowser=True,    # Auto-open in browser
    )
