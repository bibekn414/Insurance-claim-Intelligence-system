"""
backend/inference.py

Production inference module for the
Insurance Claim Intelligence System.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from backend.config import (
    CNN_MODEL_PATH,
    BERT_MODEL_PATH,
    IMG_SIZE,
    IMAGE_THRESHOLD,
    TEXT_THRESHOLD,
    CLASS_NAMES,
    LABEL_MAP,
)

# =====================================================
# Load Models
# =====================================================

print("Loading CNN Model...")

cnn_model = tf.keras.models.load_model(
    CNN_MODEL_PATH,
    compile=False,
)

print("CNN Loaded")


print("Loading DistilBERT")

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL_PATH
)

bert_model.eval()

print("DistilBERT Loaded")


# =====================================================
# Image Preprocessing
# =====================================================

def preprocess_image(image_path: str):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMG_SIZE)

    image = np.array(image).astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=0)

    return image


# =====================================================
# Image Prediction
# =====================================================

def predict_image(image_path, threshold=IMAGE_THRESHOLD):

    image = preprocess_image(image_path)

    probs = cnn_model.predict(image, verbose=0)[0]

    all_predictions = []

    for original, prob in zip(CLASS_NAMES, probs):

        all_predictions.append({
            "label": LABEL_MAP[original],
            "original_label": original,
            "confidence": round(float(prob), 4),
            "detected": bool(prob >= threshold)
        })

    top_prediction = max(
        all_predictions,
        key=lambda x: x["confidence"]
    )

    detected = [
        x for x in all_predictions
        if x["detected"]
    ]

    return {
        "source": "CNN",
        "top_prediction": top_prediction,
        "detected_labels": detected,
        "all_predictions": all_predictions
    }


# =====================================================
# Text Prediction
# =====================================================

def predict_claim(text, threshold=TEXT_THRESHOLD):

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    logits = outputs.logits[0]

    probs = torch.sigmoid(logits).numpy()

    all_predictions = []

    for original, prob in zip(CLASS_NAMES, probs):

        all_predictions.append({
            "label": LABEL_MAP[original],
            "original_label": original,
            "confidence": round(float(prob), 4),
            "detected": bool(prob >= threshold)
        })

    top_prediction = max(
        all_predictions,
        key=lambda x: x["confidence"]
    )

    detected = [
        x for x in all_predictions
        if x["detected"]
    ]

    return {
        "source": "DistilBERT",
        "top_prediction": top_prediction,
        "detected_labels": detected,
        "all_predictions": all_predictions
    }


# =====================================================
# Pretty Printer
# =====================================================

def print_predictions(result):

    print("\n")
   

    print(f"Model : {result['source']}")



    print("\nTop Prediction")

    top = result["top_prediction"]

    print(
        f"{top['label']} "
        f"({top['confidence']:.3f})"
    )

    print("\nAll Predictions")

 

    for item in result["all_predictions"]:

        status = "YES" if item["detected"] else "NO"

        print(
            f"{item['label']:<20}"
            f"{item['confidence']:<10.3f}"
            f"Detected : {status}"
        )

  

    print("\nDetected Labels")

    if len(result["detected_labels"]) == 0:

        print("None")

    else:

        for item in result["detected_labels"]:

            print(
                f"{item['label']} "
                f"({item['confidence']:.3f})"
            )




# =====================================================
# Standalone Test
# =====================================================

if __name__ == "__main__":

    print("Inference module loaded successfully.")