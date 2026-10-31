from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/distilbert_damage_classifier"

print("Loading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

print("DistilBERT Loaded Successfully!")
print("Labels:", model.config.num_labels)