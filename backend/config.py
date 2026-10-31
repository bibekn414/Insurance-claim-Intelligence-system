"""
Project Configuration
"""
# Model Paths

CNN_MODEL_PATH = "models/efficientnet_full_vehicle.keras"

BERT_MODEL_PATH = "models/distilbert_damage_classifier"

# Image Settings

IMG_SIZE = (300, 300)

IMAGE_THRESHOLD = 0.50

TEXT_THRESHOLD = 0.50

# Damage Classes

CLASS_NAMES = [
    "tray_son",
    "mat_bo_phan",
    "thung",
    "vo_kinh",
    "be_den",
    "rach",
    "mop_lom",
]

# English Labels

LABEL_MAP = {
    "tray_son": "Paint Scratch",
    "mat_bo_phan": "Missing Part",
    "thung": "Hole",
    "vo_kinh": "Broken Glass",
    "be_den": "Broken Light",
    "rach": "Tear",
    "mop_lom": "Dent",
}

# Fusion Configuration

IMAGE_WEIGHT = 0.4
TEXT_WEIGHT = 0.6

FUSION_THRESHOLD = 0.50

##Gemini

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.3
GEMINI_MAX_OUTPUT_TOKENS = 1024