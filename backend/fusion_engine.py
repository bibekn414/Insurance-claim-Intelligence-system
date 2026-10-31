"""
Fusion Engine

Combines predictions from:
1. CNN Image Classifier
2. DistilBERT Claim Text Classifier

Author: Rohit
Project: Multimodal Insurance Claim Intelligence System
"""

from backend.config import (
    CLASS_NAMES,
    LABEL_MAP,
    IMAGE_WEIGHT,
    TEXT_WEIGHT,
    FUSION_THRESHOLD
)

def prediction_list_to_dict(predictions):
    """
    Converts the prediction list returned by inference.py
    into a dictionary.

    Example:

    [
        {
            "original_label":"tray_son",
            "confidence":0.42
        }
    ]

    becomes

    {
        "tray_son":0.42
    }
    """

    return {
        item["original_label"]: item["confidence"]
        for item in predictions
    }

# Merge Predictions

def merge_predictions(
    image_probs,
    text_probs,
    image_weight=IMAGE_WEIGHT,
    text_weight=TEXT_WEIGHT
):
    """
    Performs weighted fusion of image and text predictions.

    Parameters
    ----------
    image_probs : dict
        CNN probabilities.

    text_probs : dict
        DistilBERT probabilities.

    Returns
    -------
    dict
        Weighted probabilities.
    """

    fused = {}

    for cls in CLASS_NAMES:

        img_score = image_probs.get(cls, 0.0)
        txt_score = text_probs.get(cls, 0.0)

        fused[cls] = (
            image_weight * img_score +
            text_weight * txt_score
        )

    return fused

# Agreement Score

def calculate_agreement(image_result, text_result):

    image_label = image_result["top_prediction"]["label"]

    text_label = text_result["top_prediction"]["label"]

    if image_label == text_label:

        return (
            image_result["top_prediction"]["confidence"] +
            text_result["top_prediction"]["confidence"]
        ) / 2

    return 0.0

# Consistency

def calculate_consistency(score):
    """
    Converts agreement score into a consistency label.
    """

    if score >= 0.70:
        return "High"

    elif score >= 0.40:
        return "Medium"

    else:
        return "Low"

# Recommendation

def generate_recommendation(consistency):
    """
    Generates recommendation based on consistency level.
    """

    recommendations = {
        "High":
            "Models strongly agree. Automatic assessment is reliable.",

        "Medium":
            "Moderate agreement detected. Manual verification is recommended.",

        "Low":
            "Low agreement between image and text. Human inspection is required."
    }

    return recommendations[consistency]


# Final Damage Prediction

def calculate_final_damage(
    fused_probs,
    threshold=FUSION_THRESHOLD
):

    detected = {}

    for cls, prob in fused_probs.items():

        if prob >= threshold:

            detected[LABEL_MAP[cls]] = round(prob, 3)

    if not detected:

        best = max(
            fused_probs,
            key=fused_probs.get
        )

        detected[LABEL_MAP[best]] = round(
            fused_probs[best],
            3
        )

    return detected

# Main Fusion Pipeline

def fuse_predictions(
    image_result,
    text_result
):
    """
    Complete fusion pipeline.

    Parameters
    ----------
    image_result : dict
        Output from predict_image()

    text_result : dict
        Output from predict_claim()

    Returns
    -------
    dict
        Final multimodal prediction.
    """

    image_probs = prediction_list_to_dict(
    image_result["all_predictions"]
)

    text_probs = prediction_list_to_dict(
    text_result["all_predictions"]
)

    # Weighted Fusion
    fused_probs = merge_predictions(
        image_probs,
        text_probs
    )

    # Agreement Score
    agreement = calculate_agreement(
        image_result,
        text_result
    )

    # Consistency
    consistency = calculate_consistency(
        agreement
    )

    # Recommendation
    recommendation = generate_recommendation(
        consistency
    )

    dominant_model = determine_dominant_model(
    image_result,
    text_result
    )

    confidence_gap = calculate_confidence_gap(
    image_result,
    text_result
    )

    conflict_level = determine_conflict_level(
    agreement,
    confidence_gap
    )

    recommended_action = determine_action(
    conflict_level
    )

    reason = generate_reason(
    image_result,
    text_result,
    conflict_level
    )

    # Final Damage Prediction
    final_damage = calculate_final_damage(
        fused_probs
    )



    return {

        "image_prediction": image_result["top_prediction"],

        "text_prediction": text_result["top_prediction"],

        "final_damage": final_damage,

        "agreement_score": agreement,

        "consistency": consistency,

        "dominant_model": dominant_model,

        "confidence_gap": confidence_gap,

        "conflict_level": conflict_level,

        "recommended_action": recommended_action,

        "reason": reason,

        "recommendation": recommendation,

        "fused_probabilities": {

            LABEL_MAP[cls]: round(prob, 3)
            for cls, prob in fused_probs.items()
        }

    }

# Pretty Printer

def print_fusion_result(result):
    """
    Prints fusion results in a readable format.
    """
    print("MULTIMODAL FUSION RESULT")

    image = result["image_prediction"]
    text = result["text_prediction"]

    print(f"\nImage Prediction : "f"{image['label']} ({image['confidence']:.2%})")

    print(f"Text Prediction : "f"{text['label']} ({text['confidence']:.2%})")

    print("\nFinal Damage Assessment")

    for cls, prob in result["final_damage"].items():
        print(f"  • {cls}: {prob:.3f}")

    print(f"\nAgreement Score : {result['agreement_score']}")
    print(f"Consistency     : {result['consistency']}")

    print(f"\nDominant Model : {result['dominant_model']}")
    print(f"Confidence Gap : {result['confidence_gap']}")
    print(f"Conflict Level : {result['conflict_level']}")
    print(f"Action         : {result['recommended_action']}")

    print("\nReason")
    print(result["reason"])

    print("\nRecommendation")
    print(result["recommendation"])

    print("\nFused Probabilities")

    for cls, prob in sorted(
        result["fused_probabilities"].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{cls:<20} {prob:.3f}")

    print("=" * 60)


def determine_dominant_model(image_result, text_result):
    """
    Determines which model made the stronger prediction.
    """

    image_conf = image_result["top_prediction"]["confidence"]
    text_conf = text_result["top_prediction"]["confidence"]

    if image_conf > text_conf:
        return "CNN"

    elif text_conf > image_conf:
        return "DistilBERT"

    return "Equal"

def calculate_confidence_gap(image_result, text_result):
    """
    Difference between the top prediction confidences.
    """

    image_conf = image_result["top_prediction"]["confidence"]
    text_conf = text_result["top_prediction"]["confidence"]

    return round(abs(image_conf - text_conf), 3)

def determine_conflict_level(
    agreement,
    confidence_gap
):
    """
    Determines disagreement severity.
    """

    # Different predicted classes with a large confidence gap
    if agreement == 0.0 and confidence_gap >= 0.40:
        return "High"

    # Different predicted classes but similar confidence
    elif agreement == 0.0:
        return "Medium"

    # Same prediction but low confidence
    elif agreement < 0.70:
        return "Medium"

    # Strong agreement
    return "Low"

def determine_action(conflict):

    actions = {

        "Low":
            "Auto Approve",

        "Medium":
            "Manual Verification",

        "High":
            "Manual Inspection Required"

    }

    return actions[conflict]

def generate_reason(
    image_result,
    text_result,
    conflict
):
    """
    Human-readable explanation of the fusion outcome.
    """

    image_label = image_result["top_prediction"]["label"]
    text_label = text_result["top_prediction"]["label"]

    if image_label == text_label:

        return (
            f"Both CNN and DistilBERT identify "
            f"'{image_label}' as the primary damage."
        )

    if conflict == "High":

        return (
            f"The image suggests '{image_label}', "
            f"while the claim text indicates "
            f"'{text_label}'. The modalities disagree "
            f"significantly."
        )

    return (
        f"The image predicts '{image_label}' "
        f"while the claim text predicts "
        f"'{text_label}'. Partial agreement detected."
    )

