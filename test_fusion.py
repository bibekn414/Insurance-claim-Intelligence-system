"""
Test script for the Multimodal Fusion Engine

Pipeline:
Image -> CNN
Text -> DistilBERT
CNN + DistilBERT -> Fusion Engine
"""

from backend.inference import (
    predict_image,
    predict_claim,
)

from backend.fusion_engine import (
    fuse_predictions,
    print_fusion_result,
)


def main():

    # --------------------------------------------------
    # Sample Inputs
    # --------------------------------------------------

    image_path = "sample_images/car1.jpg"

    claim_text = (
        "The front bumper is damaged and the headlight is broken. "
        "There are scratches on the left door."
    )

    # --------------------------------------------------
    # CNN Prediction
    # --------------------------------------------------

    print("\nRunning CNN...")

    image_result = predict_image(image_path)

    # --------------------------------------------------
    # DistilBERT Prediction
    # --------------------------------------------------

    print("\nRunning DistilBERT...")

    text_result = predict_claim(claim_text)

    # --------------------------------------------------
    # Fusion
    # --------------------------------------------------

    print("\nRunning Fusion Engine...")

    fusion_result = fuse_predictions(
        image_result,
        text_result,
    )

    # --------------------------------------------------
    # Print Result
    # --------------------------------------------------

    print_fusion_result(fusion_result)


if __name__ == "__main__":
    main()