from backend.inference import (
    predict_image,
    predict_claim,
    print_predictions
)

# ----------------------------------------
# Image Test
# ----------------------------------------

image_path = "sample_images/car1.jpg"

image_result = predict_image(image_path)

print_predictions(image_result)

# ----------------------------------------
# Text Test
# ----------------------------------------

claim = """
The front bumper is dented and the left headlight is broken.
"""

text_result = predict_claim(claim)

print_predictions(text_result)