from backend.inference import (
    predict_image,
    predict_claim,
)

from backend.fusion_engine import (
    fuse_predictions,
)

from backend.gemini_client import (
    generate_report,
    print_report,
)

image_result = predict_image("sample_images/car4.jpg")

text_result = predict_claim(
    "Rear windshield is completely shattered. Glass pieces are everywhere."
)

fusion_result = fuse_predictions(
    image_result,
    text_result,
)

report = generate_report(
    fusion_result
)

print_report(report)