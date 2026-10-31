from tensorflow.keras.models import load_model

MODEL_PATH = "models/efficientnet_full_vehicle.keras"

print("Loading CNN model...")

model = load_model(MODEL_PATH, compile=False)

print("✅ CNN Loaded Successfully!")
print(model.input_shape)
print(model.output_shape)