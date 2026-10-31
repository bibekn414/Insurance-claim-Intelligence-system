from backend.gemini_client import model
response = model.generate_content(
    "Say hello in one sentence."
)
print(response.text)