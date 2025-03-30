from transformers import pipeline

# Define sentence to translate
source_sentence = "Hej detta är en text skriven på svenska. Denna text kommer att översättas till engelska av en djupinlärningsmodell."

# Load in model
model = pipeline("translation", model="Helsinki-NLP/opus-mt-sv-en", device="cuda:0")

# Perform inferencce
output = model(source_sentence)

print(f"Input: {source_sentence}")
print(f"Translation: {output[0]["translation_text"]}")
