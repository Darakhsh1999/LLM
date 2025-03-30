import ollama
from config import Config

c = Config()

# Model prompt
prompt = "What tools are essential for camping in the wilds where the climate has a lot of trees. Limit respone to 3 sentences"

# ollama chat function call
respone = ollama.chat(
    model=c.model_name,
    messages=[{"role": "user", "content": prompt}],
)

print("Respone:")
print(respone.message.content)

# ollama chat function call
prompt = "List the 5 first double digit prime numbers"
respone = ollama.chat(
    model=c.model_name,
    messages=[{"role": "user", "content": prompt}],
    stream=True
)

print("Respone:")
for text_chunk in respone:
    print(text_chunk.message.content, end="", flush=True)
