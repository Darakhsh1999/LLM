import ollama
from config import Config

c = Config()

# ollama client
client = ollama.Client()

# Model prompt
prompt = "What tools are essential for camping in the wilds where the climate has a lot of trees."


# Generate respone
response = client.generate(model=c.model_name, prompt=prompt)


print("Respone:")
print(response.response)