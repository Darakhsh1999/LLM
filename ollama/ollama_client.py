import ollama

model_name = "llama3.2"

# ollama client
client = ollama.Client()

# Model prompt
prompt = "What tools are essential for camping in the wilds where the climate has a lot of trees."

# Generate respone
response = client.generate(model=model_name, prompt=prompt)


print("Respone:")
print(response.response)