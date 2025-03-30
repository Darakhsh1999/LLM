import ollama
import argparse


parser = argparse.ArgumentParser(description="Ollama input prompt ")
parser.add_argument("prompt", help="Input prompt to ask the LLM.")
args = parser.parse_args()

# ollama chat function call
respone = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": f"{args.prompt}"}],
    stream=True
)

for text_chunk in respone:
    print(text_chunk.message.content, end="", flush=True)
print("\n")
