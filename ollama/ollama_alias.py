import ollama
import argparse


system_prompt = """You're a helpful assistant. The user will ask you a question and answer it consisely if possible.
Only give detailed answer if the user ask for it. """

parser = argparse.ArgumentParser(description="Ollama input prompt ")
parser.add_argument("prompt", help="Input prompt to ask the LLM.")
args = parser.parse_args()

# ollama chat function call
respone = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rf"{args.prompt} \nothink"}
    ],
    stream=True
)


no_whitespace = False
for text_chunk in respone:
    if text_chunk.message.content in ["<think>", "</think>"]:
        continue
    print(text_chunk.message.content, end="", flush=True)
print("\n")
