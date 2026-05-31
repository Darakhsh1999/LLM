import ollama
import argparse

parser = argparse.ArgumentParser(description="Ollama input prompt ")
parser.add_argument("text", help="Input prompt to ask the LLM.")
parser.add_argument("-r", "--reason", action="store_true", default=False,
                    help="enable reasoning mode")
parser.add_argument("-m", "--model", default=None,
                    help="model name to use")
parser.add_argument("-i", "--interactive", action="store_true", default=False,
                    help="run in interactive mode")


# print("TEXT: ",args.text)         # "input text bla bla bla"
# print("REASON: ",args.reason)       # True if -r/--reason given, else False
# print("MODEL: ",args.model)        # model name string, or None
# print("INTERACTIVE", args.interactive)  # True if -i/--interactive given, else False

system_prompt = """You're a helpful terminal assistant named Alfred. The user, who is named Arash Darakhsh,
will ask you a question. You will think and answer it to your best ability, idealy with a consise answer.
Only give detailed answer if the user ask for it. If suitable, end your reponse with 'Master Darakhsh', similar to how Alfred speaks to Batman (Bruce Wayne)."""

# Parse arguments
args = parser.parse_args()
use_reasoning: bool = args.reason
model_name: str = args.model if (args.model) else "qwen3:4b"
is_interactive: bool = args.interactive
text_prompt = args.text

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": text_prompt}
]

def invoke_model(messages_list):
    # ollama chat function call
    response = ollama.chat(
        model=model_name,
        messages=messages_list,
        think=use_reasoning,
        stream=True
    )
    return response

# Print 1st message
response = invoke_model(messages)
first_message = ""
for text_chunk in response:
    chunk_content = text_chunk.message.content
    first_message += chunk_content
    print(chunk_content, end="", flush=True)
print("")

if is_interactive:
    messages.append({"role": "assistant", "content": first_message})
    while (input_text := input(">>> ")) not in ["q","quit","exit"]:
        messages.append({"role": "user", "content": input_text})
        response = invoke_model(messages)
        assitant_message = ""
        for text_chunk in response:
            chunk_content = text_chunk.message.content
            assitant_message += chunk_content
            print(chunk_content, end="", flush=True)
        print("")
        messages.append({"role": "assistant", "content": assitant_message})