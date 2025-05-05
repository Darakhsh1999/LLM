from transformers import AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")

smol_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")



def convert_to_chatml(example):
    return {
        "messages": [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }


example = dataset["train"][0]

smol_chat = smol_tokenizer.apply_chat_template(example["messages"], tokenize=False)

print(smol_chat)