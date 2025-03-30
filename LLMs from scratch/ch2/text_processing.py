import os
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

# Load in .txt file
file_path = os.path.join("..","..","data","txt","the-verdict.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()


# Byte pair encoding (BPE)
tokenizer = tiktoken.get_encoding("gpt2")
example_text = "Hello this is an example text that we will tokenize using BPE. It's an tokenizer used for GPT-2 and other LLM models."
token_ids = tokenizer.encode(example_text)
reconstructed_text = tokenizer.decode(token_ids)
print(f"The encoded text has token IDs: \n{token_ids}\nAnd after econstruction\n{reconstructed_text}")
print(10*"-")

# Made up word
fantasy_word = "Abrakadabruh"
print(tokenizer.encode(fantasy_word))
print([tokenizer.decode([_token_id]) for _token_id in tokenizer.encode(fantasy_word)])
print(tokenizer.decode(tokenizer.encode(fantasy_word)))


# The verdict example problem
encoded_text = tokenizer.encode(text)
n_tokens = len(encoded_text)
print(f"The Verdict short story is made up of {n_tokens} tokens.")
encoded_sample = encoded_text[50:] # remove first 50 tokens

class GPTDatasetV1(Dataset):

    def __init__(self, text, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self): 
        return len(self.input_ids)
        
    def __getitem__(self, idx): 
        return self.input_ids[idx], self.target_ids[idx]


def create_gptv1_dataloader(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader

verdict_dataloader = create_gptv1_dataloader(text, batch_size=1, max_length=4, stride=1, shuffle=False)
dataloader_iterator = iter(verdict_dataloader)
print(next(dataloader_iterator))
print(next(dataloader_iterator))


# Embedding token IDs
vocab_size = 6
embedding_dim = 3
torch.manual_seed(123)
embed_layer = torch.nn.Embedding(vocab_size, embedding_dim=embedding_dim)
print(embed_layer(torch.tensor([3]))) # Example embbeding of token_id = 3
print(embed_layer(torch.tensor([2,3,5,1]))) # Example embbeding of sentence (4,embed_dim)
print(10*"-")

# Realistic token embedding
vocab_size = 50257
embedding_dim = 256
embed_layer = torch.nn.Embedding(vocab_size, embedding_dim=embedding_dim)
dataloader = create_gptv1_dataloader(text, batch_size=8, max_length=4, stride=4, shuffle=False, drop_last=True)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
token_embeddings = embed_layer(inputs)
print(token_embeddings.shape)

# Positional embedding
context_length = 4
pos_embedding_layer = torch.nn.Embedding(4, embedding_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)