import os
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):

    def __init__(self, text_path, n_ctx=1024, stride=1024):

        # Load text
        with open(text_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.input_ids = []
        self.target_ids = []

        token_ids = self.tokenizer.encode(self.text) # tokenized text

        # Create input and target mappings
        for i in range(0, len(token_ids) - n_ctx, stride):
            input_chunk = token_ids[i:(i+n_ctx)]
            target_chunk = token_ids[(i+1):(i+n_ctx+1)]
            if len(input_chunk) != len(target_chunk): break # target_chunk indexes outside of token_ids and is not equal to n_ctx
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self): 
        return len(self.input_ids)
        
    def __getitem__(self, idx): 
        return self.input_ids[idx], self.target_ids[idx]
    

if __name__ == "__main__":

    text_path = os.path.join("..","..","data","txt","the-verdict.txt")
    text_dataset = GPTDataset(text_path,  n_ctx=1024, stride=1024)
