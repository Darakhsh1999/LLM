import os
import sys
sys.path.append(os.path.join(".."))

import torch
import torch.nn as nn
from ch4.gpt import GPT
from tqdm import tqdm
from torch.optim import AdamW
from ch2.data import GPTDataset
from collections import defaultdict
from torch.utils.data import Subset, DataLoader


def text2tokens(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded)[None] # [B=1,T]
    return encoded_tensor

def tokens2text(tokens, tokenizer):
    decoded = tokenizer.decode(tokens.squeeze().tolist())
    return decoded


def test(model, device, loss_fn, data_loader):

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (x,y) in data_loader:
            x, y = x.to(device), y.to(device) # move to appropriate device
            logits = model(x) # forward pass
            loss = loss_fn(logits.flatten(0,1),y.flatten()) # calculate loss
            val_loss += loss.item()
    return val_loss/len(data_loader)



def pretrain(train_config, model, optimizer, loss_fn, train_loader, val_loader, verbose=False):

    # Training variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training logs
    n_tokens = 0
    logs = defaultdict(list)

    if verbose:
        print(f"Started training with device: {device} | {len(train_loader)} train batches and {len(val_loader)} val batches.")

    
    for epoch_idx in tqdm(range(train_config["n_epochs"])):

        # Train loop
        train_loss = 0.0
        model.train()
        for batch_idx, (x,y) in tqdm(enumerate(train_loader)):

            n_tokens += x.numel()

            x, y = x.to(device), y.to(device) # move to appropriate device

            optimizer.zero_grad() # zero gradients

            logits = model(x) # forward pass

            loss = loss_fn(logits.flatten(0,1),y.flatten()) # calculate loss
            train_loss += loss.item()

            # Back propagate & update weights
            loss.backward()
            optimizer.step()
        train_loss = train_loss/len(train_loader)

        # Validation
        val_loss = test(model, device, loss_fn, val_loader)

        print(f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f}")

        # Log statistics
        logs["n_tokens"].append(n_tokens)
        logs["train_loss"].append(train_loss)
        logs["val_loss"].append(val_loss)

        # Generate
        model.eval()
        print(model.generate("Today I will", max_new_tokens=10, decode=True, device=device))
    
    return 




if __name__ == "__main__":


    # Train config
    train_config = {
        "n_epochs": 10,

    }

    # GPT config for GPT2-small
    gpt_config = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "d_e": 768, # Embedding dimension
        "n_heads": 12, # Number of attention heads
        "n_layers": 12, # Number of transformer block layers
        "p_dropout": 0.1, # Dropout rate
        "qkv_bias": False, # Query/Key/Value bias in linear layers
    }

    # GPT model
    model = GPT(config=gpt_config)
    model.eval()

    # Optimizer & loss function
    optimizer = AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()

    # Load data 
    text_path = os.path.join("..","..","data","txt","the-verdict.txt")
    data = GPTDataset(text_path, n_ctx=256, stride=256)
    train_data = Subset(data, indices=range(0,int(0.9*len(data))))
    val_data = Subset(data, indices=range(int(0.9*len(data)), len(data)))

    # Dataloader
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, drop_last=False)

    # Pretrain model
    pretrain(train_config, model, optimizer, loss_fn, train_loader, val_loader, verbose=True)


    # Save model
    model.to("cpu")
    torch.save(model.state_dict(), "gpt2.pth")