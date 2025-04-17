import os
import sys
sys.path.append(os.path.join(".."))

import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial

from ch5.load_openai import load_pretrained_gpt2_model
from torch.utils.data import random_split, DataLoader
from instruction_data import InstructionDataset, collate_fn


@torch.no_grad
def test(model, device, loss_fn, data_loader):

    model.eval()
    val_loss = 0.0
    for (x,y) in data_loader:
        x, y = x.to(device), y.to(device) # move to appropriate device
        logits = model(x) # forward pass
        assert not logits.requires_grad, "Gradients tracked in test function"
        loss = loss_fn(logits.flatten(0,1),y.flatten()) # calculate loss
        val_loss += loss.item()
    return val_loss/len(data_loader)


def finetune(train_config, model, optimizer, loss_fn, train_loader, val_loader, verbose=False):

    # Training variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if verbose:
        print(f"Started instruction finetuning with device: {device} | {len(train_loader)} train batches and {len(val_loader)} val batches.")

    for epoch_idx in range(train_config["n_epochs"]):

        # Train loop
        train_loss = 0.0
        model.train()
        for batch_idx, (x,y) in enumerate(tqdm(train_loader, desc="Batch", leave=True)):

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



if __name__ == "__main__":

    finetune_config = {
        "n_epochs": 8
    }

    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Data set
    data = InstructionDataset()
    train_data, val_data, test_data = random_split(data, [0.8, 0.1, 0.1])
    custom_collate_fn = partial(collate_fn, device=device, allowed_max_length=1024)


    # Data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False
    )


    # GPT config for GPT2-small
    gpt_config = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "d_e": 768, # Embedding dimension
        "n_heads": 12, # Number of attention heads
        "n_layers": 12, # Number of transformer block layers
        "p_dropout": 0.0, # Dropout rate
        "qkv_bias": True, # Query/Key/Value bias in linear layers
    }

    model = load_pretrained_gpt2_model(gpt_config=gpt_config) # load in pre-trained GPT2 


    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Fine tune model for movie classification
    finetune(finetune_config, model, optimizer, loss_fn, train_loader, val_loader, verbose=True)

    test_loss = test(model, device, loss_fn, test_loader)
    print(f"Test loss {test_loss:.3f}")


    input = torch.tensor(test_data[10], dtype=torch.long)[None]
    model_output = model.generate(input, max_new_tokens=55, decode=True, device=device)
    eos_idx = model_output.find("<|endoftext|>")
    if eos_idx == -1:
        print(model_output)
    else:
        print(model_output[:eos_idx])
    

    # Save model
    model.to("cpu")
    torch.save(model.state_dict(), "gpt2-instruct-finetuned.pth")