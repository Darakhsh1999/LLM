import os
import sys
sys.path.append(os.path.join(".."))

import torch
import torch.nn as nn
from tqdm import tqdm

from ch5.load_openai import load_pretrained_gpt2_model
from finetune_data import MovieData
from torch.utils.data import random_split, DataLoader


@torch.no_grad
def test(model, device, loss_fn, data_loader):

    model.eval()
    val_loss = 0.0
    n_correct = 0.0
    n_predictions = 0.0
    for (x,y) in data_loader:

        x, y = x.to(device), y.to(device) # move to appropriate device

        logits = model(x) # forward pass
        prediction_token = logits[:,-1,:]

        # calculate loss
        loss = loss_fn(prediction_token,y) # calculate loss
        val_loss += loss.item()

        # calculate accuracy
        predicted_classes = torch.argmax(prediction_token, dim=-1)
        assert predicted_classes.shape == y.shape, f"{predicted_classes.shape} != {y.shape}"
        n_correct += (predicted_classes == y).sum().item()
        n_predictions += len(y)

    accuracy = n_correct/n_predictions
    return val_loss/len(data_loader), accuracy


def finetune(train_config, model, optimizer, loss_fn, train_loader, val_loader, verbose=False):

    # Training variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if verbose:
        print(f"Started fine-tuning with device: {device} | {len(train_loader)} train batches and {len(val_loader)} val batches.")

    for epoch_idx in range(train_config["n_epochs"]):

        # Train loop
        train_loss = 0.0
        model.train()
        for batch_idx, (x,y) in enumerate(tqdm(train_loader, desc="Batch", leave=True)):

            x, y = x.to(device), y.to(device) # move to appropriate device

            optimizer.zero_grad() # zero gradients

            logits = model(x) # forward pass
            prediction_token = logits[:,-1,:]

            loss = loss_fn(prediction_token,y) # calculate loss
            train_loss += loss.item()

            # Back propagate & update weights
            loss.backward()
            optimizer.step()
        train_loss = train_loss/len(train_loader)

        # Validation
        val_loss, val_acc = test(model, device, loss_fn, val_loader)

        print(f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | Val accuracy: {val_acc:.3f}")



if __name__ == "__main__":

    finetune_config = {
        "n_epochs": 15,
        "batch_size": 32
    }

    # Load in data set
    data_dir = os.path.join(".")
    data = MovieData(data_dir=data_dir)
    train_data, val_data, test_data = random_split(data, [0.8, 0.1, 0.1])

    # Data Loaders 
    train_loader = DataLoader(
        train_data,
        batch_size=finetune_config["batch_size"],
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=finetune_config["batch_size"],
        shuffle=False,
        drop_last=False
    )
    test_loader = DataLoader(
        test_data,
        batch_size=finetune_config["batch_size"],
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

    # Freeze model layers
    for param in model.parameters():
        param.requires_grad = False

    
    # Change output mapping
    model.output_head = nn.Linear(gpt_config["d_e"], data.n_classes)


    # Unfreeze last transformer block
    for param in model.transformer_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.output_head_ln.parameters():
        param.requires_grad = True

    # Example input
    example_input = torch.tensor(data.tokenizer.encode("This movie is action packed with crazy explosions"), dtype=torch.long)[None] # [B=1,T]
    print("IN:",example_input.shape)
    with torch.no_grad():
        example_output = model(example_input) # [B=1,T,n_classes]
        print("OUT:",example_output.shape)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Fine tune model for movie classification
    finetune(finetune_config, model, optimizer, loss_fn, train_loader, val_loader, verbose=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss, test_accuracy = test(model, device, loss_fn, test_loader)
    print(f"Test loss: {test_loss:.3f} | Test accuracy: {test_accuracy:.3f}")

