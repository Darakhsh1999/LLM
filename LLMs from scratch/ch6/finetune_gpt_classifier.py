import os
import sys
sys.path.append(os.path.join(".."))

from ch5.load_openai import load_pretrained_gpt2_model









if __name__ == "__main__":

    # GPT config for GPT2-small
    gpt_config = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "d_e": 768, # Embedding dimension
        "n_heads": 12, # Number of attention heads
        "n_layers": 12, # Number of transformer block layers
        "p_dropout": 0.1, # Dropout rate
        "qkv_bias": True, # Query/Key/Value bias in linear layers
    }

    model = load_pretrained_gpt2_model(gpt_config=gpt_config) # load in pre-trained GPT2 

    # Freeze model layers
    for param in model.parameters():
        param.requires_grad = False

    
    # Change output mapping