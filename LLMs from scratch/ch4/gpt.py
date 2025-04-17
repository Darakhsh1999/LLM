import os
import sys
sys.path.append(os.path.join(".."))
import torch
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiLayerPerceptron(nn.Module):

    def __init__(self, config:dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["d_e"], 4*config["d_e"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4*config["d_e"], config["d_e"])
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):


    def __init__(self, config:dict):
        super().__init__()

        assert config["d_e"] % config["n_heads"] == 0, "d_e must be divisble by n_heads"

        self.d_e = config["d_e"]
        self.n_heads = config["n_heads"]
        self.d_h = config["d_e"] // config["n_heads"] # head_dimension = model_dimension / n_heads

        assert self.d_e == self.d_h*self.n_heads, "d_e != d_h * n_heads"

        self.W_q = nn.Linear(config["d_e"], config["d_e"], bias=config["qkv_bias"])
        self.W_k = nn.Linear(config["d_e"], config["d_e"], bias=config["qkv_bias"])
        self.W_v = nn.Linear(config["d_e"], config["d_e"], bias=config["qkv_bias"])
        self.head_output_projection = nn.Linear(config["d_e"], config["d_e"])

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config["context_length"], config["context_length"]), diagonal=1) # [T,T]
        )


    def forward(self,x):
        B, T, _ = x.shape # [B,T,d_in]

        # Query, Key, Value
        q = self.W_q(x) # [B,T,d_out]
        k = self.W_k(x) # [B,T,d_out]
        v = self.W_v(x) # [B,T,d_out]

        # Split last dimension from d_out -> (n_heads, d_h)
        q = q.view(B,T,self.n_heads,self.d_h) # [B,T,n_heads,d_h]
        k = k.view(B,T,self.n_heads,self.d_h) # [B,T,n_heads,d_h]
        v = v.view(B,T,self.n_heads,self.d_h) # [B,T,n_heads,d_h]

        # Transpose token dimension with head dimension
        q = q.transpose(1,2) # [B,n_heads,T,d_h]
        k = k.transpose(1,2) # [B,n_heads,T,d_h]
        v = v.transpose(1,2) # [B,n_heads,T,d_h]

        # Unnormalized attention scores
        attention_scores = q @ k.transpose(2,3) # [B,n_heads,T,d_h]@[B,n_heads,d_h,T] -> [B,n_heads,T,T]

        # Masking
        boolean_mask = self.mask.bool()[:T,:T] # [T,T]

        # Fill mask
        attention_scores.masked_fill_(boolean_mask, -torch.inf) # [B,n_heads,T,T]

        # Normalize attention scores
        assert k.shape[-1] == self.d_h
        attention_weights = torch.softmax(attention_scores / self.d_h**0.5, dim=-1) # [B,n_heads,T,T]

        # Context vector
        context_vector =  (attention_weights @ v).transpose(1,2) # [B,n_heads,T,d_h] -> [B,T,n_heads,d_h]
        context_vector = context_vector.contiguous().view(B,T,self.d_e) # [B,T,d_out]
        context_vector = self.head_output_projection(context_vector)

        return context_vector
    

class TransformerBlock(nn.Module):

    def __init__(self, config:dict):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MultiLayerPerceptron(config)
        self.ln1 = nn.LayerNorm(config["d_e"])
        self.ln2 = nn.LayerNorm(config["d_e"])
        self.dropout = nn.Dropout(config["p_dropout"])

    def forward(self, x):
        skip_connection = x
        x = self.attention(self.ln1(x))
        x = self.dropout(x)
        x = x + skip_connection

        skip_connection = x
        x = self.mlp(self.ln2(x))
        x = self.dropout(x)
        x = x + skip_connection
        return x


class GPT(nn.Module):

    def __init__(self, config:dict):
        super().__init__()
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_e"])
        self.position_embedding = nn.Embedding(config["context_length"], config["d_e"])
        self.dropout = nn.Dropout(config["p_dropout"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        self.output_head_ln = nn.LayerNorm(config["d_e"])
        self.output_head = nn.Linear(config["d_e"], config["vocab_size"])

    def forward(self, token_idx:Tensor) -> Tensor:
        B, T = token_idx.shape # B = batch size, T = sequence length 

        # Token and position embedding
        token_emb = self.token_embedding(token_idx)
        position_emb = self.position_embedding(torch.arange(T, device=token_idx.device))
        embedding = self.dropout(token_emb + position_emb)

        # Transformer blocks
        transformer_output = self.transformer_blocks(embedding)

        # Output mapping
        logits = self.output_head(self.output_head_ln(transformer_output))
        return logits

    def decode(self, x):
        return self.tokenizer.decode(x)

    def encode(self, input_string):
        return self.tokenizer.encode(input_string)

    @torch.no_grad
    def generate(self, _input, max_new_tokens=10, decode=False, greedy=False, temperature=1.0, device=None):
        
        device = "cpu" if device is None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(_input, str): # Tokenize if input is string
            token_input = torch.Tensor(self.encode(_input)).type(torch.LongTensor)[None,:] # [B,T]
        else:
            token_input = _input

        self.eval()
        for _ in range(max_new_tokens):

            truncated_tokens = token_input[:, -self.config["context_length"]:] # Incase we exceed context length
            truncated_tokens = truncated_tokens.to(device)

            logits = self.forward(truncated_tokens) # [B,T] -> [B,T,vocab_size]
            last_token_logits = logits[:, -1, :] # logits for last  [B,vocab_size]

            if greedy:
                idx_next = torch.argmax(last_token_logits, dim=-1, keepdim=True).cpu() # [B,1]
            else: # TopK and temperature
                last_token_logits = last_token_logits/ temperature
                topk_logits, _ = torch.topk(last_token_logits, k=50)
                last_token_logits = torch.where(
                    condition= last_token_logits < topk_logits[:,-1],
                    input=torch.tensor(float("-inf")),
                    other=last_token_logits
                )
                topk_probas = F.softmax(last_token_logits, dim=-1) # [B,vocab_size]
                idx_next = torch.multinomial(topk_probas, num_samples=1).cpu() # [B,1]
            token_input = torch.cat((token_input, idx_next), dim=1)
            
        if decode:
            return self.tokenizer.decode(token_input.squeeze().tolist())
        else: 
            return token_input



if __name__ == "__main__":

    # GPT config for GPT2-small
    config = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 1024, # Context length
        "d_e": 768, # Embedding dimension
        "n_heads": 12, # Number of attention heads
        "n_layers": 12, # Number of transformer block layers
        "p_dropout": 0.1, # Dropout rate
        "qkv_bias": False, # Query/Key/Value bias in linear layers
    }
    
    example_input = torch.randint(0, config["vocab_size"], (2,1024))

    gpt = GPT(config)

    print(gpt(example_input).shape)

    input_string = "Hello I am a"
    output = gpt.generate(input_string, decode=True)
    print(output)