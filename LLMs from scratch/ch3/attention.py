import os
import sys
sys.path.append(os.path.join(".."))
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):


    def __init__(self, d_in, d_out, l_context, n_heads, qkv_bias=False):
        super().__init__()

        assert d_out % n_heads == 0, "d_out must be divisble by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.d_h = d_out // n_heads # head_dimension = model_dimension / n_heads

        assert d_out == self.d_h*n_heads, "d_h != d_h * n_heads"

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(l_context, l_context), diagonal=1) # [T,T]
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
        # optionally we can apply dropout here

        context_vector =  (attention_weights @ v).transpose(1,2) # [B,n_heads,T,d_h] -> [B,T,n_heads,d_h]
        context_vector = context_vector.contiguous().view(B,T,self.d_out) # [B,T,d_out]


        return context_vector











if __name__ == "__main__":
    

    x_input = torch.randn((5,50,512)) # [B,T,d_in],  0 < T <= 100

    mha = MultiHeadAttention(d_in=512, d_out=512, l_context=100, n_heads=8)

    output = mha(x_input)
    print(output.shape)
