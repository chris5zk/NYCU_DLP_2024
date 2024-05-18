import torch.nn as nn
import torch.nn.functional as F
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        assert(dim % num_heads == 0)
        self.attention = scaled_dot_product_attention(dim, dim//num_heads, num_heads, attn_drop)
        self.linear_proj = nn.Linear(dim, dim)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        out = self.attention(x)
        out = self.linear_proj(out)
        return out
    
class scaled_dot_product_attention(nn.Module):
    def __init__(self, d_model, d_head, num_heads, attn_drop) -> None:
        super().__init__()
        dim_aggr = d_head * num_heads
        self.scale = torch.sqrt(torch.tensor(d_model))
        self.W_q, self.W_k, self.W_v = nn.Linear(d_model, dim_aggr), nn.Linear(d_model, dim_aggr), nn.Linear(d_model, dim_aggr)
        self.W_o = nn.Linear(d_model, dim_aggr)
        self.dropout = nn.Dropout(p=attn_drop)
        
    def forward(self, x, mask=None):
        # input x: (b, N, d_model)
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)     # (b, N, d_aggr)
        
        # scale score of Q, K: (b, N, N)
        alpha = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        
        # masking(opt.)
        if mask is not None:
            alpha.masked_fill_(mask, -1e18)
        
        # softmax
        alpha = F.softmax(alpha, -1)
        alpha = self.dropout(alpha)
        
        # alpha @ value: (b, N, d_aggr)
        output = torch.bmm(alpha, V)
        
        return output
    
class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    

if __name__ == '__main__':
    attention_layer = MultiHeadAttention(15, 3)
    
    x = torch.rand(2, 4, 15)
    print(x.shape)
    
    out = attention_layer(x)
    print(out.shape)