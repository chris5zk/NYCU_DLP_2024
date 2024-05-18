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
        self.d_model, self.d_head, self.num_heads = d_model, d_head, num_heads
        self.W_qkv = nn.Linear(d_model, 3*d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=attn_drop)
        
    def forward(self, x, mask=None):
        b, n, _ = x.size()
        
        qkv = self.W_qkv(x)
        qkv = qkv.reshape(b, n, self.num_heads, 3*self.d_head).permute(0, 2, 1, 3)  # (b, n, num_h, 3*dims) -> (b, num_h, n, 3*dims)
        q, k, v = qkv.chunk(3, dim=-1)  # (b, num_h, n, dim) x3
        
        alpha = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            alpha.masked_fill_(mask, -1e18)
        alpha = F.softmax(alpha, -1)
        alpha = self.dropout(alpha)
        
        output = torch.matmul(alpha, v)
        output = output.permute(0, 2, 1, 3)     # (b, n, h, dims)
        output = output.reshape(b, n, self.d_model)
        
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