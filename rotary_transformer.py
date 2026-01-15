import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        d = x.shape[-1]
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    # x is [B, T, nh, hs]
    # cos, sin are [T, hs] -> need broadcasting
    return (x * cos.unsqueeze(0).unsqueeze(2)) + (rotate_half(x) * sin.unsqueeze(0).unsqueeze(2))

class CausalSelfAttention(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        assert CONFIG['n_embd'] % CONFIG['n_head'] == 0
        self.head_dim = CONFIG['n_embd'] // CONFIG['n_head']
        self.n_head = CONFIG['n_head']
        
        # Merged QKV for speed and parameter density
        self.c_attn = nn.Linear(CONFIG['n_embd'], 3 * CONFIG['n_embd'], bias=False)
        self.c_proj = nn.Linear(CONFIG['n_embd'], CONFIG['n_embd'], bias=False)
        
        # RoPE (0 Params)
        self.rotary = RotaryPositionalEmbeddings(self.head_dim)
        
        # Causal Mask (Buffer, not parameter)
        self.register_buffer("bias", torch.tril(torch.ones(CONFIG['block_size'], CONFIG['block_size']))
                                     .view(1, 1, CONFIG['block_size'], CONFIG['block_size']))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape to [B, T, nh, hs]
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # v needs transpose for attention

        # Apply RoPE to Q and K (This replaces Absolute Positional Embeddings)
        cos, sin = self.rotary(q) # Calculate rotation matrix
        q = apply_rotary_pos_emb(q, cos, sin).transpose(1, 2) # [B, nh, T, hs]
        k = apply_rotary_pos_emb(k, cos, sin).transpose(1, 2) 

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class FeedForward(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG['n_embd'], 4 * CONFIG['n_embd'], bias=False),
            nn.ReLU(),
            nn.Linear(4 * CONFIG['n_embd'], CONFIG['n_embd'], bias=False),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.ln1 = nn.LayerNorm(CONFIG['n_embd'], bias=False)
        self.attn = CausalSelfAttention(CONFIG)
        self.ln2 = nn.LayerNorm(CONFIG['n_embd'], bias=False)
        self.ffwd = FeedForward(CONFIG)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class RecursiveGPT(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.CONFIG = CONFIG
        
        # 1. Token Embeddings ONLY (No Positional Embeddings Here!)
        self.token_embedding_table = nn.Embedding(CONFIG['vocab_size'], CONFIG['n_embd'])
        
        self.shared_block = Block(CONFIG)
        self.ln_f = nn.LayerNorm(CONFIG['n_embd'], bias=False)
        self.lm_head = nn.Linear(CONFIG['n_embd'], CONFIG['vocab_size'], bias=False)

        # 2. Weight Tying (Max Density)
        self.token_embedding_table.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embed tokens (No + pos_emb here, handled in Attention via RoPE)
        x = self.token_embedding_table(idx) 
        
        # Recursive Passes
        for _ in range(self.CONFIG['recursion']):
            x = self.shared_block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=None, greedy=False):
        if max_new_tokens is None:
            max_new_tokens = self.CONFIG['block_size']
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.CONFIG['block_size']:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if greedy:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop generation if we hit EOT (Optional but good for speed)
            # if idx_next.item() == self.CONFIG['eot_token_id']:
            #     break
        return idx