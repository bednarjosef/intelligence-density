import torch
import torch.nn as nn
from torch.nn import functional as F


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return [1.0 / (x ** (i + 1)) for i in range(num_heads)]

class Head(nn.Module):
    def __init__(self, CONFIG, head_index):
        super().__init__()
        self.CONFIG = CONFIG
        n_embd, n_head = CONFIG['n_embd'], CONFIG['n_head']
        head_size = n_embd // n_head
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CONFIG['block_size'], CONFIG['block_size'])))
        
        slopes = get_alibi_slope(n_head)
        self.slope = slopes[head_index]

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        
        indices = torch.arange(T, device=self.CONFIG['device'])
        distance = indices[None, :] - indices[:, None]
        
        alibi_bias = distance * self.slope
        
        wei += alibi_bias

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        n_embd, n_head = CONFIG['n_embd'], CONFIG['n_head']
        self.heads = nn.ModuleList([Head(CONFIG, i) for i in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        n_embd = CONFIG['n_embd']
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        n_embd = CONFIG['n_embd']
        self.sa = MultiHeadAttention(CONFIG)
        self.ffwd = FeedForward(CONFIG)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class RecursiveGPT(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.CONFIG = CONFIG
        self.token_embedding_table = nn.Embedding(CONFIG['vocab_size'], CONFIG['n_embd'])
        
        self.shared_block = Block(CONFIG)
        self.ln_f = nn.LayerNorm(CONFIG['n_embd'])
        self.lm_head = nn.Linear(CONFIG['n_embd'], CONFIG['vocab_size'])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        
        x = tok_emb 
        
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
            
            loss_elements = F.cross_entropy(logits, targets, reduction='none')
            mask = (targets != self.CONFIG['pad_token_id']).float()
            
            # Apply mask
            loss = (loss_elements * mask).sum() / mask.sum()

        return logits, loss
        
    def generate(self, idx, max_new_tokens=None):
        if max_new_tokens == None:
            max_new_tokens = self.CONFIG['block_size']
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.CONFIG['block_size']:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    