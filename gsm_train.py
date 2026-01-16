import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import wandb 
import re

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
chars = "\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# EOT Token ID (The tilde '~')
EOS_TOKEN_ID = stoi['~']

CONFIG = {
    'project_name': 'gsm-gpt',
    'n_embd': 288,
    'n_head': 12,
    'block_size': 1024,
    'recursion': 8,
    'vocab_size': vocab_size,
    'dropout': 0.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'batch_size': 48,
    'lr': 3e-4,
    'max_iters': 20000,
    'eval_interval': 500,
    'ignore_index': -100,
}

# -----------------------------------------------------------------------------
# 2. The Model
# -----------------------------------------------------------------------------
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
    return (x * cos.unsqueeze(0).unsqueeze(2)) + (rotate_half(x) * sin.unsqueeze(0).unsqueeze(2))

class CausalSelfAttention(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.head_dim = CONFIG['n_embd'] // CONFIG['n_head']
        self.n_head = CONFIG['n_head']
        self.c_attn = nn.Linear(CONFIG['n_embd'], 3 * CONFIG['n_embd'], bias=False)
        self.c_proj = nn.Linear(CONFIG['n_embd'], CONFIG['n_embd'], bias=False)
        self.rotary = RotaryPositionalEmbeddings(self.head_dim)
        self.register_buffer("bias", torch.tril(torch.ones(CONFIG['block_size'], CONFIG['block_size']))
                                     .view(1, 1, CONFIG['block_size'], CONFIG['block_size']))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(q)
        q = apply_rotary_pos_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_pos_emb(k, cos, sin).transpose(1, 2)
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
    def forward(self, x): return self.net(x)

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
        self.token_embedding_table = nn.Embedding(CONFIG['vocab_size'], CONFIG['n_embd'])
        self.shared_block = Block(CONFIG)
        self.ln_f = nn.LayerNorm(CONFIG['n_embd'], bias=False)
        self.lm_head = nn.Linear(CONFIG['n_embd'], CONFIG['vocab_size'], bias=False)
        self.token_embedding_table.weight = self.lm_head.weight 

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        for _ in range(self.CONFIG['recursion']):
            x = self.shared_block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=CONFIG['ignore_index'])
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, stream=True):
        self.eval()
        prompt_str = decode(idx[0].tolist())
        if stream:
            print(prompt_str, end='', flush=True) 
        
        generated_text = ""
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.CONFIG['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # --- STOP GENERATION IF EOS TOKEN IS PRODUCED ---
            if idx_next.item() == EOS_TOKEN_ID:
                break
            
            idx = torch.cat((idx, idx_next), dim=1)
            
            char = itos.get(idx_next.item(), '')
            generated_text += char
            
            if stream:
                print(char, end='', flush=True)
            
        if stream:
            print() 
        self.train()
        return prompt_str + generated_text

# -----------------------------------------------------------------------------
# 3. Data Pipeline (High Performance GPU Loading)
# -----------------------------------------------------------------------------
def encode(s):
    return [stoi.get(c, 0) for c in s]

def decode(l):
    return ''.join([itos.get(i, '') for i in l])

print("Loading compressed data...")
try:
    data_train = torch.load('train_data_uint8.pt')
    data_val = torch.load('val_data_uint8.pt')
except FileNotFoundError:
    print("Error: Data files not found. Please run prepare_data.py first.")
    exit()

print(f"Moving dataset to {CONFIG['device']}...")
train_inputs = data_train['inputs'].to(CONFIG['device']) 
train_labels = data_train['labels'].to(CONFIG['device'])
val_inputs = data_val['inputs'].to(CONFIG['device'])
val_labels = data_val['labels'].to(CONFIG['device'])
print("Data loaded!")

def get_batch(split, block_size, batch_size):
    if split == 'train': inputs, labels = train_inputs, train_labels
    else: inputs, labels = val_inputs, val_labels
    
    ix = torch.randint(len(inputs) - block_size, (batch_size,), device=CONFIG['device'])
    offsets = torch.arange(block_size, device=CONFIG['device'])
    idx_matrix = ix.unsqueeze(1) + offsets
    
    x = inputs[idx_matrix]
    y = labels[idx_matrix + 1]
    
    x = x.to(dtype=torch.long)
    y = y.to(dtype=torch.long)
    
    # 100 was our masking value in prepare_data, mapping it back to -100 for PyTorch
    y[y == 100] = -100 
    
    return x, y

# -----------------------------------------------------------------------------
# 5. Training Loop
# -----------------------------------------------------------------------------
def train():
    wandb.init(
        project=CONFIG['project_name'], 
        config=CONFIG
    )
    
    print(f"Initializing RecursiveGPT (Widened) ~{sum(p.numel() for p in RecursiveGPT(CONFIG).parameters())/1e6:.1f}M Params")
    model = RecursiveGPT(CONFIG)
    model.to(CONFIG['device'])

    print("Compiling model...")
    model = torch.compile(model)
    torch.set_float32_matmul_precision('high')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    model.train()
    iter_num = 0
    t0 = time.time()
    
    print("\nStarting Training Loop...")

    while iter_num < CONFIG['max_iters']:
        X, Y = get_batch('train', CONFIG['block_size'], CONFIG['batch_size'])
        
        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        iter_num += 1
        
        # Logging to Console & WandB
        if iter_num % 100 == 0:
            dt = time.time() - t0
            t0 = time.time()
            loss_val = loss.item()
            
            # --- CALCULATE METRICS ---
            tokens_seen = iter_num * CONFIG['batch_size'] * CONFIG['block_size']
            examples_seen = iter_num * CONFIG['batch_size']
            
            print(f"step {iter_num} | train_loss {loss_val:.4f} | time {dt:.2f}s")
            
            # --- LOG DETAILED METRICS ---
            wandb.log({
                "train/loss": loss_val, 
                "train/step": iter_num,
                "train/tokens_seen": tokens_seen,
                "train/examples_seen": examples_seen
            })
            
        # Evaluation
        if iter_num % CONFIG['eval_interval'] == 0:
            print("\n" + "="*50)
            
            losses = []
            for _ in range(10): 
                X_val, Y_val = get_batch('val', CONFIG['block_size'], CONFIG['batch_size'])
                with torch.no_grad():
                    _, v_loss = model(X_val, Y_val)
                losses.append(v_loss.item())
            val_loss = sum(losses) / len(losses)
            print(f"VALIDATION LOSS: {val_loss:.4f}")
            
            wandb.log({"val/loss": val_loss, "val/step": iter_num})
            
            print("\n--- Model Thinking Process ---")
            test_prompt = "Problem: There are 5 birds on a tree. 3 fly away. How many are left?\nSolution:"
            context = torch.tensor([encode(test_prompt)], dtype=torch.long, device=CONFIG['device'])
            
            # Use max_new_tokens to prevent infinite generation, but model should stop early on '~'
            full_generation = model.generate(context, max_new_tokens=512, stream=True)
            
            sample_table = wandb.Table(columns=["step", "generation"])
            sample_table.add_data(iter_num, full_generation)
            wandb.log({"generations": sample_table})

            print("="*50 + "\n")

    print("Training Complete.")
    torch.save(model.state_dict(), "recursive_gsm_1m.pth")
    wandb.save("recursive_gsm_1m.pth") 
    wandb.finish()

if __name__ == '__main__':
    train()