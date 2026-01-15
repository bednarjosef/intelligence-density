import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import math
import time
import sys

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
chars = "\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

CONFIG = {
    'n_embd': 288,
    'n_head': 6,
    'block_size': 512,
    'recursion': 4,
    'vocab_size': vocab_size,
    'dropout': 0.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'batch_size': 64,
    'lr': 1e-3,
    'max_iters': 5000,
    'eval_interval': 500,
    
    # New Configs
    'val_samples': 1000,   # Reserve first 1000 samples for validation
    'ignore_index': -100,  # PyTorch default for ignoring targets
}

# -----------------------------------------------------------------------------
# 2. The Model (With Loss Handling)
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
            # PyTorch CrossEntropyLoss handles the ignore_index automatically
            loss = F.cross_entropy(logits, targets, ignore_index=CONFIG['ignore_index'])
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100):
        self.eval()
        # Decode the prompt first to show context
        print(decode(idx[0].tolist()), end='', flush=True) 
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.CONFIG['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Streaming print with flush=True
            char = itos.get(idx_next.item(), '')
            print(char, end='', flush=True)
            
        print() # Newline at end
        self.train()
        return idx

# -----------------------------------------------------------------------------
# 3. Data Pipeline (Streaming, Filtering, Masking, Splitting)
# -----------------------------------------------------------------------------
def encode(s):
    return [stoi.get(c, 0) for c in s]

def decode(l):
    return ''.join([itos.get(i, '') for i in l])

class GSMStreamDataset(IterableDataset):
    def __init__(self, block_size, mode='train', limit=None):
        self.block_size = block_size
        self.mode = mode
        self.limit = limit
        # Streaming load
        self.dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M", streaming=True)
        
    def __iter__(self):
        buffer = []
        skip_count = CONFIG['val_samples']
        count = 0
        
        for i, sample in enumerate(self.dataset):
            # FILTER: Source Check
            if sample['problem_source'] != 'augmented_gsm8k':
                continue
            
            # SPLIT LOGIC:
            # If train mode: skip the first N samples
            # If val mode: take ONLY the first N samples
            if self.mode == 'train':
                if i < skip_count: continue
            elif self.mode == 'val':
                if i >= skip_count: break
                if self.limit and count >= self.limit: break
            
            count += 1
            
            # 1. Format Text
            # We add a clear separator "Solution:" which we will search for later
            problem_text = f"Problem: {sample['problem']}\n"
            solution_text = f"Solution: {sample['generated_solution']}\n\n"
            full_text = problem_text + solution_text
            
            # 2. Tokenize
            tokens = encode(full_text)
            
            # 3. Create Masked Targets
            # Targets are same as input, BUT we set the "Problem" part to -100
            targets = list(tokens) # Copy
            
            # Find index where solution starts
            # encode("Solution:") length is 9. We find where the problem text ends.
            sep_len = len(encode(problem_text))
            
            # Mask everything up to the solution start
            # (Set to -100 so CrossEntropy ignores it)
            for j in range(sep_len):
                targets[j] = CONFIG['ignore_index']
                
            # Add to buffer
            # We store tuples: (input_token, target_token)
            for t, tgt in zip(tokens, targets):
                buffer.append((t, tgt))
            
            # Yield chunks
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size + 1:]
                
                # Unzip back into X and Y
                # X is input (0 to N-1)
                # Y is target (1 to N) which includes masks
                
                chunk_tokens = [x[0] for x in chunk]
                chunk_targets = [x[1] for x in chunk]
                
                x_tensor = torch.tensor(chunk_tokens[:-1], dtype=torch.long)
                y_tensor = torch.tensor(chunk_targets[1:], dtype=torch.long)
                
                yield x_tensor, y_tensor

# -----------------------------------------------------------------------------
# 4. Helpers for Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(model):
    model.eval()
    loader = DataLoader(GSMStreamDataset(CONFIG['block_size'], mode='val', limit=50), batch_size=CONFIG['batch_size'])
    losses = []
    print("Calculating Validation Loss...")
    for X, Y in loader:
        X, Y = X.to(CONFIG['device']), Y.to(CONFIG['device'])
        _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if len(losses) > 0 else 0.0

# -----------------------------------------------------------------------------
# 5. Training Loop
# -----------------------------------------------------------------------------
def train():
    print(f"Initializing RecursiveGPT (Widened) ~{sum(p.numel() for p in RecursiveGPT(CONFIG).parameters())/1e6:.1f}M Params")
    model = RecursiveGPT(CONFIG)
    model.to(CONFIG['device'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    # Train Loader (Mode='train' skips first 1000)
    train_dataset = GSMStreamDataset(CONFIG['block_size'], mode='train')
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'])
    
    model.train()
    iter_num = 0
    t0 = time.time()
    
    print("\nStarting Training Loop...")
    
    for X, Y in train_loader:
        if iter_num >= CONFIG['max_iters']: break
        
        X, Y = X.to(CONFIG['device']), Y.to(CONFIG['device'])
        
        # Forward
        logits, loss = model(X, Y)
        
        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Added gradient clipping for safety
        optimizer.step()
        
        iter_num += 1
        
        # Logging
        if iter_num % 100 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"step {iter_num} | train_loss {loss.item():.4f} | time {dt:.2f}s")
            
        # Evaluation
        if iter_num % CONFIG['eval_interval'] == 0:
            print("\n" + "="*50)
            
            # 1. Validation Loss
            val_loss = evaluate_loss(model)
            print(f"VALIDATION LOSS: {val_loss:.4f}")
            
            # 2. Visual Sample
            print("\n--- Model Thinking Process ---")
            # Create a dummy prompt for the model to complete
            test_prompt = "Problem: There are 5 birds on a tree. 3 fly away. How many are left?\nSolution:"
            context = torch.tensor([encode(test_prompt)], dtype=torch.long, device=CONFIG['device'])
            
            model.generate(context, max_new_tokens=150)
            print("="*50 + "\n")

    print("Training Complete.")
    torch.save(model.state_dict(), "recursive_gsm_1m.pth")

if __name__ == '__main__':
    train()