import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# --- 1. CONFIGURATION ---
CONFIG = {
    'batch_size': 64,
    'block_size': 512,  # INCREASED: 10-digit math chain-of-thought is long!
    'n_embd': 16,       # Tiny embedding (High density)
    'n_head': 4,        # 4 heads for multitasking
    'recursion': 4,     # Recurse 4 times (Deep thinking, small memory)
    'lr': 5e-3,         # Lower learning rate for stability
    'max_iters': 70000, 
    'eval_interval': 500, 
    'max_digits': 10,   # Train on up to 10-digit numbers
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Running on {CONFIG['device']}...")

# --- 2. DATA GENERATION & TOKENIZER ---
# Added '.' as the STOP TOKEN
chars = "0123456789+= C." 
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

def generate_batch(split, batch_size, current_max_digits):
    # current_max_digits: The hardest problem allowed right now (e.g., 3 or 10)
    X, Y = [], []
    
    # Store raw sequences to find max length later
    raw_seqs = []
    
    for _ in range(batch_size):
        # Pick random difficulty up to the current limit
        d = random.randint(1, current_max_digits)
        
        a = random.randint(0, 10**d - 1)
        b = random.randint(0, 10**d - 1)
        
        a_str = f"{a:0{d}d}"
        b_str = f"{b:0{d}d}"
        
        scratchpad = []
        carry = 0
        ans_digits = []
        
        for i in range(d-1, -1, -1):
            da = int(a_str[i])
            db = int(b_str[i])
            total = da + db + carry
            digit_sum = total % 10
            new_carry = total // 10
            
            step_str = f"{da}+{db}+{carry}={total}={digit_sum} C={new_carry} "
            scratchpad.append(step_str)
            ans_digits.append(str(digit_sum))
            carry = new_carry
            
        if carry > 0:
            ans_digits.append(str(carry))
            
        final_ans = "".join(ans_digits[::-1])
        full_reasoning = "".join(scratchpad)
        
        # Add Stop Token
        seq_str = f"{a_str}+{b_str}={full_reasoning}{final_ans}."
        raw_seqs.append(seq_str)

    # --- DYNAMIC PADDING LOGIC ---
    # Find the longest sequence in this specific batch
    batch_max_len = max(len(s) for s in raw_seqs)
    
    # Pad everything to match ONLY the longest in this batch (not the global 512)
    # This saves massive compute on short/medium batches
    for seq_str in raw_seqs:
        padding = batch_max_len - len(seq_str)
        seq_str += ' ' * padding # Pad with space
        
        data = torch.tensor(encode(seq_str), dtype=torch.long)
        X.append(data[:-1])
        Y.append(data[1:])
        
    X = torch.stack(X).to(CONFIG['device'])
    Y = torch.stack(Y).to(CONFIG['device'])
    return X, Y


# --- 3. RECURSIVE TRANSFORMER MODEL ---

def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return [1.0 / (x ** (i + 1)) for i in range(num_heads)]

class Head(nn.Module):
    def __init__(self, head_size, head_index, num_heads):
        super().__init__()
        self.key = nn.Linear(CONFIG['n_embd'], head_size, bias=False)
        self.query = nn.Linear(CONFIG['n_embd'], head_size, bias=False)
        self.value = nn.Linear(CONFIG['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CONFIG['block_size'], CONFIG['block_size'])))
        
        slopes = get_alibi_slope(num_heads)
        self.slope = slopes[head_index]

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        
        indices = torch.arange(T, device=CONFIG['device'])
        distance = indices[None, :] - indices[:, None]
        alibi_bias = distance * self.slope
        
        wei += alibi_bias
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, i, num_heads) for i in range(num_heads)])
        self.proj = nn.Linear(CONFIG['n_embd'], CONFIG['n_embd'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class RecursiveGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, CONFIG['n_embd'])
        self.shared_block = Block(CONFIG['n_embd'], CONFIG['n_head'])
        self.ln_f = nn.LayerNorm(CONFIG['n_embd'])
        self.lm_head = nn.Linear(CONFIG['n_embd'], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb 
        
        for _ in range(CONFIG.get('recursion', 2)):
            x = self.shared_block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
        
    def generate(self, idx, max_new_tokens=CONFIG['block_size']):
        # STOP TOKEN OPTIMIZATION
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -CONFIG['block_size']:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # --- STOP CHECK ---
            if idx_next.item() == stoi['.']:
                break
            # ------------------

            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 4. ACCURACY CHECKER ---

def get_accuracy(model, num_samples=100):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(num_samples):
            d = random.randint(1, CONFIG['max_digits'])
            a = random.randint(0, 10**d - 1)
            b = random.randint(0, 10**d - 1)
            prompt = f"{a:0{d}d}+{b:0{d}d}="
            expected = a + b
            
            idx = torch.tensor([encode(prompt)], dtype=torch.long).to(CONFIG['device'])
            
            # Generate stops automatically now!
            generated_idx = model.generate(idx) 
            generated_str = decode(generated_idx[0].tolist())
            
            try:
                # We expect "1+2=... 3" (No dot in generated_str usually, loop breaks before append)
                # But if we did append it, we just filter for digits anyway.
                parts = generated_str.replace('=', ' ').split(' ')
                candidates = [p for p in parts if p.isdigit()]
                if candidates and int(candidates[-1]) == expected:
                    correct += 1
            except:
                pass
    model.train()
    return correct / num_samples

# --- 5. SETUP & TRAINING ---

model = RecursiveGPT().to(CONFIG['device'])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
param_count = sum(p.numel() for p in model.parameters())
print(f'Model has {param_count:,} parameters.')

plt.ion()
fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 8))
fig.suptitle(f"Training on 1-{CONFIG['max_digits']} Digit Addition")

loss_history = []
accuracy_history = []
iters_history = []
eval_iters = []

line_loss, = ax_loss.plot([], [], 'b-', label='Loss')
ax_loss.legend()
line_acc, = ax_acc.plot([], [], 'g-', label='Accuracy')
ax_acc.set_ylim(-0.05, 1.05)
ax_acc.legend()

print("Starting training with Curriculum & Dynamic Padding...")

try:
    for iter in range(CONFIG['max_iters']):
        
        # --- CURRICULUM LOGIC ---
        # Level 1: Steps 0-1000   -> Up to 3 digits (Fast learning)
        # Level 2: Steps 1000-3000 -> Up to 6 digits
        # Level 3: Steps 3000+     -> Up to 10 digits (Full power)
        if iter < 10000:
            difficulty = 1
        elif iter < 20000:
            difficulty = 2
        elif iter < 30000:
            difficulty = 3
        elif iter < 40000:
            difficulty = 4
        elif iter < 45000:
            difficulty = 5
        elif iter < 50000:
            difficulty = 6
        elif iter < 53000:
            difficulty = 7
        elif iter < 55000:
            difficulty = 8
        elif iter < 56500:
            difficulty = 9
        else:
            difficulty = CONFIG['max_digits'] # 10
            
        # Pass difficulty to generator
        xb, yb = generate_batch('train', CONFIG['batch_size'], difficulty)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        iters_history.append(iter)

        if iter % CONFIG['eval_interval'] == 0:
            # We only verify accuracy on the current difficulty level to keep logs relevant
            # (Testing 10-digit accuracy when we are only training on 3 is pointless)
            # You can tweak this if you want to see OOD earlier.
            
            # NOTE: Evaluation function also needs to support dynamic difficulty if strict,
            # but for now, the get_accuracy function generates its own internal randoms.
            # It's fine to leave get_accuracy as is, or pass 'difficulty' to it if you modified it.
            
            current_acc = get_accuracy(model, num_samples=50) 
            accuracy_history.append(current_acc)
            eval_iters.append(iter)
            
            line_loss.set_data(iters_history, loss_history)
            line_acc.set_data(eval_iters, accuracy_history)
            
            ax_loss.set_xlim(0, max(100, iter))
            ax_loss.set_ylim(0, max(loss_history) * 1.1)
            ax_acc.set_xlim(0, max(100, iter))
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            print(f"Step {iter} (Lvl {difficulty}): Loss {loss_val:.4f} | Acc {current_acc*100:.0f}%")

    print(f"Final Training Loss: {loss.item():.4f}")

except KeyboardInterrupt:
    print("\nTraining interrupted.")

plt.ioff()
plt.show()

# --- 6. SAVE MODEL ---

print("\nSaving model to disk...")
save_path = "recursive_adder.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# --- 7. FINAL METRICS ---
print("\nFinal Intelligence Density Check...")
final_acc = get_accuracy(model, num_samples=100)
param_count = sum(p.numel() for p in model.parameters())
density = (final_acc * 100) / (param_count / 1000)

print(f"Parameters: {param_count}")
print(f"Final Accuracy (1-{CONFIG['max_digits']} digits): {final_acc*100:.1f}%")
print(f"INTELLIGENCE DENSITY: {density:.4f}")