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
    'block_size': 160,  # INCREASED: 10-digit math requires ~250+ tokens of thinking
    'n_embd': 16,       # Slightly larger to handle longer context dependencies
    'n_head': 4,
    'recursion': 4,
    'lr': 5e-3,         
    'max_iters': 30000,  # Enough to learn the general rule
    'eval_interval': 500, 
    'max_digits': 3,   # Train on anything from 1+1 to 10-digit+10-digit
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Running on {CONFIG['device']}...")


# --- 2. DATA GENERATION & TOKENIZER ---
chars = "0123456789+= C" 
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

def generate_batch(split, batch_size=CONFIG['batch_size']):
    X, Y = [], []
    for _ in range(batch_size):
        # VARIABLE DIFFICULTY:
        # Pick a random length between 1 and max_digits for THIS example
        d = random.randint(1, CONFIG['max_digits'])
        
        a = random.randint(0, 10**d - 1)
        b = random.randint(0, 10**d - 1)
        
        a_str = f"{a:0{d}d}"
        b_str = f"{b:0{d}d}"
        
        scratchpad = []
        carry = 0
        ans_digits = []
        
        # Loop backwards
        for i in range(d-1, -1, -1):
            da = int(a_str[i])
            db = int(b_str[i])
            total = da + db + carry
            digit_sum = total % 10
            new_carry = total // 10
            
            # Format: "9+9+1=19=9 C=1 "
            step_str = f"{da}+{db}+{carry}={total}={digit_sum} C={new_carry} "
            scratchpad.append(step_str)
            ans_digits.append(str(digit_sum))
            carry = new_carry
            
        if carry > 0:
            ans_digits.append(str(carry))
            
        final_ans = "".join(ans_digits[::-1])
        full_reasoning = "".join(scratchpad)
        
        seq_str = f"{a_str}+{b_str}={full_reasoning}{final_ans}"
        
        # Padding
        padding = CONFIG['block_size'] - len(seq_str)
        if padding < 0: 
            # If thinking is too long, crop it (rare with 400 block_size)
            seq_str = seq_str[:CONFIG['block_size']]
            padding = 0
        else:
            seq_str += ' ' * padding
            
        data = torch.tensor(encode(seq_str), dtype=torch.long)
        X.append(data[:-1])
        Y.append(data[1:])
        
    X = torch.stack(X).to(CONFIG['device'])
    Y = torch.stack(Y).to(CONFIG['device'])
    return X, Y

# --- 3. RECURSIVE TRANSFORMER MODEL ---

# --- ALIBI HELPER FUNCTION ---
def get_alibi_slope(num_heads):
    # Returns a geometric sequence of slopes for the heads
    # e.g. for 4 heads: [0.5, 0.25, 0.125, 0.0625]
    x = (2 ** 8) ** (1 / num_heads)
    return [1.0 / (x ** (i + 1)) for i in range(num_heads)]

class Head(nn.Module):
    def __init__(self, head_size, head_index, num_heads):
        super().__init__()
        self.key = nn.Linear(CONFIG['n_embd'], head_size, bias=False)
        self.query = nn.Linear(CONFIG['n_embd'], head_size, bias=False)
        self.value = nn.Linear(CONFIG['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CONFIG['block_size'], CONFIG['block_size'])))
        
        # ALiBi: Calculate the slope for this specific head
        slopes = get_alibi_slope(num_heads)
        self.slope = slopes[head_index]

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Standard Attention Score
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        
        # --- ALiBi LOGIC ---
        # Create a distance matrix: [[0, -1, -2], [0, 0, -1], ...]
        # We construct it dynamically to handle any T (generalization!)
        indices = torch.arange(T, device=CONFIG['device'])
        # Distance: j - i (how far back is the key from the query?)
        # We only care about causal history, so mask future
        distance = indices[None, :] - indices[:, None] # (T, T)
        
        # Apply the linear penalty
        # Closer tokens = small penalty (near 0)
        # Farther tokens = large negative penalty
        alibi_bias = distance * self.slope
        
        # Add bias to attention scores
        wei += alibi_bias
        # -------------------

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Pass the head index to each head so it knows its ALiBi slope
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

    def forward(self, x):
        return self.net(x)

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
        
        # DELETED: self.position_embedding_table 
        # ALiBi doesn't use learned positions!
        
        self.shared_block = Block(CONFIG['n_embd'], CONFIG['n_head'])
        self.ln_f = nn.LayerNorm(CONFIG['n_embd'])
        self.lm_head = nn.Linear(CONFIG['n_embd'], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        
        # No Position Embeddings! Just raw token embeddings.
        x = tok_emb 
        
        for _ in range(CONFIG.get('recursion', 2)): # Default to 2 if not set
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
        
    # ... (generate function stays same) ...
    def generate(self, idx, max_new_tokens=CONFIG['block_size']):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -CONFIG['block_size']:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 4. ACCURACY CHECKER ---

def get_accuracy(model, num_samples=100):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(num_samples):
            # Test on random lengths too!
            d = random.randint(1, CONFIG['max_digits'])
            a = random.randint(0, 10**d - 1)
            b = random.randint(0, 10**d - 1)
            prompt = f"{a:0{d}d}+{b:0{d}d}="
            expected = a + b
            
            idx = torch.tensor([encode(prompt)], dtype=torch.long).to(CONFIG['device'])
            # Generate enough tokens for 10-digit reasoning
            generated_idx = model.generate(idx)
            generated_str = decode(generated_idx[0].tolist())
            
            try:
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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.33)
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

print("Starting training...")

try:
    for iter in range(CONFIG['max_iters']):
        xb, yb = generate_batch('train')
        
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
            current_acc = get_accuracy(model, num_samples=100)
            accuracy_history.append(current_acc)
            eval_iters.append(iter)
            
            line_loss.set_data(iters_history, loss_history)
            line_acc.set_data(eval_iters, accuracy_history)
            
            ax_loss.set_xlim(0, max(100, iter))
            ax_loss.set_ylim(0, max(loss_history) * 1.1)
            ax_acc.set_xlim(0, max(100, iter))
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            print(f"Step {iter}: Loss {loss_val:.4f} | Accuracy {current_acc*100:.0f}%")

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