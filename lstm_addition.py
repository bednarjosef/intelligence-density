import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURATION ---
CONFIG = {
    'batch_size': 64,
    'block_size': 24,   # Max sequence length
    'n_embd': 128,      # Vector size (Hidden dimension)
    'n_layer': 2,       # Number of LSTM layers
    'lr': 1e-3,         # Learning Rate
    'max_iters': 3000,  # Training steps
    'eval_interval': 200,
    'digits': 3,        # Learning 3-digit addition (e.g., 123+456)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Running on {CONFIG['device']}...")

# --- 2. DATA GENERATION & TOKENIZER ---
chars = "0123456789+= " 
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

def generate_batch(split, batch_size=CONFIG['batch_size']):
    X, Y = [], []
    for _ in range(batch_size):
        d = CONFIG['digits']
        a = random.randint(0, 10**d - 1)
        b = random.randint(0, 10**d - 1)
        
        # --- THE MAGIC REVERSAL ---
        # We reverse the digits of a, b, and the result
        # a=12, b=34 -> "21+43=64" (instead of "12+34=46")
        a_rev = str(a)[::-1]
        b_rev = str(b)[::-1]
        res_rev = str(a+b)[::-1]
        
        prob_str = f"{a_rev}+{b_rev}={res_rev}"
        # --------------------------
        
        padding = CONFIG['block_size'] - len(prob_str)
        if padding < 0: 
            prob_str = prob_str[:CONFIG['block_size']]
            padding = 0
            
        seq_str = prob_str + ' ' * padding
        data = torch.tensor(encode(seq_str), dtype=torch.long)
        X.append(data[:-1])
        Y.append(data[1:])
        
    X = torch.stack(X).to(CONFIG['device'])
    Y = torch.stack(Y).to(CONFIG['device'])
    return X, Y

# --- 3. THE MODEL (TinyLSTM) ---

class TinyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding: Converts "1" -> [0.1, -0.5, ...]
        self.token_embedding = nn.Embedding(vocab_size, CONFIG['n_embd'])
        
        # LSTM: The Recurrent Brain
        self.lstm = nn.LSTM(
            input_size=CONFIG['n_embd'], 
            hidden_size=CONFIG['n_embd'], 
            num_layers=CONFIG['n_layer'], 
            batch_first=True
        )
        
        # Head: Converts hidden state back to probabilities
        self.lm_head = nn.Linear(CONFIG['n_embd'], vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: (Batch, Time)
        x = self.token_embedding(idx) # (Batch, Time, n_embd)
        
        # LSTM output: out, (hidden, cell)
        out, _ = self.lstm(x) 
        
        logits = self.lm_head(out) # (Batch, Time, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # For generation, we maintain the hidden state to be efficient
        # (or for simplicity here, we can re-run the sequence, 
        # but LSTMs allow state passing which is cleaner)
        
        # 1. First pass: Process the prompt
        x = self.token_embedding(idx)
        out, (h, c) = self.lstm(x) # Get final state after prompt
        
        # Get last token prediction
        logits = self.lm_head(out[:, -1, :]) 
        
        # Loop for new tokens
        for _ in range(max_new_tokens):
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop if space
            if idx_next.item() == stoi[' ']:
                break
                
            # Forward ONLY the new token using the saved hidden states (h, c)
            x_next = self.token_embedding(idx_next)
            out_next, (h, c) = self.lstm(x_next, (h, c))
            logits = self.lm_head(out_next[:, -1, :])
            
        return idx

# --- 4. TRAINING SETUP ---

model = TinyLSTM().to(CONFIG['device'])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])

# Count Parameters
param_count = sum(p.numel() for p in model.parameters())

print(f"\n" + "="*40)
print(f" MODEL INITIALIZED: TinyLSTM")
print(f" TOTAL PARAMETERS:  {param_count:,}")
print(f" CONFIG: {CONFIG['n_layer']} Layers, {CONFIG['n_embd']} Hidden Dim")
print(f"="*40 + "\n")

loss_history = []
iters_history = []

# --- 5. TRAINING LOOP ---
print("Starting training...")
try:
    for iter in range(CONFIG['max_iters']):
        xb, yb = generate_batch('train')
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Record loss
        if iter % 10 == 0:
            loss_history.append(loss.item())
            iters_history.append(iter)

        if iter % CONFIG['eval_interval'] == 0:
            print(f"Step {iter:4d} | Loss: {loss.item():.4f}")

    print(f"Final Training Loss: {loss.item():.4f}")

except KeyboardInterrupt:
    print("\nTraining interrupted! Proceeding to evaluation...")

# --- 6. INTELLIGENCE DENSITY & VISUALIZATION ---

def evaluate_and_plot(model):
    print("\n" + "="*40)
    print(" EVALUATING INTELLIGENCE DENSITY (REVERSED MODE)")
    print("="*40)
    
    model.eval()
    
    true_values = []
    pred_values = []
    correct = 0
    total = 200 
    
    # MAGIC TEST: Try increasing digits to 5 or 8 to see if it generalizes!
    # If the model really learned the "algorithm", it should solve lengths it never saw.
    test_digits = CONFIG['digits'] 
    
    print(f"Testing on {total} new problems (Digits: {test_digits})...")
    
    for _ in range(total):
        a = random.randint(0, 10**test_digits - 1)
        b = random.randint(0, 10**test_digits - 1)
        
        # Reverse inputs for the model
        a_rev = str(a)[::-1]
        b_rev = str(b)[::-1]
        prompt = f"{a_rev}+{b_rev}="
        target_val = a + b
        
        idx = torch.tensor([encode(prompt)], dtype=torch.long).to(CONFIG['device'])
        
        # Generate reversed answer
        with torch.no_grad():
            generated_idx = model.generate(idx, max_new_tokens=len(str(target_val)) + 2)
        
        generated_str = decode(generated_idx[0].tolist())
        
        try:
            # Parse: "21+43=64" -> get "64" -> reverse back to "46" -> int(46)
            pred_rev = generated_str.split('=')[1].strip()
            # Clean up any non-digit chars that might hallucinate
            pred_rev = "".join(filter(str.isdigit, pred_rev))
            
            # UN-REVERSE to get normal number
            pred_normal = pred_rev[::-1]
            pred_val = int(pred_normal)
        except:
            pred_val = -1 

        if pred_val == target_val:
            correct += 1
            
        true_values.append(target_val)
        pred_values.append(pred_val)

    clean_true = [t for t, p in zip(true_values, pred_values) if p != -1]
    clean_pred = [p for t, p in zip(true_values, pred_values) if p != -1]
    
    accuracy = correct / total
    
    if clean_pred:
        errors = [abs(t - p) for t, p in zip(clean_true, clean_pred)]
        mae = sum(errors) / len(errors)
    else:
        mae = 9999
        
    param_count = sum(p.numel() for p in model.parameters())
    density = (accuracy * 100) / (param_count / 1000)

    print(f"\n--- FINAL RESULTS (MAGIC) ---")
    print(f"Parameters Used:       {param_count:,}")
    print(f"Accuracy:              {accuracy*100:.1f}%")
    print(f"Mean Absolute Error:   {mae:.1f}")
    print(f"-"*25)
    print(f"INTELLIGENCE DENSITY:  {density:.4f}")
    print(f"-"*25)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(iters_history, loss_history, label='Training Loss', color='purple')
    ax1.set_title("Training Loss (Reversed Input)")
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(clean_true, clean_pred, alpha=0.6, color='orange', s=15, label='LSTM Predictions')
    max_val = max(max(clean_true) if clean_true else 100, max(clean_pred) if clean_pred else 100)
    ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect Accuracy')
    ax2.set_title(f"Reality vs. Prediction (MAE: {mae:.1f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

evaluate_and_plot(model)