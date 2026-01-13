import torch
import torch.nn as nn
from torch.nn import functional as F

# --- 1. CONFIGURATION (MUST MATCH TRAINING EXACTLY) ---
CONFIG = {
    'block_size': 160,
    'n_embd': 16,       
    'n_head': 4,       
    'n_layer': 4,       
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# --- 2. MODEL ARCHITECTURE (Copy-Paste from Training) ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(CONFIG['n_embd'], head_size, bias=False)
        self.query = nn.Linear(CONFIG['n_embd'], head_size, bias=False)
        self.value = nn.Linear(CONFIG['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CONFIG['block_size'], CONFIG['block_size'])))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
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
        # HARDCODED VOCAB SIZE (0-9, +, =, space, C) -> 14 chars
        self.token_embedding_table = nn.Embedding(14, CONFIG['n_embd'])
        self.position_embedding_table = nn.Embedding(CONFIG['block_size'], CONFIG['n_embd'])
        self.shared_block = Block(CONFIG['n_embd'], CONFIG['n_head'])
        self.ln_f = nn.LayerNorm(CONFIG['n_embd'])
        self.lm_head = nn.Linear(CONFIG['n_embd'], 14)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=CONFIG['device']))
        x = tok_emb + pos_emb
        for _ in range(CONFIG['n_layer']):
            x = self.shared_block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -CONFIG['block_size']:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 3. HELPER FUNCTIONS ---
chars = "0123456789+= C" 
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

def load_model(path="recursive_adder.pth"):
    print(f"Loading model from {path}...")
    model = RecursiveGPT().to(CONFIG['device'])
    try:
        model.load_state_dict(torch.load(path, map_location=CONFIG['device']))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: .pth file not found. Make sure you trained it first!")
        exit()
    model.eval()
    return model

def solve(model, expression):
    # Parse "123+456"
    try:
        a_str, b_str = expression.split('+')
        a, b = int(a_str), int(b_str)
    except ValueError:
        return "Invalid format. Use '123+456'"

    # AUTO-PADDING LOGIC
    # The model was trained on padding with zeros based on max length.
    # We infer the 'd' (digits) from the input length to give it the best hint.
    max_len = max(len(a_str), len(b_str))
    
    # Format: "009+012=" (padded to matching lengths)
    prompt = f"{a:0{max_len}d}+{b:0{max_len}d}="
    
    print(f"\nThinking process for '{prompt}':")
    
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(CONFIG['device'])
    
    # Generate
    with torch.no_grad():
        # Generate enough tokens for the scratchpad
        generated_idx = model.generate(idx, max_new_tokens=300)
        
    output = decode(generated_idx[0].tolist())
    
    # Pretty print the scratchpad
    # The output looks like: "12+34= 2+4+0=6... 46"
    # We split strictly by '=' to isolate the prompt from the rest
    reasoning = output.split('=')[1:] 
    full_text = output[len(prompt):] # Everything after "="
    
    print(full_text)
    
    # Extract Answer
    parts = output.replace('=', ' ').split(' ')
    candidates = [p for p in parts if p.isdigit()]
    
    if candidates:
        return candidates[-1]
    else:
        return "Error: No answer found"

# --- 4. MAIN LOOP ---
if __name__ == "__main__":
    model = load_model('recursive_adder_6k_b.pth')
    print("\n--- AI ADDER LOADED ---")
    print("Type an addition problem (e.g. 123+456) or 'q' to quit.")
    
    while True:
        user_input = input("\nProblem: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
            
        result = solve(model, user_input.strip())
        print(f"Final Answer: {result}")