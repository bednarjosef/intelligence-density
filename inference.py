import torch
import torch.nn as nn
from torch.nn import functional as F

from rotary_transformer import RecursiveGPT

# --- 1. CONFIGURATION (MUST MATCH TRAINING EXACTLY) ---
CONFIG = {
    'block_size': 256,
    'n_embd': 32,       
    'n_head': 4,       
    'recursion': 4,       
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# --- 3. HELPER FUNCTIONS ---
EOT_TOKEN = '.'
PAD_TOKEN = '|'

chars = "0123456789+=: RC" + EOT_TOKEN + PAD_TOKEN 
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

EOT_TOKEN_ID = stoi[EOT_TOKEN]
PAD_TOKEN_ID = stoi[PAD_TOKEN]

CONFIG = {
    'block_size': 256,
    'n_embd': 32,       
    'n_head': 4,       
    'recursion': 4,
    'vocab_size': vocab_size,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_model(path="recursive_adder.pth"):
    print(f"Loading model from {path}...")
    model = RecursiveGPT(CONFIG).to(CONFIG['device'])
    try:
        model.load_state_dict(torch.load(path, map_location=CONFIG['device']))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: .pth file not found. Make sure you trained it first!")
        exit()
    model.eval()
    return model

def solve(model, expression):
    a_str, b_str = expression.split('+')
    
    a_rev = a_str[::-1]
    b_rev = b_str[::-1]
    
    max_len = max(len(a_str), len(b_str))
    a_fwd = a_str.zfill(max_len)
    b_fwd = b_str.zfill(max_len)
    
    prompt = f"{a_fwd}+{b_fwd} R {a_fwd[::-1]}+{b_fwd[::-1]} "
    
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(CONFIG['device'])
    
    # 2. Generate
    with torch.no_grad():
        generated_idx = model.generate(idx, greedy=True)
        
    output = decode(generated_idx[0].tolist())
    print(output) # See the thinking!
    
    try:
        ans_part = output.split('R ')[-1]
        import re
        digits = re.findall(r'\d+', ans_part)[0]
        return digits[::-1]
    except:
        return "Error"

# --- 4. MAIN LOOP ---
if __name__ == "__main__":
    model = load_model('recursive_adder.pth')
    print("\n--- AI ADDER LOADED ---")
    print("Type an addition problem (e.g. 123+456) or 'q' to quit.")
    
    while True:
        user_input = input("\nProblem: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
            
        result = solve(model, user_input.strip())
        print(f"Final Answer: {result}")