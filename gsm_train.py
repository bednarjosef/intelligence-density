import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import math
import time

from rotary_transformer import RecursiveGPT

# -----------------------------------------------------------------------------
# 1. Configuration for ~1M Parameters
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
}


# -----------------------------------------------------------------------------
# 2. Data Pipeline (Streaming & Filtering)
# -----------------------------------------------------------------------------
def encode(s):
    return [stoi.get(c, 0) for c in s] # Return 0 (first char) if unknown char found

class GSMStreamDataset(IterableDataset):
    def __init__(self, block_size):
        self.block_size = block_size
        # Load dataset in streaming mode (no massive download)
        print("Loading OpenMathInstruct-2 (Streaming)...")
        self.dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M", streaming=True)
        
    def __iter__(self):
        # Buffer to hold text until we have enough for a block
        buffer = []
        
        for sample in self.dataset:
            # FILTER: Only use augmented_gsm8k
            if sample['problem_source'] != 'augmented_gsm8k':
                continue
                
            # Formatting: Problem + Solution
            text = f"Problem: {sample['problem']}\nSolution: {sample['generated_solution']}\n\n"
            tokenized = encode(text)
            buffer.extend(tokenized)
            
            # Yield chunks of size (block_size + 1) for X, Y pair
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size + 1:]
                yield torch.tensor(chunk, dtype=torch.long)

# -----------------------------------------------------------------------------
# 3. Training Loop
# -----------------------------------------------------------------------------
def train():
    print(f"Initializing RecursiveGPT with config: {CONFIG}")
    model = RecursiveGPT(CONFIG)
    model.to(CONFIG['device'])
    
    # Print Param Count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {num_params:,} (Target: ~1M)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    # Data Loader
    train_dataset = GSMStreamDataset(CONFIG['block_size'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'])
    
    # Loop
    model.train()
    iter_num = 0
    t0 = time.time()
    
    print("\nStarting Training...")
    # Since it's an IterableDataset, we just loop over the loader
    for batch in train_loader:
        if iter_num >= CONFIG['max_iters']: break
        
        batch = batch.to(CONFIG['device'])
        X, Y = batch[:, :-1], batch[:, 1:]
        
        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        iter_num += 1
        
        if iter_num % 100 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"step {iter_num} | loss {loss.item():.4f} | time {dt*1000:.2f}ms")
            
        if iter_num % CONFIG['eval_interval'] == 0:
            print("\n--- GENERATION SAMPLE ---")
            context = torch.tensor([encode("Problem: If John has 5 apples")], dtype=torch.long, device=CONFIG['device'])
            model.generate(context, max_new_tokens=100)
            print("\n-------------------------\n")

    print("Training Complete.")
    torch.save(model.state_dict(), "recursive_gsm_1m.pth")

if __name__ == '__main__':
    train()