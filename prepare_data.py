import torch
from datasets import load_dataset
from tqdm import tqdm
import os

# -----------------------------------------------------------------------------
# 1. Config
# -----------------------------------------------------------------------------
chars = "\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
stoi = { ch:i for i,ch in enumerate(chars) }
encode = lambda s: [stoi.get(c, stoi[' ']) for c in s]

VAL_SIZE = 2000 
SPLIT_NAME = "train_5M" # <--- CHANGED to 5M

# -----------------------------------------------------------------------------
# 2. Load & Process
# -----------------------------------------------------------------------------
def prepare():
    print(f"Downloading {SPLIT_NAME} (this is ~5x larger, please wait)...")
    dataset = load_dataset("nvidia/OpenMathInstruct-2", split=SPLIT_NAME)
    
    input_ids = []
    labels = []
    
    gsm_count = 0
    
    print("Filtering and Tokenizing...")
    for sample in tqdm(dataset):
        # FILTER: Only augmented_gsm8k
        if sample['problem_source'] != 'augmented_gsm8k':
            continue
            
        gsm_count += 1
            
        # 1. Text Formatting
        problem_part = f"Problem: {sample['problem']}\n"
        solution_part = f"Solution: {sample['generated_solution']}\n\n"
        full_text = problem_part + solution_part
        
        # 2. Tokenize
        tokenized = encode(full_text)
        
        # 3. Masking
        target_tokens = list(tokenized)
        len_problem = len(encode(problem_part))
        for i in range(len_problem):
            target_tokens[i] = -100
            
        input_ids.extend(tokenized)
        labels.extend(target_tokens)

    print(f"\nStats:")
    print(f"Total Raw Examples Scanned: {len(dataset):,}")
    print(f"GSM8K Examples Found:       {gsm_count:,}")
    print(f"Total Tokens:               {len(input_ids):,}")
    
    # -----------------------------------------------------------------------------
    # 3. Save
    # -----------------------------------------------------------------------------
    print("Converting to tensors...")
    # Use int32 to save RAM (unless your vocabulary > 2 billion, which it isn't)
    all_inputs = torch.tensor(input_ids, dtype=torch.int32)
    all_labels = torch.tensor(labels, dtype=torch.int32)
    
    # Create Val Split (Last N tokens)
    # We want roughly 98/2 split since we have so much data now
    split_idx = len(all_inputs) - (VAL_SIZE * 256) # Approx tokens
    
    train_inputs = all_inputs[:split_idx]
    train_labels = all_labels[:split_idx]
    val_inputs = all_inputs[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"Train Tokens: {len(train_inputs):,}")
    print(f"Val Tokens:   {len(val_inputs):,}")
    
    print("Saving to disk...")
    torch.save({'inputs': train_inputs, 'labels': train_labels}, 'train_data.pt')
    torch.save({'inputs': val_inputs, 'labels': val_labels}, 'val_data.pt')
    print("Done!")

if __name__ == '__main__':
    prepare()