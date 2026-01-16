import torch
from datasets import load_dataset
from tqdm import tqdm
import os

# -----------------------------------------------------------------------------
# 1. Config & Tokenizer
# -----------------------------------------------------------------------------
# Ensure '~' is the last character to act as EOS
chars = "\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
stoi = { ch:i for i,ch in enumerate(chars) }
encode = lambda s: [stoi.get(c, stoi[' ']) for c in s]

# Config
VAL_SIZE = 2000 
SPLIT_NAME = "train_5M"
MAX_LENGTH = 1024 # Filter out examples longer than this

def prepare():
    print(f"Downloading {SPLIT_NAME}...")
    dataset = load_dataset("nvidia/OpenMathInstruct-2", split=SPLIT_NAME)
    
    input_ids = []
    labels = []
    
    stats = {
        "processed": 0,
        "skipped_too_long": 0,
        "skipped_wrong_source": 0
    }
    
    print(f"Processing (Filtering > {MAX_LENGTH} tokens)...")
    for sample in tqdm(dataset):
        # 1. Source Filter
        if sample['problem_source'] != 'augmented_gsm8k': 
            stats["skipped_wrong_source"] += 1
            continue
            
        # 2. Format Text with EOT Token
        problem = f"Problem: {sample['problem']}\n"
        # APPEND THE EOS TOKEN '~' HERE
        solution = f"Solution: {sample['generated_solution']}~\n" 
        full_text = problem + solution
        
        # 3. Tokenize
        tokens = encode(full_text)
        
        # 4. Length Filter (Critical for CoT)
        if len(tokens) > MAX_LENGTH:
            stats["skipped_too_long"] += 1
            continue
            
        # 5. Masking (Problem = -100, Solution = tokens)
        targets = list(tokens)
        prob_len = len(encode(problem))
        
        # Mask the problem part (set to 100 for uint8 storage, mapped to -100 later)
        for i in range(prob_len):
            targets[i] = 100 
            
        input_ids.extend(tokens)
        labels.extend(targets)
        stats["processed"] += 1

    print("\n--- Statistics ---")
    print(f"Kept Examples:       {stats['processed']:,}")
    print(f"Skipped (Too Long):  {stats['skipped_too_long']:,}")
    print(f"Skipped (Wrong Src): {stats['skipped_wrong_source']:,}")
    
    # -----------------------------------------------------------------------------
    # 3. Compress & Save
    # -----------------------------------------------------------------------------
    print("\nCompressing to uint8...")
    all_inputs = torch.tensor(input_ids, dtype=torch.uint8)
    all_labels = torch.tensor(labels, dtype=torch.uint8)
    
    # Create Val Split (Last N tokens)
    split_idx = len(all_inputs) - (VAL_SIZE * MAX_LENGTH) 
    
    # Safety check
    if split_idx < 0:
        raise ValueError("Dataset is too small for the requested validation size.")

    print(f"Train Tokens: {split_idx:,}")
    print(f"Val Tokens:   {len(all_inputs) - split_idx:,}")

    torch.save({
        'inputs': all_inputs[:split_idx], 
        'labels': all_labels[:split_idx]
    }, 'train_data_uint8.pt')
    
    torch.save({
        'inputs': all_inputs[split_idx:], 
        'labels': all_labels[split_idx:]
    }, 'val_data_uint8.pt')
    
    print(f"Done! Saved compressed data.")

if __name__ == '__main__':
    prepare()