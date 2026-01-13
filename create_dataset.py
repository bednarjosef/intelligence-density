import torch
import random
import os
from tqdm import tqdm

from transformer_addition import encode, EOT_TOKEN, PAD_TOKEN

max_digits = 10

DATA_CONFIG = {
    'total_samples': 100_000,
    'val_split': 0.01,
    'max_digits': max_digits,
    'block_size': 200,
    'save_dir': f'datasets/ds_max{max_digits}'
}

def generate_example(max_digits, block_size):
    d = random.randint(1, max_digits)
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
    seq_str = f"{a_str}+{b_str}={full_reasoning}{final_ans}{EOT_TOKEN}"
    # print(seq_str)
    # print(len(seq_str))
    
    padding = block_size - len(seq_str)
    if padding < 0:
        return None
    seq_str += PAD_TOKEN * padding
    
    return torch.tensor(encode(seq_str), dtype=torch.uint8) # uint8 saves space!


def create_dataset():
    print(f"Generating {DATA_CONFIG['total_samples']} samples...")
    data = []
    
    # Loop with progress bar
    for _ in tqdm(range(DATA_CONFIG['total_samples'])):
        ex = generate_example(DATA_CONFIG['max_digits'], DATA_CONFIG['block_size'])
        if ex is not None:
            data.append(ex)
            
    # Stack into one massive tensor
    all_data = torch.stack(data)
    print(f"Data shape: {all_data.shape}, Memory: {all_data.element_size() * all_data.numel() / 1024**2:.2f} MB")

    # Shuffle
    idx = torch.randperm(all_data.shape[0])
    all_data = all_data[idx]

    # Split
    n_val = int(len(all_data) * DATA_CONFIG['val_split'])
    val_data = all_data[:n_val]
    train_data = all_data[n_val:]

    # Save
    os.makedirs(DATA_CONFIG['save_dir'], exist_ok=True)
    
    train_path = os.path.join(DATA_CONFIG['save_dir'], 'train.pt')
    val_path = os.path.join(DATA_CONFIG['save_dir'], 'val.pt')
    
    # Save as dictionary to keep metadata if needed
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    
    print(f"Saved train data to {train_path} ({len(train_data)} samples)")
    print(f"Saved val data to {val_path} ({len(val_data)} samples)")

if __name__ == "__main__":
    create_dataset()
