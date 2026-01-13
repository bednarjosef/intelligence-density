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
    'block_size': 256,
    'save_dir': f'datasets/ds_max{max_digits}'
}

# def generate_example(max_digits, block_size):
#     d = random.randint(1, max_digits)
#     a = random.randint(0, 10**d - 1)
#     b = random.randint(0, 10**d - 1)
    
#     a_str = f"{a:0{d}d}"
#     b_str = f"{b:0{d}d}"
    
#     scratchpad = []
#     carry = 0
#     ans_digits = []
    
#     for i in range(d-1, -1, -1):
#         da = int(a_str[i])
#         db = int(b_str[i])
#         total = da + db + carry
#         digit_sum = total % 10
#         new_carry = total // 10
#         step_str = f"{da}+{db}+{carry}={total}={digit_sum} C={new_carry} "
#         scratchpad.append(step_str)
#         ans_digits.append(str(digit_sum))
#         carry = new_carry
        
#     if carry > 0:
#         ans_digits.append(str(carry))
        
#     final_ans = "".join(ans_digits[::-1])
#     full_reasoning = "".join(scratchpad)
#     seq_str = f"{a_str}+{b_str}={full_reasoning}{final_ans}{EOT_TOKEN}"
#     # print(seq_str)
#     # print(len(seq_str))
    
#     padding = block_size - len(seq_str)
#     if padding < 0:
#         return None
#     seq_str += PAD_TOKEN * padding
    
#     return torch.tensor(encode(seq_str), dtype=torch.uint8) # uint8 saves space!


def generate_example(max_digits, block_size):
    # --- 1. SMART RANDOMIZATION (Prevent "Magnitude Mismatch" errors) ---
    rand_val = random.random()
    
    if rand_val < 0.15: 
        # HARD CASE 1: Chain Reaction (9999... + 1)
        # Forces the model to learn long carry ripples
        d = random.randint(1, max_digits)
        a = int('9' * d) 
        b = 1 if random.random() < 0.5 else int('9' * random.randint(1, d))
            
    elif rand_val < 0.30:
        # HARD CASE 2: Sparse Numbers (1000... + 5)
        # Forces attention to track position carefully
        d = random.randint(3, max_digits)
        a = 10**(d-1) 
        b = random.randint(0, 10**(d//2)) 
        
    elif rand_val < 0.50:
        # HARD CASE 3: Magnitude Mismatch (Huge + Tiny)
        d_a = random.randint(1, max_digits)
        d_b = random.randint(1, max_digits)
        a = random.randint(0, 10**d_a - 1)
        b = random.randint(0, 10**d_b - 1)
        
    else:
        # STANDARD CASE (Uniform Random)
        d = random.randint(1, max_digits)
        a = random.randint(0, 10**d - 1)
        b = random.randint(0, 10**d - 1)

    # Convert to strings
    a_str = str(a)
    b_str = str(b)
    
    # --- 2. PREPARE REVERSED INPUTS ---
    # We pad to the same length so the indices align perfectly (0: , 1: , etc.)
    max_len = max(len(a_str), len(b_str))
    
    # Standard pad "00123"
    a_fwd = a_str.zfill(max_len)
    b_fwd = b_str.zfill(max_len)
    
    # REVERSE for the model ("32100")
    # This ensures the Ones place is ALWAYS at index 0
    a_rev = a_fwd[::-1]
    b_rev = b_fwd[::-1]
    
    # --- 3. GENERATE REASONING TRACE ---
    scratchpad = []
    carry = 0
    ans_digits = []
    
    # We iterate 0..max_len (Count Up)
    for i in range(max_len):
        da = int(a_rev[i])
        db = int(b_rev[i])
        
        total = da + db + carry
        digit_sum = total % 10
        new_carry = total // 10
        
        # Format: "0: 4+7=11=1 C=1 "
        # We explicitly state the index "i:" to anchor the attention
        step_str = f"{i}: {da}+{db}+{carry}={total}={digit_sum} C={new_carry} "
        
        scratchpad.append(step_str)
        ans_digits.append(str(digit_sum))
        carry = new_carry
        
    # Handle Final Carry
    if carry > 0:
        # We add one final step for the overflow
        # "5: 1 C=0 " or just append the digit
        # Let's keep the pattern consistent:
        # Implicitly adding 0+0+carry
        step_str = f"{max_len}: {carry} C=0 "
        scratchpad.append(step_str)
        ans_digits.append(str(carry))
        
    # Final Answer is LSB -> MSB (Reversed) because we appended digits in order
    # Example: 11771 reversed is 17711
    final_ans_rev = "".join(ans_digits)
    
    full_reasoning = "".join(scratchpad)
    
    # --- 4. CONSTRUCT FINAL STRING ---
    # Format: "Input R RevInput Steps R RevAns."
    # Example: "2837+8934 R 4398+7382 0: ... R 17711."
    seq_str = f"{a_str}+{b_str} R {a_rev}+{b_rev} {full_reasoning}R {final_ans_rev}{EOT_TOKEN}"
    # print(seq_str)
    
    # --- 5. PADDING ---
    padding = block_size - len(seq_str)
    if padding < 0:
        return None
    seq_str += PAD_TOKEN * padding
    
    return torch.tensor(encode(seq_str), dtype=torch.uint8)


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
