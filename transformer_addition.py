import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from addition_transformer import RecursiveGPT

EOT_TOKEN = '.'
PAD_TOKEN = '|'

chars = "0123456789+= C" + EOT_TOKEN + PAD_TOKEN 
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

EOT_TOKEN_ID = stoi[EOT_TOKEN]
PAD_TOKEN_ID = stoi[PAD_TOKEN]

CONFIG = {
    'batch_size': 64,
    'block_size': 200,
    'n_embd': 32,
    'n_head': 4,
    'recursion': 4,
    'lr': 5e-3,         
    'max_iters': 30000,
    'eval_interval': 500,
    'max_digits': 3,
    'vocab_size': vocab_size,
    'eval_samples': 50,
    'pad_token_id': PAD_TOKEN_ID,
    'eot_token_id': EOT_TOKEN_ID,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}



# def generate_batch(split, batch_size=CONFIG['batch_size']):
#     X, Y = [], []
#     for _ in range(batch_size):
#         # VARIABLE DIFFICULTY:
#         # Pick a random length between 1 and max_digits for THIS example
#         d = random.randint(1, CONFIG['max_digits'])
        
#         a = random.randint(0, 10**d - 1)
#         b = random.randint(0, 10**d - 1)
        
#         a_str = f"{a:0{d}d}"
#         b_str = f"{b:0{d}d}"
        
#         scratchpad = []
#         carry = 0
#         ans_digits = []
        
#         # Loop backwards
#         for i in range(d-1, -1, -1):
#             da = int(a_str[i])
#             db = int(b_str[i])
#             total = da + db + carry
#             digit_sum = total % 10
#             new_carry = total // 10
            
#             # Format: "9+9+1=19=9 C=1 "
#             step_str = f"{da}+{db}+{carry}={total}={digit_sum} C={new_carry} "
#             scratchpad.append(step_str)
#             ans_digits.append(str(digit_sum))
#             carry = new_carry
            
#         if carry > 0:
#             ans_digits.append(str(carry))
            
#         final_ans = "".join(ans_digits[::-1])
#         full_reasoning = "".join(scratchpad)
        
#         seq_str = f"{a_str}+{b_str}={full_reasoning}{final_ans}"
        
#         # Padding
#         padding = CONFIG['block_size'] - len(seq_str)
#         if padding < 0: 
#             # If thinking is too long, crop it (rare with 400 block_size)
#             seq_str = seq_str[:CONFIG['block_size']]
#             padding = 0
#         else:
#             seq_str += ' ' * padding
            
#         data = torch.tensor(encode(seq_str), dtype=torch.long)
#         X.append(data[:-1])
#         Y.append(data[1:])
        
#     X = torch.stack(X).to(CONFIG['device'])
#     Y = torch.stack(Y).to(CONFIG['device'])
#     return X, Y

# def get_accuracy(model, num_samples=100):
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for _ in range(num_samples):
#             # Test on random lengths too!
#             d = random.randint(1, CONFIG['max_digits'])
#             a = random.randint(0, 10**d - 1)
#             b = random.randint(0, 10**d - 1)
#             prompt = f"{a:0{d}d}+{b:0{d}d}="
#             expected = a + b
            
#             idx = torch.tensor([encode(prompt)], dtype=torch.long).to(CONFIG['device'])
#             # Generate enough tokens for 10-digit reasoning
#             generated_idx = model.generate(idx)
#             generated_str = decode(generated_idx[0].tolist())
            
#             try:
#                 parts = generated_str.replace('=', ' ').split(' ')
#                 candidates = [p for p in parts if p.isdigit()]
#                 if candidates and int(candidates[-1]) == expected:
#                     correct += 1
#             except:
#                 pass
#     model.train()
#     return correct / num_samples


def estimate_accuracy(model, data, samples=50):
    model.eval()
    correct = 0
    total = 0
    
    # 1. Pick random samples
    ix = torch.randint(0, data.shape[0], (samples,), device=CONFIG['device'])
    
    # 2. Prepare Batches by Prompt Length
    # We cannot stack "1+1=" (4 chars) and "100+100=" (8 chars) directly.
    # So we group them!
    prompts_by_len = {}
    
    for i in ix:
        full_seq = data[i].tolist()
        full_str = decode(full_seq)
        if '=' not in full_str: continue
        
        # Parse "123+456="
        prompt_str = full_str.split('=')[0] + '='
        
        # FIX: Split manually to handle leading zeros (e.g., "024")
        lhs = prompt_str[:-1] # "829+024"
        num1, num2 = lhs.split('+')
        expected_val = int(num1) + int(num2)
        
        # Group by length of prompt
        plen = len(prompt_str)
        if plen not in prompts_by_len:
            prompts_by_len[plen] = []
        prompts_by_len[plen].append((prompt_str, expected_val))

    # 3. Process each group as a single BATCH
    with torch.no_grad():
        for plen, items in prompts_by_len.items():
            # Create batch tensor
            batch_prompts = [item[0] for item in items]
            expected_vals = [item[1] for item in items]
            
            # Encode all at once
            batch_indices = [encode(p) for p in batch_prompts]
            prompt_tensor = torch.tensor(batch_indices, dtype=torch.long, device=CONFIG['device'])
            
            # GENERATE IN PARALLEL (The Speedup!)
            # This runs 1 big kernel instead of N small ones
            gen_tensor = model.generate(prompt_tensor)
            
            # 4. Check answers
            for j, seq_idx in enumerate(gen_tensor):
                gen_str = decode(seq_idx.tolist())
                
                try:
                    # Parse output
                    clean_gen = gen_str.strip()
                    import re
                    # Find all numbers in the generated string
                    numbers = re.findall(r'\d+', clean_gen)
                    if numbers:
                        # The last number should be the answer
                        model_ans = int(numbers[-1])
                        if model_ans == expected_vals[j]:
                            correct += 1
                except:
                    pass
                total += 1

    model.train()
    return correct / total if total > 0 else 0


def load_data(split, max_digits, device):
    save_dir = f'datasets/ds_max{max_digits}'
    filename = 'train.pt' if split == 'train' else 'val.pt'
    path = os.path.join(save_dir, filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}. Run create_dataset.py first!")
        
    print(f"Loading {split} data from {path}...")
    data = torch.load(path)

    data = data.to(dtype=torch.long).to(device)
    print(f"--> Loaded {len(data)} examples to {device}")
    return data

def get_batch(data, batch_size):
    ix = torch.randint(0, data.shape[0], (batch_size,), device=data.device)
    batch = data[ix]
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y


def main():
    print(f"Running on {CONFIG['device']}...")
    model = RecursiveGPT(CONFIG).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.33)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Model has {param_count:,} parameters.')

    # Setup Plotting
    plt.ion()
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Training Recursive Transformer (n_embd={CONFIG['n_embd']})")

    # Data containers for plotting
    iters, losses = [], []
    eval_iters, train_accs, val_accs = [], [], []

    # Lines
    line_loss, = ax_loss.plot([], [], 'b-', alpha=0.5, label='Batch Loss')
    line_train_acc, = ax_acc.plot([], [], 'g-', label='Train Accuracy')
    line_val_acc, = ax_acc.plot([], [], 'r-', label='Val Accuracy')

    ax_loss.set_ylabel("Cross Entropy Loss")
    ax_loss.legend()
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.legend()

    print(f'Loading train/val data...')
    train_data = load_data('train', CONFIG['max_digits'], CONFIG['device'])
    val_data = load_data('val', CONFIG['max_digits'], CONFIG['device'])

    print("Starting training...")
    start_time = time.time()

    try:
        for iter in range(CONFIG['max_iters']):
            xb, yb = get_batch(train_data, CONFIG['batch_size'])
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            if iter % 10 == 0:
                iters.append(iter)
                losses.append(loss.item())
                
                if iter % 50 == 0:
                    line_loss.set_data(iters, losses)
                    ax_loss.set_xlim(0, max(100, iter))
                    ax_loss.set_ylim(0, max(losses) * 1.1)

            if iter % CONFIG['eval_interval'] == 0:
                print(f"Step {iter}: Evaluating...", end='\r')

                t_acc = estimate_accuracy(model, train_data, samples=CONFIG['eval_samples'])
                v_acc = estimate_accuracy(model, val_data, samples=CONFIG['eval_samples'])

                eval_iters.append(iter)
                train_accs.append(t_acc)
                val_accs.append(v_acc)
                
                # Update Accuracy Graph
                line_train_acc.set_data(eval_iters, train_accs)
                line_val_acc.set_data(eval_iters, val_accs)
                ax_acc.set_xlim(0, max(100, iter))
                
                # Refresh Plot
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                dt = time.time() - start_time
                print(f"Step {iter}: Loss {loss.item():.4f} | Train Acc {t_acc:.2f} | Val Acc {v_acc:.2f} | Time {dt:.1f}s")

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
    final_acc = estimate_accuracy(model, val_data, samples=100)
    param_count = sum(p.numel() for p in model.parameters())
    density = (final_acc * 100) / (param_count / 1000)

    print(f"Parameters: {param_count}")
    print(f"Final Accuracy (1-{CONFIG['max_digits']} digits): {final_acc*100:.1f}%")
    print(f"INTELLIGENCE DENSITY: {density:.4f}")


if __name__ == '__main__':
    main()
