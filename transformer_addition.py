import re
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from rotary_transformer import RecursiveGPT

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
    'batch_size': 64,
    'block_size': 256,
    'n_embd': 448,
    'n_head': 7,
    'recursion': 4,
    'lr': 5e-3,         
    'max_iters': 30000,
    'eval_interval': 2500,
    'max_digits': 3,
    'vocab_size': vocab_size,
    'eval_samples': 20,
    'pad_token_id': PAD_TOKEN_ID,
    'eot_token_id': EOT_TOKEN_ID,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def estimate_accuracy(model, data, samples=50, verbose=False):
    model.eval()
    correct = 0
    total = 0
    
    ix = torch.randint(0, data.shape[0], (samples,), device=CONFIG['device'])
    
    for idx, i in enumerate(ix):
        full_seq = data[i].tolist()
        full_str = decode(full_seq)
        
        clean_str = full_str.replace(PAD_TOKEN, '').replace(EOT_TOKEN, '').strip()
        
        parts = clean_str.split(' ')

        question = parts[0]
        answer = parts[-1]
        
        question_indices = encode(question)
        questions_tensor = torch.tensor(question_indices, dtype=torch.long, device=CONFIG['device']).unsqueeze(0)
        
        max_new = CONFIG['block_size'] - len(question_indices)
        if max_new <= 0: continue
        
        gen_answer_indices = model.generate(questions_tensor, max_new_tokens=max_new, greedy=True)
        gen_answer = decode(gen_answer_indices[0].tolist())

        if verbose and idx < 5:
            print(f'Prompt: {question}')
            print(f'Generated: {gen_answer}')

        clean_gen_answer = gen_answer.replace(PAD_TOKEN, '').replace(EOT_TOKEN, '').strip()
        gen_result = clean_gen_answer[-1]

        if gen_result == answer:
            correct += 1
                
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
                v_acc = estimate_accuracy(model, val_data, samples=CONFIG['eval_samples'], verbose=True)

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
