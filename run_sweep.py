import torch
import os
import time
from etz_hachaim.language.tokenizer import CharTokenizer

from models.transformer_lm import MiniTransformerLM
from models.etz_unified import EtzUnifiedLM
from models.bigram import BigramLM

print("="*60)
print("--- TEXT TOURNAMENT: Etz HaChaim vs a small Transformer (Tiny Shakespeare) ---")
print("="*60)

# 1. Shared data preparation (run dataset/download_data.py first)
with open('dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 2. Hyperparameters, identical for every model.
# NB: one run per model, no seeds, and the models do not have the same parameter count.
batch_size = 32
block_size = 64
max_iters = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = torch.zeros(10)
    for k in range(10):
        X, Y = get_batch('val')
        logits, loss, _ = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

# 3. The contenders
models_to_test = {
    "B2 (Transformer)": MiniTransformerLM(vocab_size, d_model=128, n_layer=4, n_head=4, block_size=block_size),
    "E0 (Etz, 1 world)": EtzUnifiedLM(vocab_size, d_model=64, num_worlds=1, tzimtzum_active=False),
    "E4 (Etz, full)": EtzUnifiedLM(vocab_size, d_model=64, num_worlds=4, tzimtzum_active=True), 
    "E5 (Etz + attention)": EtzUnifiedLM(vocab_size, d_model=64, num_worlds=4, tzimtzum_active=True, temporal_attention=True) 
}

results = {}

# 4. The tournament
for model_name, model in models_to_test.items():
    print(f"--- Running: {model_name} ---")
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start_time = time.time()
    
    for iter in range(max_iters):
        xb, yb = get_batch('train')
        logits, task_loss, intent_loss = model(xb, yb)
        
        # The full Etz uses the Keter (intent) loss; the other models return 0 for it
        total_loss = task_loss + 0.1 * intent_loss 
        
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        
    end_time = time.time()
    val_loss = estimate_loss(model)
    
    print(f"   [OK] Done in {end_time - start_time:.1f}s | validation loss: {val_loss:.4f}\n")
    
    results[model_name] = {
        "params": param_count,
        "val_loss": val_loss,
        "time_seconds": end_time - start_time
    }

# 5. Summary
print("="*60)
print("--- FINAL RESULTS ---")
print("="*60)
for name, data in results.items():
    print(f"{name.ljust(25)} | Params: {data['params']:<9,} | Validation Loss: {data['val_loss']:.4f}")
print("="*60)
