from transformers import GPT2LMHeadModel
import torch, numpy as np
import os

prompt_ids = [15496, 11, 314, 1101]  # "Hello, I am"
m = GPT2LMHeadModel.from_pretrained("gpt2").eval()
ids = torch.tensor([prompt_ids])
with torch.no_grad():
    logits = m(ids).logits  # [1, S, 50257]

out_path = os.path.join(os.path.dirname(__file__), "ref_logits.npy")
np.save(out_path, logits.numpy())
print(f"Saved logits shape {tuple(logits.shape)} to {out_path}")
