"""
M3 activation oracle: dump per-stage GPT-2 activations + final logits for the fixed prompt, so the
Zig forward pass can bisect against them. Output is self-describing raw-f32 binary (mmap'd by Zig,
not embedded as source): [u32 ndim][u32 dims...][f32 data], little-endian.

Usage: gen_activation_goldens.py <output_dir>   (writes act_*.bin into <output_dir>)
"""
import os
import sys
import struct

import numpy as np
import torch
from transformers import GPT2LMHeadModel

PROMPT_IDS = [15496, 11, 314, 1101]  # "Hello, I am" — matches gen_ref_logits.py


def dump(out_dir: str, name: str, arr) -> None:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", a.ndim))
        for d in a.shape:
            f.write(struct.pack("<I", d))
        f.write(a.tobytes())
    print(f"  {name}: shape {tuple(a.shape)} -> {path}")


def main() -> None:
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tr = model.transformer

    taps = {}

    # Modules whose forward returns a tuple (attn, block) expose the activation at [0]; others
    # (ln, mlp, drop) return the tensor directly. Squeeze the batch dim either way → [S, n_embd].
    def save_out(key):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            taps[key] = t.detach()[0].numpy()
        return hook

    def save_in(key, idx=0):
        def hook(_m, inp, _out):
            taps[key] = inp[idx].detach()[0].numpy()
        return hook

    handles = [
        tr.drop.register_forward_hook(save_out("embed")),         # wte + wpe (dropout is identity in eval)
        tr.h[0].ln_1.register_forward_hook(save_out("l0_ln1")),
        tr.h[0].attn.register_forward_hook(save_out("l0_attn")),  # post c_proj, pre-residual
        tr.h[0].ln_2.register_forward_hook(save_in("l0_resid1")), # x after attention residual
        tr.h[0].mlp.register_forward_hook(save_out("l0_mlp")),    # pre-residual
        tr.h[0].register_forward_hook(save_out("l0_out")),
        tr.h[5].register_forward_hook(save_out("l5_out")),
        tr.ln_f.register_forward_hook(save_out("lnf")),
    ]

    ids = torch.tensor([PROMPT_IDS])
    with torch.no_grad():
        logits = model(ids).logits  # [1, S, vocab]

    for h in handles:
        h.remove()

    # Every captured activation must be [S, n_embd]; logits [S, vocab].
    s = len(PROMPT_IDS)
    for key, t in taps.items():
        assert t.shape[0] == s, f"{key} has unexpected shape {t.shape}"

    print("Dumping M3 activation goldens:")
    dump(out_dir, "act_embed.bin", taps["embed"])
    dump(out_dir, "act_l0_ln1.bin", taps["l0_ln1"])
    dump(out_dir, "act_l0_attn.bin", taps["l0_attn"])
    dump(out_dir, "act_l0_resid1.bin", taps["l0_resid1"])
    dump(out_dir, "act_l0_mlp.bin", taps["l0_mlp"])
    dump(out_dir, "act_l0_out.bin", taps["l0_out"])
    dump(out_dir, "act_l5_out.bin", taps["l5_out"])
    dump(out_dir, "act_lnf.bin", taps["lnf"])
    dump(out_dir, "act_logits.bin", logits[0].numpy())


if __name__ == "__main__":
    main()
