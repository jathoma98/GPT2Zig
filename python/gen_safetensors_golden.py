"""
Generates safetensors_golden.zig: the manifest of all tensor metadata, bit-exact
f32 spot-check values, per-element samples for the Tensor row/col-major indexing
tests, and the parsed config.json hyperparameters.

argv when called by build.zig:
  sys.argv[1]  model path     (from addFileArg — tracked by build cache)
  sys.argv[2]  config path    (from addFileArg — tracked by build cache)
  sys.argv[3]  output .zig    (from addOutputFileArg)
argv when called manually (no args): dumps JSON summary to stdout.
"""

import sys
import json
import struct
import numpy as np
from safetensors import safe_open
import zigout

SPOT_CHECK = ["wte.weight", "h.0.attn.c_attn.bias"]
N_SPOT = 5

# (name, [(r, c), ...]) — row-major logical indices to sample for the at() test.
# Corners + interior, chosen to catch off-by-one / axis-swap bugs.
ELEMENT_SAMPLES = [
    ("wte.weight", [(0, 0), (0, 767), (50256, 0), (50256, 767), (1, 1), (12345, 500)]),
    # 1D bias is modeled as a [1, N] row vector → r is always 0.
    ("h.0.attn.c_attn.bias", [(0, 0), (0, 1), (0, 767), (0, 2303), (0, 400)]),
]

# wte.weight viewed as wteᵀ [768, 50257]: at(k, n) must equal wte[n, k].
# Validates the col_major index path against real data == the M3 tied-embedding transpose.
WTE_T_SAMPLES = [(0, 0), (0, 50256), (767, 0), (767, 50256), (5, 12345), (100, 1)]

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/gpt2/model.safetensors"
config_path = sys.argv[2] if len(sys.argv) > 2 else "models/gpt2/config.json"
output_path = sys.argv[3] if len(sys.argv) > 3 else None


def elem(arr, r, c):
    """Row-major logical element. 1D arrays are treated as a [1, N] row vector."""
    return arr[c] if arr.ndim == 1 else arr[r, c]


with safe_open(model_path, framework="numpy") as f:
    keys = sorted(f.keys())
    manifest = []
    for k in keys:
        t = f.get_tensor(k)
        manifest.append({
            "name": k,
            "rank": len(t.shape),
            "shape": list(t.shape),
            "n_elements": int(np.prod(t.shape)),
        })
    spot = {k: f.get_tensor(k).flatten()[:N_SPOT].tolist() for k in SPOT_CHECK}

    element_samples = []
    for name, indices in ELEMENT_SAMPLES:
        arr = f.get_tensor(name)
        for (r, c) in indices:
            element_samples.append((name, r, c, float(elem(arr, r, c))))

    wte = f.get_tensor("wte.weight")
    wte_t_samples = [(k, n, float(wte[n, k])) for (k, n) in WTE_T_SAMPLES]

with open(config_path, "r", encoding="utf-8") as cf:
    cfg = json.load(cf)
config = {
    "n_layer": cfg["n_layer"],
    "n_head": cfg["n_head"],
    "n_embd": cfg["n_embd"],
    "n_ctx": cfg["n_ctx"],
    "vocab_size": cfg["vocab_size"],
    "ln_eps_bits": cfg["layer_norm_epsilon"],
    "eos_token_id": cfg["eos_token_id"],
}

if output_path is not None:
    lines = [
        zigout.header("gen_safetensors_golden.py"),
        "pub const TensorEntry = struct {",
        "    name: []const u8,",
        "    rank: u8,",
        "    shape: [4]u32,  // unused dims are 0",
        "    n_elements: u64,",
        "};",
        "pub const manifest: []const TensorEntry = &.{",
    ]
    for e in manifest:
        name_lit = f'"{e["name"]}"'  # tensor names are always safe ASCII
        shape_lit = zigout.padded_shape(e["shape"])
        lines.append(
            f'    .{{ .name = {name_lit}, .rank = {e["rank"]}, '
            f'.shape = {shape_lit}, .n_elements = {e["n_elements"]} }},'
        )
    lines.append("};")
    lines.append("")
    lines.append(f"// First {N_SPOT} floats of each spot-check tensor, as u32 bit patterns.")
    lines.append(f"// Compare via @as(f32, @bitCast(bits)) for exact equality.")
    for k in SPOT_CHECK:
        field = k.replace(".", "_").replace(" ", "_")
        arr_lit = zigout.f32_bits_array(spot[k])
        lines.append(f"pub const {field}_first{N_SPOT}: [{N_SPOT}]u32 = {arr_lit};")

    # Per-element samples: f32 stored as its u32 bit pattern for exact comparison.
    lines.append("")
    lines.append("pub const ElementSample = struct {")
    lines.append("    name: []const u8,")
    lines.append("    r: u32,")
    lines.append("    c: u32,")
    lines.append("    bits: u32,")
    lines.append("};")
    lines.append("// Row-major at(r, c) samples vs numpy t[r, c].")
    lines.append("pub const element_samples: []const ElementSample = &.{")
    for (name, r, c, v) in element_samples:
        lines.append(
            f'    .{{ .name = "{name}", .r = {r}, .c = {c}, .bits = {zigout.f32_bits(v)} }},'
        )
    lines.append("};")
    lines.append("// Col-major view of wte.weight as wteᵀ [768, 50257]: at(k, n) == wte[n, k].")
    lines.append("pub const wte_t_samples: []const ElementSample = &.{")
    for (r, c, v) in wte_t_samples:
        lines.append(
            f'    .{{ .name = "wte.weight", .r = {r}, .c = {c}, .bits = {zigout.f32_bits(v)} }},'
        )
    lines.append("};")

    # config.json hyperparameters. ln_eps as f32 bits to validate float parsing exactly.
    lines.append("")
    lines.append("pub const config = .{")
    lines.append(f'    .n_layer = @as(u32, {config["n_layer"]}),')
    lines.append(f'    .n_head = @as(u32, {config["n_head"]}),')
    lines.append(f'    .n_embd = @as(u32, {config["n_embd"]}),')
    lines.append(f'    .n_ctx = @as(u32, {config["n_ctx"]}),')
    lines.append(f'    .vocab_size = @as(u32, {config["vocab_size"]}),')
    lines.append(f'    .ln_eps_bits = @as(u32, {zigout.f32_bits(config["ln_eps_bits"])}),')
    lines.append(f'    .eos_token_id = @as(u32, {config["eos_token_id"]}),')
    lines.append("};")

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
else:
    print(json.dumps({"manifest_count": len(manifest), "spot": {
        k: [f"0x{struct.unpack('<I', struct.pack('<f', v))[0]:08X}" for v in vals]
        for k, vals in spot.items()
    }, "config": config, "element_samples": len(element_samples)}, indent=2))
