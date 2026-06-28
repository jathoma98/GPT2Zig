"""
Generates safetensors_golden.zig: the manifest of all tensor metadata plus
bit-exact f32 spot-check values for two tensors.

argv when called by build.zig:
  sys.argv[1]  model path     (from addFileArg — tracked by build cache)
  sys.argv[2]  output .zig    (from addOutputFileArg)
argv when called manually (no args): dumps JSON summary to stdout.
"""

import sys
import struct
import numpy as np
from safetensors import safe_open
import zigout

SPOT_CHECK = ["wte.weight", "h.0.attn.c_attn.bias"]
N_SPOT = 5

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/gpt2/model.safetensors"
output_path = sys.argv[2] if len(sys.argv) > 2 else None

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

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
else:
    import json
    print(json.dumps({"manifest_count": len(manifest), "spot": {
        k: [f"0x{struct.unpack('<I', struct.pack('<f', v))[0]:08X}" for v in vals]
        for k, vals in spot.items()
    }}, indent=2))
