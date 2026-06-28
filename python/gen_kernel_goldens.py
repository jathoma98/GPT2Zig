"""
Generates kernel_golden.zig: small fixed-seed random inputs and their numpy reference
outputs for the four M2 math kernels (matmul, layernorm, softmax, gelu_tanh). All f32
values are emitted as u32 IEEE-754 bit patterns so Zig reconstructs the exact reference
via @bitCast, then compares its computed value with a 1e-5 abs tolerance (transcendentals
differ slightly between numpy and Zig's libm).

argv when called by build.zig:
  sys.argv[1]  output .zig   (from addOutputFileArg)
argv when called manually (no args): dumps a JSON summary to stdout.
"""

import sys
import json
import numpy as np
import zigout

np.random.seed(0)


def f32(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


# === Kernel inputs + numpy reference outputs (all float32) ===

# matmul: y = x @ W + b.  x[3,5] @ W[5,7] + b[7] -> y[3,7]
mm_x = f32(np.random.randn(3, 5))
mm_w = f32(np.random.randn(5, 7))
mm_b = f32(np.random.randn(7))
mm_y = f32(mm_x @ mm_w + mm_b)

# layernorm: per-row over the last axis, BIASED variance (np.var defaults to /D).
ln_x = f32(np.random.randn(4, 768))
ln_gamma = f32(np.random.randn(768))
ln_beta = f32(np.random.randn(768))
ln_eps = np.float32(1e-5)
ln_mu = ln_x.mean(axis=-1, keepdims=True)
ln_var = ln_x.var(axis=-1, keepdims=True)  # biased: divides by D
ln_out = f32((ln_x - ln_mu) / np.sqrt(ln_var + ln_eps) * ln_gamma + ln_beta)

# softmax: row-wise over the last axis, max-subtracted.
sm_x = f32(np.random.randn(4, 16))
sm_e = np.exp(sm_x - sm_x.max(axis=-1, keepdims=True))
sm_out = f32(sm_e / sm_e.sum(axis=-1, keepdims=True))

# gelu_new (tanh approx) — explicit formula, NOT scipy.special.erf.
gl_x = f32(np.random.randn(256))
_c = np.float32(np.sqrt(2.0 / np.pi))
gl_out = f32(0.5 * gl_x * (1.0 + np.tanh(_c * (gl_x + np.float32(0.044715) * gl_x**3))))


def emit_kernel(lines, name, fields, scalars=None):
    """fields: list of (zig_name, flat_f32_iterable). scalars: list of (zig_name, zig_expr)."""
    lines.append(f"pub const {name} = struct {{")
    for zname, expr in (scalars or []):
        lines.append(f"    pub const {zname} = {expr};")
    for zname, arr in fields:
        flat = np.asarray(arr, dtype=np.float32).ravel()
        body = ", ".join(zigout.f32_bits(v) for v in flat.tolist())
        lines.append(f"    pub const {zname}: [{flat.size}]u32 = .{{ {body} }};")
    lines.append("};")


output_path = sys.argv[1] if len(sys.argv) > 1 else None

if output_path is not None:
    lines = [zigout.header("gen_kernel_goldens.py"), ""]

    emit_kernel(lines, "matmul",
                [("x", mm_x), ("w", mm_w), ("b", mm_b), ("y", mm_y)],
                [("m", "3"), ("k", "5"), ("n", "7")])
    lines.append("")
    emit_kernel(lines, "layernorm",
                [("x", ln_x), ("gamma", ln_gamma), ("beta", ln_beta), ("out", ln_out)],
                [("rows", "4"), ("cols", "768"),
                 ("eps_bits", f"@as(u32, {zigout.f32_bits(float(ln_eps))})")])
    lines.append("")
    emit_kernel(lines, "softmax",
                [("x", sm_x), ("out", sm_out)],
                [("rows", "4"), ("cols", "16")])
    lines.append("")
    emit_kernel(lines, "gelu",
                [("x", gl_x), ("out", gl_out)],
                [("len", "256")])

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
else:
    print(json.dumps({
        "matmul": {"x": list(mm_x.shape), "w": list(mm_w.shape), "y": list(mm_y.shape)},
        "layernorm": {"x": list(ln_x.shape), "out_first": float(ln_out.flat[0])},
        "softmax": {"x": list(sm_x.shape), "row0_sum": float(sm_out[0].sum())},
        "gelu": {"len": int(gl_x.size), "out_first": float(gl_out[0])},
    }, indent=2))
