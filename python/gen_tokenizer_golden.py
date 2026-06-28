import sys
import tiktoken
import zigout

enc = tiktoken.get_encoding("gpt2")
cases = [
    "hello world",
    " hello",
    "don't",
    "they're",
    "2024",
    "café",
    "\n",
    "\t",
    "   ",
    "aaaaaaaa",
    "",
    "<|endoftext|>",
]
result = [{"in": c, "out": enc.encode(c, allowed_special="all")} for c in cases]

if len(sys.argv) > 1:
    lines = [
        zigout.header("gen_tokenizer_golden.py"),
        "pub const Case = struct { in: []const u8, out: []const u32 };",
        "pub const cases: []const Case = &.{",
    ]
    for case in result:
        s, ids = case["in"], case["out"]
        in_lit = zigout.string(s)
        out_lit = zigout.u32_slice(ids)
        # \\ strings capture to EOL, so .out must start on the next line
        lines.append(f"    .{{ .in = {in_lit}")
        lines.append(f"    , .out = {out_lit} }},")
    lines.append("};")

    with open(sys.argv[1], "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
else:
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
