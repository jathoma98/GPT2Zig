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
    # M4 additions — ASCII-safe + accented-letter edge cases (the \\p{L} approximation is exact for
    # these). Avoid exotic non-ASCII symbols/digits/whitespace, where it can diverge from tiktoken.
    "Hello, I am",  # the M3 anchor prompt
    "a, b.",
    "!!!??? ...",
    "abc123",
    "a  b",  # collapsed interior spaces: " " gets its own token, "b" keeps one
    "  x",  # two leading spaces
    "I'm you're it's",  # multiple contractions
    "The quick brown fox jumps over the lazy dog.",
    "tab\tafter",
    "line1\nline2",
    "résumé naïve",
    "before<|endoftext|>after",  # special token embedded mid-string
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
