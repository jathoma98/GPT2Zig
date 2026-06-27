import tiktoken, json

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
out = [{"in": c, "out": enc.encode(c, allowed_special="all")} for c in cases]
print(json.dumps(out, ensure_ascii=False, indent=2))
