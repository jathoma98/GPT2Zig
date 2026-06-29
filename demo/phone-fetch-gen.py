#!/usr/bin/env python3
# Build a neofetch-style snapshot for the OnePlus, using the phone's REAL specs.
# Output is raw bytes (ANSI escapes included) -> replay on device with `cat`.

G = "\033[1;32m"   # android green (logo + labels)
W = "\033[0m"      # reset
B = "\033[1m"      # bold (title)

logo = [
    "         -o          o-",
    "          +hydNNNNdyh+",
    "        +mMMMMMMMMMMMMm+",
    "      `dMMm:NMMMMMMN:mMMd`",
    "      hMMMMMMMMMMMMMMMMMMh",
    "  ..  yyyyyyyyyyyyyyyyyyyy  ..",
    ".mMMm`MMMMMMMMMMMMMMMMMMMM`mMMm.",
    ":MMMM:MMMMMMMMMMMMMMMMMMMM:MMMM:",
    ":MMMM:MMMMMMMMMMMMMMMMMMMM:MMMM:",
    ":MMMM:MMMMMMMMMMMMMMMMMMMM:MMMM:",
    ":MMMM:MMMMMMMMMMMMMMMMMMMM:MMMM:",
    "-MMMM-MMMMMMMMMMMMMMMMMMMM-MMMM-",
    " +yy+ MMMMMMMMMMMMMMMMMMMM +yy+",
    "      mMMMMMMMMMMMMMMMMMMm",
    "      `/++MMMMh++hMMMM++/`",
    "          MMMMo  oMMMM",
    "          MMMMo  oMMMM",
    "          oNMm-  -mMNs",
]

title = "shell@OnePlus"
fields = [
    ("", title, True),
    ("", "-" * len(title), False),
    ("OS", "Android 16 (API 36) aarch64"),
    ("Host", "OnePlus CPH2583 (OP595DL1)"),
    ("Kernel", "6.1.118-android14-11"),
    ("Uptime", "59 days, 16 hours"),
    ("Shell", "mksh"),
    ("CPU", "Qualcomm Snapdragon 8 Gen 3 (SM8650) (8)"),
    ("GPU", "Qualcomm Adreno (TM) 750"),
    ("Vulkan", "1.3 — compute (GPT-2 inference)"),
    ("Memory", "~14.8 GiB"),
]

def info_line(item):
    if len(item) == 3 and item[2] is True:        # title
        return f"{B}{G}{item[1]}{W}"
    if len(item) == 3 and item[2] is False:        # underline
        return f"{G}{item[1]}{W}"
    key, val = item[0], item[1]
    return f"{G}{B}{key}{W}: {val}"

# color swatch rows like neofetch
sw1 = "".join(f"\033[4{n}m   " for n in range(8)) + W
sw2 = "".join(f"\033[10{n}m   " for n in range(8)) + W
info = [info_line(f) for f in fields] + ["", sw1, sw2]

LOGO_W = 34
out = []
rows = max(len(logo), len(info))
for i in range(rows):
    l = logo[i] if i < len(logo) else ""
    r = info[i] if i < len(info) else ""
    out.append(f"{G}{l}{W}".ljust(LOGO_W + len(G) + len(W)) + "  " + r)

import sys
sys.stdout.write("\n" + "\n".join(out) + "\n\n")
