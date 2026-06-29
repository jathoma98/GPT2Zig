# LAN Multi-Device Setup: 1 Master + 3 Slaves

This guide takes the GPT-2 pipeline-parallel inference engine from the loopback-only PoC
(everything on one Mac, glued together with `socat`) to a real four-machine fleet talking
over your LAN.

**Topology for this demo**

| Role    | Machine                                  | GPU            | Reaches master via            |
| ------- | ---------------------------------------- | -------------- | ----------------------------- |
| Master  | This MacBook                             | Apple (MoltenVK) | —                           |
| Slave 1 | Ubuntu 26 VM in UTM (on the MacBook)     | (CPU/llvmpipe or virtio) | UTM shared-net gateway |
| Slave 2 | Windows laptop                           | NVIDIA         | LAN IP                        |
| Slave 3 | OnePlus phone (Qualcomm Adreno)          | Adreno (Vulkan 1.3) | `adb reverse` tunnel    |

The master runs HEAD (embedding) + the first layer chunk + TAIL (final norm, logits, sampling),
then routes the residual stream through each slave in pipeline order. Each slave owns one
contiguous block of transformer layers. One message per machine boundary per token — LAN latency
is fine for this. See [model-parallel-cross-machine-inference.md](model-parallel-cross-machine-inference.md)
for the architectural rationale.

---

## 0. What changed in the code

Previously the master hardcoded its listen address to `127.0.0.1`, so slaves could only reach it
through a `socat`/port-forward shim. The master now binds a **configurable** interface:

- `serverParams.listenAddr` in the master config — defaults to `"0.0.0.0"` (all interfaces, i.e.
  reachable over LAN). Set it to `"127.0.0.1"` to go back to loopback-only.
- Slaves already had `serverParams.masterAddr` (the IP/hostname they dial); no change there.

Files touched: `src/dist/runconfig.zig` (parse `listenAddr`), `src/dist/master.zig` (bind it),
`src/main.zig` (thread it through). The wire protocol, framing, and FSMs are unchanged.

Example configs live in `config/`:
- [`config/master-3slaves.json`](../config/master-3slaves.json) — master, binds `0.0.0.0`, waits for 3 slaves.
- [`config/slave-lan.json`](../config/slave-lan.json) — slave template; edit `masterAddr` to the Mac's LAN IP.
- [`config/slave-adb.json`](../config/slave-adb.json) — phone slave; dials `127.0.0.1` (tunneled by adb).

---

## 1. Find the Mac's LAN IP (the master address)

On the MacBook:

```bash
ipconfig getifaddr en0      # Wi-Fi; try en1 if blank (Ethernet/Thunderbolt)
```

Call this `MASTER_IP` (e.g. `192.168.1.42`). The Windows laptop will dial this. The Ubuntu VM
dials a *different* address (the UTM gateway — see §4), and the phone dials `127.0.0.1` (tunneled).
Because the master binds `0.0.0.0`, all three paths hit the same listener.

> A LAN IP from DHCP can change between sessions. For a demo, either note it right before recording
> or give the Mac a DHCP reservation in your router. You can also use the Mac's `.local` mDNS name
> (`scutil --get LocalHostName` → `name.local`) as `masterAddr` from the Windows box.

---

## 2. macOS firewall & safety

Binding `0.0.0.0` means any host on your LAN can open a TCP connection to port `9876`. For a home
LAN this is fine, but be aware:

- **No authentication.** The master accepts the first `expectedSlaves` connections that arrive and
  hands each a layer range. A rogue device on the LAN could grab a slot. Acceptable for a trusted
  home network / demo; do **not** expose port `9876` to the internet (don't port-forward it on your
  router).
- **macOS Application Firewall.** If it's on (System Settings → Network → Firewall), the first run
  triggers an "Allow incoming connections for GPT2Zig?" prompt — click **Allow**. To pre-authorize
  from the CLI:
  ```bash
  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /full/path/to/zig-out/bin/GPT2Zig
  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /full/path/to/zig-out/bin/GPT2Zig
  ```
- **To lock it down** for a quieter demo, set `listenAddr` to a *specific* interface IP instead of
  `0.0.0.0` — but note the three slaves arrive on three different interfaces (LAN `en0`, UTM
  `bridge`/`vmnet`, and loopback for adb), so `0.0.0.0` is the only single value that serves all
  three at once. Keep `0.0.0.0` for this demo.

---

## 3. Build prerequisites (all machines)

Each machine runs the **same** `GPT2Zig` binary; the config picks master vs slave. Each slave needs:
- The binary built for *its* OS/CPU.
- A Vulkan-capable GPU + loader (`libvulkan.so` / `vulkan-1.dll`). The engine loads Vulkan at
  runtime via `dlopen`/`LoadLibrary`, so you don't link it at build time.
- A copy of `models/gpt2/model.safetensors` (~500 MB) at the path named in its config.

Install Zig 0.16 on any machine you build *on*. You can also cross-compile everything from the Mac
with `-Dtarget=...` (used for the phone below).

Sanity check on the Mac:
```bash
zig build -Doptimize=ReleaseSafe         # builds zig-out/bin/GPT2Zig
zig build run -- config/master-local.json   # 0 slaves: whole model in-process, no sockets
```
If local mode produces text, the engine + model are good; only networking remains.

---

## 4. Slave 1 — Ubuntu 26 VM (UTM)

### Networking
UTM's default **"Shared Network"** puts the VM on a private `192.168.64.x` subnet, NAT'd to the host.
From inside the VM the Mac host is reachable at the **gateway**, typically `192.168.64.1`:
```bash
ip route | awk '/default/ {print $3}'    # the gateway = the Mac, e.g. 192.168.64.1
```
Use that gateway address as `masterAddr` in the VM's slave config (it is *not* the Mac's `en0` IP).

> Alternative: set the VM's network to **"Bridged"** in UTM. Then the VM gets a real `192.168.1.x`
> LAN address and uses the same `MASTER_IP` as the Windows box. Bridged is simpler conceptually but
> depends on your router allowing it; Shared Network always works.

### Build & run (inside the VM)
```bash
sudo apt update && sudo apt install -y git build-essential
# install Zig 0.16 (download tarball or use your preferred method), then:
git clone <repo> GPT2Zig && cd GPT2Zig
zig build -Doptimize=ReleaseSafe
# copy model.safetensors into models/gpt2/ (scp from the Mac, see below)

cat > config/slave.json <<'EOF'
{ "model_path": "models/gpt2/model.safetensors",
  "serverParams": { "type": "slave", "masterAddr": "192.168.64.1" } }
EOF
zig build run -- config/slave.json
```
Copy the model from the Mac: `scp models/gpt2/model.safetensors user@192.168.64.x:~/GPT2Zig/models/gpt2/`

> GPU note: a UTM VM usually has **no** passthrough GPU, so Vulkan may fall back to a software
> rasterizer (llvmpipe) or be absent. If `Gpu.init` reports Vulkan unavailable, install
> `mesa-vulkan-drivers` (`sudo apt install -y mesa-vulkan-drivers vulkan-tools` and verify with
> `vulkaninfo`). It'll be slow, but functionally it serves its layer shard — fine for a demo.

---

## 5. Slave 2 — Windows laptop (NVIDIA)

### Networking
Find the laptop's LAN IP with `ipconfig` (the IPv4 on your Wi-Fi/Ethernet adapter) only if you need
the Mac to reach *it* — you don't; the slave dials *out* to the master. Just make sure the laptop is
on the same LAN/subnet as the Mac, and that the **Mac's** firewall allows inbound `9876` (§2).

### Build & run (on Windows, PowerShell)
Install Zig 0.16 and the NVIDIA driver (ships the Vulkan ICD). Then:
```powershell
git clone <repo> GPT2Zig; cd GPT2Zig
zig build -Doptimize=ReleaseSafe
# copy model.safetensors into models\gpt2\

@'
{ "model_path": "models/gpt2/model.safetensors",
  "serverParams": { "type": "slave", "masterAddr": "192.168.1.42" } }
'@ | Out-File -Encoding ascii config\slave.json   # <- set masterAddr to MASTER_IP

zig build run -- config/slave.json
```

> If Windows Defender Firewall prompts about the binary making *outbound* connections, allow it
> (outbound is usually permitted by default). NVIDIA gives you real Vulkan 1.3 — this slave will be
> the fast one.

---

## 6. Slave 3 — OnePlus phone (Adreno, via adb)

The phone runs the slave binary in `adb shell` (no Android Activity/APK), and its socket is tunneled
to the Mac over USB with `adb reverse`. This is the fiddliest device — budget extra time.

### 6a. Cross-compile the binary for Android (on the Mac)
```bash
zig build -Doptimize=ReleaseSafe -Dtarget=aarch64-linux-android
# -> zig-out/bin/GPT2Zig (an aarch64 Android/bionic binary)
```
Bionic (Android's libc) is required so the binary can `dlopen` the device's `/system/lib64/libvulkan.so`.
A musl/gnu static target cannot load the system Vulkan loader.

### 6b. Push binary + model to the phone
`adb shell` runs with limited permissions; `/data/local/tmp` is the reliable writable, executable
location:
```bash
adb shell mkdir -p /data/local/tmp/gpt2zig
adb push zig-out/bin/GPT2Zig            /data/local/tmp/gpt2zig/GPT2Zig
adb push models/gpt2/model.safetensors  /data/local/tmp/gpt2zig/model.safetensors
adb push config/slave-adb.json          /data/local/tmp/gpt2zig/slave.json
adb shell chmod 755 /data/local/tmp/gpt2zig/GPT2Zig
```
`config/slave-adb.json` already points `model_path` at `/data/local/tmp/gpt2zig/model.safetensors`
and sets `masterAddr` to `127.0.0.1`.

### 6c. Tunnel the phone's socket to the Mac
`adb reverse` makes a port on the *phone* forward to a port on the *Mac*:
```bash
adb reverse tcp:9876 tcp:9876
```
Now when the slave on the phone dials `127.0.0.1:9876`, adb forwards it to the Mac's `127.0.0.1:9876`,
which the `0.0.0.0`-bound master is listening on. (`adb reverse` resets on unplug/replug — re-run it.)

### 6d. Run the slave
```bash
adb shell "cd /data/local/tmp/gpt2zig && ./GPT2Zig slave.json"
```

> Adreno exposes Vulkan 1.3 to native binaries, but running compute from a bare `adb shell` (outside
> an app sandbox) can be hit-or-miss depending on OnePlus's OS build and SELinux policy. If
> `Gpu.init` fails to find/load Vulkan, that's the likely culprit — this is the one device that may
> need a real APK wrapper. Validate first with a Vulkan compute sample over adb shell before the demo,
> and have a fallback (e.g. run slave 3 as a second process on the Mac, or drop to 2 slaves with
> `config/master-2slaves.json`).

---

## 7. Bring it all up

**Start order:** master first (it must be listening), then the three slaves. Slaves retry the
connection for ~10 s (40 attempts × 250 ms), so a little start-order slop is forgiving.

1. **Mac (master):**
   ```bash
   zig build run -- config/master-3slaves.json
   ```
   Watch for `master: listening on 0.0.0.0:9876` and
   `master: waiting for slave 1/3 to connect...`.

2. **Phone tunnel (Mac):** `adb reverse tcp:9876 tcp:9876`

3. **Each slave:** run its `slave.json` (§4, §5, §6d).

The master logs `slave N connected` + `slave N ← directive: layers [lo, hi)` for each, then runs the
forward pass and streams generated tokens to stdout. The `.net` log scope (grep stderr for `(net)`)
shows bring-up at info and per-frame traffic at debug.

**Layer split:** with 4 participants and GPT-2's 12 layers, each owns 3 layers (master = HEAD +
layers [0,3) + TAIL; slaves get [3,6), [6,9), [9,12)). See `partition.layerRange`.

---

## 8. SSH / adb access for the demo recording

Get all four shells visible in one window before you hit record, with no password prompts on screen.

### One-time passwordless access
```bash
# Ubuntu VM: install sshd in the VM first:
#   sudo apt install -y openssh-server && sudo systemctl enable --now ssh
ssh-keygen -t ed25519                 # if you don't already have a key
ssh-copy-id user@192.168.64.3         # Ubuntu VM (its 192.168.64.x address)
ssh-copy-id user@192.168.1.50         # Windows (OpenSSH Server; see below)
```

Enable OpenSSH Server on Windows (PowerShell as admin):
```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic; Start-Service sshd
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server' -Enabled True `
  -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

`~/.ssh/config` aliases so panes show clean names:
```
Host slave1-ubuntu
    HostName 192.168.64.3
    User user
Host slave2-windows
    HostName 192.168.1.50
    User user
```

### Four-pane layout
```bash
tmux new-session -s demo \; \
  send-keys 'cd ~/Workspaces/GPT2Zig; echo MASTER (mac)' C-m \; \
  split-window -h 'ssh slave1-ubuntu' \; \
  split-window -v 'ssh slave2-windows' \; \
  select-pane -L \; split-window -v 'adb shell' \; \
  select-layout tiled
```
This gives Mac-master / Ubuntu / Windows / phone in one recordable window. (iTerm2 splits via ⌘D /
⌘⇧D work equally well.) The phone is `adb shell` rather than SSH; it has no SSH server.

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| Slave logs `connection refused`, then gives up | Master not listening yet, or wrong `masterAddr`. Start master first; verify the IP (VM uses the **gateway**, not the Mac's `en0` IP). |
| Slave can't reach master at all | macOS firewall blocking `9876` (§2), or devices on different subnets/VLANs (guest Wi-Fi isolates clients). |
| `master: invalid listenAddr` | `listenAddr` isn't a valid IP literal. Use `"0.0.0.0"` / `"127.0.0.1"` (not a hostname). |
| `Vulkan unavailable` on a slave | No GPU/driver. Ubuntu VM: install `mesa-vulkan-drivers`. Phone: see §6 caveat. |
| Phone slave can't connect | `adb reverse` was reset (replug) — re-run `adb reverse tcp:9876 tcp:9876`. Confirm with `adb reverse --list`. |
| Master hangs at `waiting for slave N/3` | Fewer slaves connected than `expectedSlaves`. Drop to `config/master-2slaves.json`, or start the missing slave. |
| Garbled output / wrong tokens | All machines must run the **same** model file and the same build. Re-`scp`/`adb push` `model.safetensors`. |

To test the wiring without three physical machines, run several slaves as separate processes on the
Mac itself (all dialing `127.0.0.1`) against `config/master-3slaves.json` — the master can't tell
local from remote slaves.
