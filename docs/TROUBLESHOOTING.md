# Troubleshooting

Commands are given for both systems. On Linux use `python3 bin/moe_cache.py ...`; on Windows use `python bin\moe_cache.py ...`. Everything below behaves the same on both.

Still stuck? Open a [Bug report](https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=bug_report.yml) and paste the output of `doctor`.

## Answers look wrong, misspelled or loop

moe-cache is exact by default: its output is token-identical to stock llama.cpp (`tests/identity.sh`, CI on Linux and Windows). So when answers look wrong, first find out
whether stock llama.cpp does the same (last row of the table). Go down the table in this order; each row is a cheap check.

| # | Check | Why it matters | How |
|---|---|---|---|
| 1 | **Does stock llama-server do the same?** | If stock gives the same text, the cause is the model file, the settings or the build, not moe-cache | Run the same prompt, same flags, with plain `llama-server` (no plugin). Use temperature 0 and a fixed seed (`--seed 42`) |
| 2 | **Temperature-0 twice** | The same prompt at temperature 0 must give identical text twice. If it does not, suspect unstable hardware (a RAM overclock such as XMP / EXPO at its limit) or a broken build. This is general knowledge, not something we measured | Send the same request twice with `"temperature": 0, "seed": 42`, compare the two answers |
| 3 | **Sampling settings** | Low-bit quants (IQ2, IQ3, Q2) tend to loop on long answers without the model card's settings (temperature, top-p, top-k, min-p, presence penalty). The launcher passes them for Qwen3 / 3.5 / 3.6 / 3.8 and GLM models; plain `llama-server` does not, and your client may send its own values (they win) | The launcher prints the settings it passes. `--no-sampling-defaults` turns them off. Set the card's values in your client if you start `llama-server` yourself |
| 4 | **KV cache type** | q8_0 KV is approximate (passes the quality gate: KL 0.0079, top-1 96.7 %, tasks = stock, Flash-Next Q2_0). At `--ctx 131072` and above it is the default. q4_0 KV is not tested by us | Look at the launcher's start lines or your `-ctk` / `-ctv` flags. `--exact` keeps llama.cpp's f16 KV (and costs VRAM and long-prompt speed) |
| 5 | **An approximate mode** | `--fast`, `--fast=2`, `--fast=fewer` and `--hybrid` change the output on purpose and only when you ask. `fewer` is unproven at 64k-256k ([QUALITY_GATE.md](QUALITY_GATE.md)) | Start without them (`--exact` also forces exact output) and compare |
| 6 | **The model file** | A broken or truncated download, or a different upload of the same quant, gives odd text | Compare the sha256 of your file with the one on the model's Hugging Face page ("Files" > the file > "Copy SHA256"): Linux `sha256sum MODEL.gguf`, Windows `certutil -hashfile MODEL.gguf SHA256` (PowerShell: `Get-FileHash MODEL.gguf`). Download again if it differs |
| 7 | **llama.cpp build, backend, driver** | An old build may lack the model's fixes; CUDA, Vulkan and HIP are different code. AMD on Windows: a Vulkan driver older than 32.0.32015.7029 crashes on Flash-Next checkpoints | `doctor` lists build, backend and driver and says what to fix. Update llama.cpp (tested: b11669, and b11966 or newer) and the GPU driver |
| 8 | **Context too small for the conversation** | When the chat grows past `--ctx`, llama.cpp drops or shifts old text and the model can lose the thread or repeat itself | Raise `--ctx`, or start a new chat. Long contexts cost VRAM: see the README table for tested sizes |

### Check whether the model loops: `tools/loop_check.py`

`loop_check.py` sends 20 long-answer prompts to a running llama-server and counts repeated word sequences. It needs only Python, no extra packages.

1. Start the server as you normally do (`bin/moe-cache-server MODEL.gguf` on Linux, `python bin\moe_cache.py server MODEL.gguf` on Windows).
2. In a second terminal:

| System | Command |
|---|---|
| Linux | `python3 tools/loop_check.py --url http://127.0.0.1:8080 --label mine --json loop_mine.json` |
| Windows | `python tools\loop_check.py --url http://127.0.0.1:8080 --label mine --json loop_mine.json` |

It sends the prompts **without** sampling fields, so the server's own defaults are what gets measured. To compare, run it again against a server started with
`--no-sampling-defaults` (same seed, `--seed` on the script defaults to 1). It takes a while: 20 answers of up to 1500 tokens each.

What the output means:

| Output | Meaning |
|---|---|
| repeat share | share of 8-word sequences that already appeared earlier in the same answer. Normal text is low; a high share means repetition |
| `LOOP` | the answer ends in a repeat (more than half of the last 200 words repeat) or it hit the token limit while repeating |
| summary line | loops / answers, plus median and max repeat share |

Thinking text is counted with the answer, because loops happen there too. A few `LOOP` answers with the sampling defaults off and none with them on means sampling was the cause (row 3). `LOOP` in
both, and also on stock llama-server, points at the file or the quant (rows 6 and 1). `LOOP` only with moe-cache: please open a bug report and attach the JSON file.

Our own task runs (202 exact-answer tasks) answer in 128 tokens or less, which is too short to catch long-answer loops; that is why this script exists.

## Slow

| Symptom | Cause | What to do |
|---|---|---|
| Not faster than stock, or the launcher says it started stock | The model fits in your RAM, so stock llama.cpp already runs from the file cache with the same kernels. moe-cache helps when memory is short or the model is larger than RAM | Use stock llama.cpp (README, "When to use something else"). |
| Answers slow down sharply, or very slow prompts, while a browser or game is open | They hold GPU memory, so the card is too full | Close them, or add `--gpu-margin 2` (GiB of VRAM left free; the default is about 0.5) |
| Misses (experts read from the drive) are slow | The model sits on a SATA SSD, hard disk or USB drive | Run `doctor`: its "drive speed" line times random reads and warns when the drive is slow. Move the model to an NVMe drive |
| Very long prompts take minutes | Reading a 63k-token prompt took about 290 s on our Windows PC (RESULTS 64). Decode is faster than prompt reading, but a long prompt is still a wait | Keep the context as short as the task allows. Closing the browser helps long prompts too (34k: 109-110 vs 67-68 tok/s with a browser open, RESULTS 71) |

## Does not start

1. Run `doctor` on your model: Linux `python3 bin/moe_cache.py doctor MODEL.gguf`, Windows `python bin\moe_cache.py doctor D:\models\MODEL.gguf`. Every line says OK, WARN or FAIL and what to do.
2. Fix the FAIL lines first (typical ones: a llama.cpp build without shared ggml libraries, an AMD driver that is too old, not enough free VRAM or RAM).
3. Still failing: open a [Bug report](https://github.com/IbrahimSiddiqui007/moe-cache/issues/new?template=bug_report.yml) and paste the doctor output and the last lines of the server log.
