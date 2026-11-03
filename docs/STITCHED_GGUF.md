# Stitched GGUF: one file, two quantisations (MODELS-640)

`tools/stitch_gguf.py` builds one GGUF from two quantisations of the same model: the layers that stay resident come from one file (for example IQ3_XXS), the layers
that stream from disk come from the other (for example Q2_0). The tensor bytes are **copied**, nothing is re-quantised, so each tensor is exactly what its source file
holds. Standard library only; it reads and writes in chunks (a model is tens of GB); it never overwrites a file.

```
python3 tools/stitch_gguf.py --resident M-IQ3_XXS.gguf --stream M-Q2_0.gguf --n-cpu-moe 24 --out M-stitched.gguf --verify
python3 tools/stitch_gguf.py --resident A.gguf --stream B.gguf --stream-layers 0-9,20 --scope experts --dry-run
```

| Option | Meaning |
|---|---|
| `--resident FILE` | metadata and every tensor outside the streamed layers (embeddings, output, norms, resident layers) |
| `--stream FILE` | the tensors of the streamed layers |
| `--n-cpu-moe N` | streamed layers = `0 .. N-1`. llama.cpp's `--n-cpu-moe N` keeps the experts of the **first N** layers on the CPU side (the launcher's plan prints N), so those are the layers read from disk |
| `--stream-layers 0-9,20` | the same with explicit block indices |
| `--scope layer` (default) / `experts` | `layer`: every `blk.L.*` tensor of a streamed layer; `experts`: only `ffn_*_exps` (attention and norms stay resident) |
| `--verify` | read the output back and compare each tensor with its source, byte for byte |
| `--dry-run` | run the checks and print the plan, write nothing |

## What is checked before a byte is written
Same architecture, block count and expert count; the same tensor names; the same shape for every tensor; the same alignment; every tensor lies inside its file and, for
the common quant types (F32, F16, Q4_0..Q8_0, Q2_K..Q6_K, IQ1..IQ4), is at least as long as its shape needs, so a truncated download is refused. The two files must
come from the same base model: the tool cannot tell two different fine-tunes with equal shapes apart. Use files from the same publisher and revision.

## Output
`OUT.gguf` and `OUT.gguf.stitch.json`: per tensor its source (`resident` / `stream`), type and bytes, per layer its source (`resident`, `stream`, or `mixed` with
`--scope experts`) and the bytes taken from each file. The output's metadata is the resident file's, byte for byte (`general.file_type` therefore names the resident
quant; llama.cpp uses it only for the printed name, the real types are per tensor).

## What it is not
It is **not exact**: the result differs from both sources, and its quality is not the quality of either. It also departs from the rule that the base model is the
IQ3_XXS file, so it needs the owner's word and the quality gate ([QUALITY_GATE.md](QUALITY_GATE.md)) before it is used for anything but a speed measurement.
Nothing in the launcher uses a stitched file automatically. Tested only on small synthetic files (`tests/stitch_gguf_test.py`); no real model has been stitched yet.

## Plan for the real measurement (needs the owner's OK; not done)
Why: the streamed layers dominate the bytes read from the SSD; Q2_0 experts are smaller than IQ3_XXS ones, so fewer bytes stream per token and per prompt block
(ledger guess: streamed bytes -12 %, prompt +5-8 %, decode +4-6 %). Whether the quality loss is acceptable is the open question.

1. **Files** (owner approves the downloads and the disk: two quant files of the same Flash-Next revision, then the stitched file, about the size of the IQ3_XXS file).
   Stitch with the `--n-cpu-moe` the launcher plans for 64k on the PC; keep the manifest with the results.
2. **Speed, order A B B A** (A = pure IQ3_XXS, B = stitched), same launcher flags, 64k context, cold start each time (restart the server, drop nothing else), nothing else running on the PC:
   about 25 minutes per run (load about 3, 64k prompt, 256 generated tokens) = about 100 minutes for the four runs. Per run record: prefill tok/s at 64k, decode tok/s,
   bytes read from disk (`read_bytes` in the plugin's stats), cache hit rate, peak RAM. A-vs-A and B-vs-B differences are the noise floor.
3. **Quality**, only if speed passes: the quality gate on the stitched file against the IQ3_XXS file (KL and top-1 on the gate's text, then its tasks), about 30 minutes.
4. **Kill rule**: stop and delete the stitched file if (a) the B runs gain less than +3 % prefill **and** less than +2 % decode over the A runs, or the gain is smaller than the A-vs-A spread; or
   (b) the quality gate shows KL above the limit fixed in `QUALITY_GATE.md` before measuring, or any task regression against the IQ3_XXS file; or (c) a B run needs more peak RAM than the A runs. One failed
   run caused by the machine (another program, thermal throttling) is repeated once, not counted.
