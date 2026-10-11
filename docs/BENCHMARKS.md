# Benchmarks: what is normally measured, and what moe-cache-bench covers

moe-cache does not change what a model knows: the generated tokens are identical to stock llama.cpp. So the benchmarks that matter here are
**speed and memory** (where moe-cache differs) and **identity / quality drift** (to prove that it does not differ). The usual capability
benchmarks are listed too, because people ask for them; they can be run against the same server with an external harness.

## 0. Quick recipe: measure your own use case

1. `bin/moe-cache-bench` and open the page (it opens by itself; the page carries the access token, so use the page the tool opened, not a bookmark from an earlier run).
2. Model: your GGUF. Limit: the memory you want to emulate. For a model that fits in RAM you will see about stock speed (a few percent either way); the gain is when the limit is smaller than the model.
3. Paste your own prompts if the default four are not like your work (blocks separated by `---`, optional `EXPECT: word` lines).
4. Start. The tool waits for the CPU to cool to 65 C and for background load to fall under its limit before each run, and flags a run that could not be clean. Repeat flagged runs.
5. Open the report: speed per memory limit, first-token latency, SSD read per token, cache hit rate by layer. Identity must say PASS; if it says FAIL, do not trust the speed and please open an issue with the model name.
6. Share the HTML report or the CSV: both are free of prompts, outputs and local paths.

On a remote machine: `ssh -L 8765:127.0.0.1:8765 host`, then open `http://127.0.0.1:8765` locally (the tool only listens on loopback).
For a comparable set of cells across models use `bench/standard.py` (see TEST_PROTOCOL.md).

## 0b. Benchmark many models and publish the numbers

```bash
bench/run_matrix.py --ollama --dir ~/models --exclude Q2_0           # plan: which MoE models, which limits (dense models are skipped)
bench/run_matrix.py --ollama --dir ~/models --exclude Q2_0 --run     # run (hours; start it before you go to bed)
bench/run_matrix.py --model big.gguf --limits 12,16,24 --run         # one model, limits of your choice
```

Per model it runs two placements (all experts on the CPU, and as many expert layers on the GPU as fit) at several hard memory limits (default 0.4, 0.6, 0.8 and 1.1 times the file size), stock and moe-cache, both orders,
and writes a markdown table with the hardware line and a CSV. Before posting: look at the `Clean` column and rerun rows marked NO (close browsers and other heavy programs first);
quote the hardware line, the llama.cpp commit (`moe-cache-check` prints it) and the moe-cache version; say that the memory limits are cgroup limits on a machine with more RAM. The tables contain no prompts, outputs or paths.

## 1. Speed and resource benchmarks (built into moe-cache-bench)
| Measure | What it tells you | How moe-cache-bench does it |
|---|---|---|
| Generation speed (tok/s), steady state | the number users feel | 4 prompts, request 1 = warm-up, mean of requests 2-4 |
| First request vs steady | cold-start cost | reported separately |
| Prompt processing speed (tok/s) | long-prompt behaviour, summaries, code review | one long prompt (about 2,000 tokens) |
| Time to first token | latency of a chat reply | prompt time from the server timings |
| Speed under a memory limit | the case moe-cache is made for | each run is placed in a memory-limited cgroup (several limits can be compared) |
| Cache hit rate, SSD read volume | why a run is fast or slow | read from the plugin's own statistics |
| Run-to-run variation | whether a number can be trusted | repeats, forward and reverse order, start temperature recorded |
| `llama-bench` style pp/tg sweeps | comparison with other people's numbers | not duplicated: use `llama-bench` itself |

### Deep metrics recorded for every request (hover any chart in moe-cache-bench for the exact numbers)
| Metric | Source | Why it matters |
|---|---|---|
| SSD MB read per generated token | `/proc/PID/io` (stock and moe-cache alike) | the cost of not fitting in RAM; stock pays it as page-cache misses |
| Major page faults per token, CPU ms per token | `/proc/PID/stat` | thrashing and spinning show up here |
| Cache hit rate (decode steps, per request, per layer) | plugin counters (`SIGUSR1` snapshot) | how well the LRU cache fits the workload; which layers keep missing |
| Evictions per token, mean idle time of evicted experts | plugin counters | cache churn; a long idle time means the LRU choice was good |
| Time per token: expert compute / SSD reads / cache bookkeeping / everything else | plugin timers | tells you whether to buy a faster SSD, more RAM or a faster CPU |
| Time to first token, output speed and end-to-end latency at 100 / 1,000 / 4,000 input tokens | the server's timings | the Artificial-Analysis-style latency profile |
| Peak memory (cgroup), RSS anonymous vs file-backed, GPU memory, CPU temperature | `/sys/fs/cgroup`, `/proc`, `nvidia-smi`, hwmon | memory headroom, and whether heat explains a slow-down |

## 2. Identity and quality-drift benchmarks (built in; the perplexity ones need `llama-perplexity` and a text file)
| Measure | What it tells you |
|---|---|
| Token identity vs stock (temperature 0, fixed seed) | the plugin changes nothing: the generated token ids must be the same |
| Perplexity on a text file (Wikitext-2 test set by convention in llama.cpp) | how well the model predicts text; must match stock |
| KL divergence between stock and plugin (`llama-perplexity --kl-divergence`) | 0 means the two output distributions are the same |
| Custom prompts with expected answers | your own regression set: pass/fail per prompt, stock vs plugin |

## 3. Capability benchmarks (not re-implemented: run them with an external harness against the server)
These measure the model, not moe-cache. The datasets are large and need downloads, so moe-cache-bench only prints the ready-made command for
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) against the running server.

| Benchmark | Measures | Format |
|---|---|---|
| MMLU / MMLU-Pro | broad knowledge (MMLU-Pro: harder, 10 answer choices) | multiple choice |
| GPQA (Diamond) | graduate-level science reasoning | multiple choice |
| BBH (BIG-Bench Hard, 23 tasks) | multi-step reasoning | multiple choice / short answer |
| MuSR | long multi-step reasoning | multiple choice |
| IFEval | following precise instructions | generative, rule-checked |
| MATH (level 5), GSM8K | mathematics (GSM8K: grade-school word problems) | generative |
| HumanEval, MBPP | writing correct code (HumanEval: 164 problems with unit tests) | generative, executed |
| ARC, HellaSwag, WinoGrande, TruthfulQA | common sense and truthfulness | multiple choice |
| Long-context tests (needle in a haystack, RULER) | use of long inputs | generative |

The first group of rows (MMLU-Pro, GPQA, BBH, MuSR, IFEval, MATH level 5) is the Open LLM Leaderboard v2 suite, which uses a fork of lm-evaluation-harness.

## 4. How to run a capability benchmark through moe-cache
1. Start the server: `bin/moe-cache-server MODEL.gguf --ram 24` (or plain `llama-server` for the stock side).
2. `pip install lm-eval`, then for example:
   `lm_eval --model local-completions --model_args base_url=http://localhost:8080/v1/completions,model=any,num_concurrent=1 --tasks gsm8k,hellaswag --limit 200`
3. Run it once for stock and once with the plugin. The scores must match (same tokens in, same tokens out); the speed is what differs.

Sources: Open LLM Leaderboard v2 task list and descriptions via the lm-evaluation-harness leaderboard README; HumanEval, GPQA and MMLU-Pro descriptions from
the benchmark overviews found on [Athina](https://hub.athina.ai/blogs/top-10-llm-benchmarking-evals) and [Medium](https://medium.com/@dibyajyoti_20397/llm-benchmarks-simplified-from-mmlu-to-gpqa-7e88b6a83c0c);
perplexity and KL-divergence usage from the llama.cpp [perplexity README](https://huggingface.co/OpenTransformer/llama.cpp-prismml/blob/main/tools/perplexity/README.md).


## Finding your models, running many, and serving (since 0.2.2)

`bin/moe-cache-bench` now looks for your models itself. It scans the Ollama store (`~/.ollama`, `/var/lib/ollama`, `$OLLAMA_MODELS`), the llama.cpp cache
(`~/.cache/llama.cpp`), Hugging Face (`~/.cache/huggingface/hub`), LM Studio, `~/models`, `~/Downloads`, the folders in `MOE_CACHE_MODEL_DIRS`
(colon separated) and any folder you add in the page. Draft heads (`mtp-*`) and vision projectors (`mmproj-*`) are left out, split models show once with the
size of all parts. The list shows architecture, expert count, quant and size; **select all MoE** queues every MoE model, one after another, each with the
same settings and its own saved result.

The **Serve** card starts a llama-server for one model (stock or with the plugin, optionally with all free VRAM as llama.cpp's GPU expert cache) and shows
the address, an API key and a ready-made `lm_eval` command, so quality benchmarks no longer need a separately started server. Servers started here are
stopped when the page's server stops; at most two run at once.
