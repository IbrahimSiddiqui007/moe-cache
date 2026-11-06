# Documentation map

| Folder | What is in it | For |
|---|---|---|
| `docs/` (this folder) | install, Windows, models, agents, the control page, configuration, FAQ, troubleshooting, security, how it works, platform limits, the roadmap | people who use moe-cache |
| [`docs/internal/`](internal/) | design notes, coding standards, the release checklist and how a release is made, performance analysis | people who build, test or release it |
| [`docs/releases/`](releases/) | one `RELEASE_NOTES_vX.Y.Z.md` per version (the release workflow publishes the file of the tag) | everyone, per version |
| [`docs/results-archive/`](results-archive/) | every older measurement ([Windows desktop](results-archive/RESULTS.md), [Linux laptop](results-archive/RESULTS_OLD_LAPTOP.md), the older card page) | people who want the evidence |

The current numbers are in [RESULTS_CURRENT.md](RESULTS_CURRENT.md). `python3 tools/check_doc_links.py` checks that every relative link in the Markdown files points at something that exists (the quick tests run it).
