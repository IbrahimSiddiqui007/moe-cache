# Documentation map

One naming rule (DOCS-1003): files are `lowercase-kebab-case.md`; `README.md` and the release notes (`releases/RELEASE_NOTES_vX.Y.Z.md`, named by the tag) are the only exceptions. One page per topic.

| Folder | What is in it | For |
|---|---|---|
| [`guides/`](guides/) | how to do something: install, quick start, Windows, coding agents, the control page, LAN access, FAQ, troubleshooting, sharing your numbers | people who use moe-cache |
| [`reference/`](reference/) | what it is and how it is set: configuration, models, backends, platforms and compatibility, quality gate, exactness, how it works, techniques, security, roadmap, the benchmark record | people who look something up |
| [`results/`](results/) | the current numbers ([results-current.md](results/results-current.md), [benchmark-card.html](results/benchmark-card.html)) and every older measurement in [`results/archive/`](results/archive/) | people who want the evidence |
| [`releases/`](releases/) | one `RELEASE_NOTES_vX.Y.Z.md` per version (the release workflow publishes the file of the tag) | everyone, per version |
| [`internal/`](internal/) | design notes, coding standards, the release checklist and how a release is made, performance analysis | people who build, test or release it |

`python3 tools/check_doc_links.py` checks that every relative link and every repository path in a code span points at something that exists (the quick tests run it).
