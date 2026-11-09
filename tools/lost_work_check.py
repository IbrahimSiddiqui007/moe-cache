#!/usr/bin/env python3
"""lost_work_check: find work that sits on a branch or tag but is not in the release line.

Why this exists (2026-11-09): next/v0.3.7 was archive-tagged instead of merged and took the owner-approved
desktop day 1-2 work (app icon, updater) with it; nobody noticed for six days. A rule in RULES.md is not
enough, so this tool makes the check mechanical.

What it does (read-only: it only runs git cherry / show / diff-tree / rev-list / cat-file / for-each-ref and
never writes a ref, an index or the working tree):
  1. Lists every local branch, remote branch and tag (except the base itself, main, release tags v<digit>,
     CI trigger branches and anything matching --ignore).
  2. For each ref, takes the commits that `git cherry <base> <ref>` marks '+' (patch not in the base).
  3. For each such commit measures how much of its effect survives in the base, even if it arrived under a
     different sha (rebase, cherry-pick, squash): added lines that are still in the same file (or a file with
     the same name, after a move), files whose blob is identical, deleted files that are gone.
       SURVIVED >= 80 %   PARTIAL 20-80 %   ABSENT < 20 %
  4. A PARTIAL or ABSENT commit is OK only if it is RECORDED as a loser or as open work:
       - the ref name or the commit sha (7+ chars) appears in a LEDGER_ARCHIVE.md row, or
       - an entry in the accept file names the ref (fnmatch) and a ledger code that exists in one of the three
         ledgers (tasks, ideas, archive), with a reason.
  5. Ledger claims: every "merged <sha>" in the ledgers must be an ancestor of the base, or have its effect in the
     base (survival >= 80 %), or the row must say it is not merged, or the accept file must list "claim:<sha>".
Exit code 0 = nothing unrecorded, 1 = something is lost or unrecorded, 2 = tool error.

Typical use:
  python3 parts/tools/lost_work_check.py                         # everything (release session, handover scan)
  python3 parts/tools/lost_work_check.py --ref claude/foo        # before archiving or deleting one branch
  python3 parts/tools/lost_work_check.py --json > check.json
Accept file format (tab separated, '#' comments): ref<TAB>ledger code<TAB>reason[<TAB>sha,sha]   (ref may use * wildcards;
with a 4th column only those commits are accepted, so a later unrelated commit on the same ref still fails).
"""
import argparse, collections, fnmatch, json, os, re, subprocess, sys
from pathlib import Path

SURV_OK, SURV_PARTIAL = 0.80, 0.20
DEFAULT_IGNORE = [r"(^|/)ci/", r"(^|/)windows-ci$", r"(^|/)sanitize$", r"/HEAD$", r"^v\d", r"/v\d", r"^stash"]


class Git:
    def __init__(self, repo):
        self.repo = str(repo)
        self._blob = {}
        self._tree = None

    def run(self, *args):
        p = subprocess.run(["git", "-C", self.repo, *args], capture_output=True, text=True, errors="replace")
        return p.returncode, p.stdout

    def out(self, *args):
        return self.run(*args)[1]

    def show_file(self, rev, path):
        rc, o = self.run("show", f"{rev}:{path}")
        return o if rc == 0 else None

    def blob(self, rev, path):
        k = (rev, path)
        if k not in self._blob:
            rc, o = self.run("rev-parse", "--verify", "-q", f"{rev}:{path}")
            self._blob[k] = o.strip() if rc == 0 else None
        return self._blob[k]

    def paths_by_name(self, rev):
        if self._tree is None:
            self._tree = collections.defaultdict(list)
            for p in self.out("ls-tree", "-r", "--name-only", rev).split("\n"):
                if p:
                    self._tree[p.rsplit("/", 1)[-1]].append(p)
        return self._tree


norm = lambda s: re.sub(r"\s+", " ", s.strip())


def survival(git, base, commit):
    """Return (score, total_items, surviving_items, detail) for one commit."""
    raw = git.out("diff-tree", "-r", "--no-commit-id", "--raw", "--no-renames", commit)
    items = surv = 0
    files = []
    cache = {}

    def body(path):
        if path not in cache:
            t = git.show_file(base, path)
            cache[path] = norm(t) if t is not None else None
        return cache[path]

    for line in raw.split("\n"):
        m = re.match(r"^:\d+ \d+ [0-9a-f]+ ([0-9a-f]+) ([A-Z])\t(.+)$", line)
        if not m:
            continue
        newblob, status, path = m.groups()
        if status == "D":
            items += 1
            gone = git.blob(base, path) is None
            surv += 1 if gone else 0
            files.append((path, "deleted", gone))
            continue
        if git.blob(base, path) == newblob:           # same file content in the base
            patch_lines = [l for l in git.out("show", "--format=", "-U0", commit, "--", path).split("\n")
                           if l.startswith("+") and not l.startswith("+++") and len(norm(l[1:])) >= 12]
            n = max(1, len(patch_lines))
            items += n; surv += n
            files.append((path, "identical", True))
            continue
        patch = git.out("show", "--format=", "-U0", "--no-color", commit, "--", path)
        if "Binary files" in patch or patch.startswith("GIT binary"):
            items += 1
            here = git.blob(base, path) is not None
            files.append((path, "binary differs" if here else "binary missing", False))
            continue
        added = [norm(l[1:]) for l in patch.split("\n") if l.startswith("+") and not l.startswith("+++")]
        added = [t for t in added if len(t) >= 12]
        if not added:
            continue
        candidates = [path] + [p for p in git.paths_by_name(base).get(path.rsplit("/", 1)[-1], []) if p != path]
        best = 0
        for cp in candidates[:6]:
            b = body(cp)
            if b is None:
                continue
            best = max(best, sum(1 for t in added if t in b))
        items += len(added); surv += best
        files.append((path, f"{best}/{len(added)} lines", best == len(added)))
    score = (surv / items) if items else 1.0
    return score, items, surv, files


def status_of(score):
    return "SURVIVED" if score >= SURV_OK else ("PARTIAL" if score >= SURV_PARTIAL else "ABSENT")


def ledger_rows(ledger_dir):
    rows = {}
    for name in ("LEDGER_TASKS.md", "LEDGER_IDEAS.md", "LEDGER_ARCHIVE.md"):
        p = Path(ledger_dir) / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").split("\n"):
            m = re.match(r"^\| ([A-Z]+-\d+) \|", line)
            if m:
                rows.setdefault(m.group(1), []).append((name, line))
    return rows


def load_accept(path):
    out = []
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").split("\n"):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            c = line.split("\t")
            if len(c) >= 2:
                commits = [x.strip() for x in c[3].split(",") if x.strip()] if len(c) > 3 else []
                out.append((c[0].strip(), c[1].strip(), c[2].strip() if len(c) > 2 else "", commits))
    return out


def list_refs(git, base, ignore, include):
    base_short = base.split("/", 1)[-1] if "/" in base else base
    out = git.out("for-each-ref", "--format=%(refname)\t%(objectname)", "refs/heads", "refs/remotes", "refs/tags")
    refs = collections.OrderedDict()
    for line in out.split("\n"):
        if not line:
            continue
        full, sha = line.split("\t")
        short = re.sub(r"^refs/(heads|remotes|tags)/", "", full)
        if include and not any(fnmatch.fnmatch(short, i) or short == i for i in include):
            continue
        if short in (base, base_short, f"origin/{base_short}", "origin/main", "main", "master", "origin/master", "origin"):
            continue
        if any(re.search(p, short) for p in ignore):
            continue
        refs[short] = sha
    return refs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve()
    ap.add_argument("--repo", default=str(here.parents[3] / "moe-cache"), help="git repo (default: sibling moe-cache)")
    ap.add_argument("--base", default="origin/experimental", help="release line (default origin/experimental)")
    ap.add_argument("--ledger-dir", default=str(here.parents[2]), help="folder with the three ledgers")
    ap.add_argument("--accept", default=str(here.with_name("lost_work_accepted.tsv")), help="accept file (see docstring)")
    ap.add_argument("--ref", action="append", help="check only this ref (fnmatch allowed); repeatable")
    ap.add_argument("--ignore", action="append", default=[], help="extra regex of refs to skip")
    ap.add_argument("--no-claims", action="store_true", help="skip the ledger 'merged <sha>' check")
    ap.add_argument("--records", action="append", default=[], help="extra text file whose lines count as archive records (e.g. archive/README.md)")
    ap.add_argument("--public", action="store_true", help="CI mode in the public repo: no ledgers; accept-file codes are not validated, claims not checked")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="also list SURVIVED commits")
    a = ap.parse_args(argv)

    git = Git(a.repo)
    if git.run("rev-parse", "--verify", "-q", a.base)[0] != 0:
        print(f"error: base {a.base} not found in {a.repo}", file=sys.stderr)
        return 2
    rows = ledger_rows(a.ledger_dir)
    arch_rows = [(code, l) for code, ls in rows.items() for (n, l) in ls if n == "LEDGER_ARCHIVE.md"]
    for rf in a.records:
        if Path(rf).exists():
            arch_rows += [(f"record:{Path(rf).name}", l) for l in Path(rf).read_text(encoding="utf-8", errors="replace").split("\n") if l.strip()]
    accept = load_accept(a.accept)

    def recorded(ref, sha):
        for pat, code, why, commits in accept:
            if commits and not any(sha.startswith(x) or x.startswith(sha[:7]) for x in commits):
                continue
            if (fnmatch.fnmatch(ref, pat) or ref == pat) and (code in rows or a.public):
                return f"{code} (accept file: {why[:60]})"
        for code, l in arch_rows:
            if re.search(r"(?<![\w/.-])" + re.escape(ref) + r"(?![\w.-])", l) or (len(sha) >= 7 and sha[:7] in l):
                return f"{code} (archive row)"
        return None

    refs = list_refs(git, a.base, DEFAULT_IGNORE + a.ignore, a.ref)
    by_tip = collections.OrderedDict()
    for ref, sha in refs.items():
        by_tip.setdefault(sha, []).append(ref)
    results, seen_commit = [], {}
    for tip, names in by_tip.items():
        cherry = git.out("cherry", a.base, tip)
        commits = [l[2:].strip() for l in cherry.split("\n") if l.startswith("+ ")]
        for c in commits:
            if c not in seen_commit:
                score, items, surv, files = survival(git, a.base, c)
                subj = git.out("log", "-1", "--format=%s", c).strip()
                seen_commit[c] = (score, items, surv, files, subj)
            score, items, surv, files, subj = seen_commit[c]
            st = status_of(score)
            rec = None
            if st != "SURVIVED":
                for n in names:
                    rec = recorded(n, c)
                    if rec:
                        break
            results.append(dict(refs=names, commit=c[:9], subject=subj[:80], status=st, score=round(score, 2), items=items,
                                recorded=rec, bad=(st != "SURVIVED" and not rec), files=[f"{p}: {d}" for p, d, ok in files[:6]]))
    claims = [] if (a.no_claims or a.public) else check_claims(git, a.base, a.ledger_dir, accept)
    bad = [r for r in results if r["bad"]]
    bad_claims = [c for c in claims if c["bad"]]
    if a.json:
        print(json.dumps(dict(base=a.base, refs=len(refs), results=results, claims=claims, bad=len(bad) + len(bad_claims)), indent=1))
    else:
        shown = [r for r in results if a.verbose or r["status"] != "SURVIVED"]
        print(f"lost_work_check: base {a.base} ({git.out('rev-parse', '--short', a.base).strip()}), {len(refs)} refs, "
              f"{len(results)} commits not matched by git cherry, {len(bad)} unrecorded, {len(bad_claims)} bad ledger claims\n")
        if shown:
            print("| Ref | Commit | Subject | Effect in base | Recorded by |\n|---|---|---|---|---|")
            for r in shown:
                print(f"| {', '.join(r['refs'])} | {r['commit']} | {r['subject'].replace('|', '/')} | {r['status']} {int(r['score']*100)} % | "
                      f"{r['recorded'] or '**NOT RECORDED**'} |")
        if claims:
            cb = [c for c in claims if c["bad"] or a.verbose]
            if cb:
                print("\n| Ledger line | Claim | Result |\n|---|---|---|")
                for c in cb:
                    print(f"| {c['where']} | merged {c['sha']} | {c['result']} |")
        print("\nRESULT:", "FAIL (lost or unrecorded work)" if (bad or bad_claims) else "OK (everything is in the base or recorded)")
    return 1 if (bad or bad_claims) else 0


def check_claims(git, base, ledger_dir, accept):
    out = []
    accepted = {c[0][6:] for c in accept if c[0].startswith("claim:")}
    seen = set()
    for name in ("LEDGER_TASKS.md", "LEDGER_IDEAS.md", "LEDGER_ARCHIVE.md"):
        p = Path(ledger_dir) / name
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
            if "merged" not in line.lower():
                continue
            # a claim is "merged <sha>", "merged as/at/in/by <sha>", "merged into experimental (<sha>)": a short gap, no sentence
            for m in re.finditer(r"\b[Mm]erged(?: into (?:experimental|main))?(?: as| at| in| by| via| with| commit)?[\s:(,]{1,4}([0-9a-f]{7,40})\b", line):
                sha = m.group(1)
                if (name, i, sha) in seen:
                    continue
                seen.add((name, i, sha))
                where = f"{name}:{i}"
                rc, full = git.run("rev-parse", "--verify", "-q", f"{sha}^{{commit}}")
                if rc != 0:
                    out.append(dict(where=where, sha=sha, result="not in this repo (upstream or typo): not checked", bad=False))
                    continue
                full = full.strip()
                if git.run("merge-base", "--is-ancestor", full, base)[0] == 0:
                    out.append(dict(where=where, sha=sha, result="ancestor of base", bad=False))
                    continue
                score = survival(git, base, full)[0]
                if score >= SURV_OK:
                    out.append(dict(where=where, sha=sha, result=f"effect in base ({int(score*100)} %), other sha", bad=False))
                elif re.search(r"\b(not|never|un)\s?merged|not for merge|archived|killed", line, re.I) or sha in accepted \
                        or sha[:7] in accepted:
                    out.append(dict(where=where, sha=sha, result=f"effect {int(score*100)} % but the row says not merged / archived / accepted", bad=False))
                else:
                    out.append(dict(where=where, sha=sha, result=f"CLAIMED MERGED, effect in base only {int(score*100)} %", bad=True))
    return out


if __name__ == "__main__":
    sys.exit(main())
