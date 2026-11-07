"""Tweet performance log (stdlib only, so it runs with the system python3 even when the uv venv is unavailable).

  python3 src/tweet_log.py posted vikings-1 2026-11-07T18:30 https://x.com/...      # mark as posted
  python3 src/tweet_log.py metrics vikings-1 24 impressions=812 likes=31 reposts=4 replies=6 bookmarks=3
  python3 src/tweet_log.py summary

Snapshots at 24 h and 168 h (7 days) keep tweets comparable. Engagement rate = (likes + reposts + replies + quotes
+ bookmarks) / impressions; with a small account, impressions mostly track follower count and luck, so the rate is
the fairer test of the wording. Once ~15 tweets have 24 h snapshots, summary reports the Spearman correlation
between classifier score and engagement rate, i.e. whether the Upworthy-trained ranker transfers to this account.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "tweets"
POSTS, METRICS = DIR / "posts.csv", DIR / "metrics.csv"
COUNTS = ["impressions", "likes", "reposts", "replies", "quotes", "bookmarks", "profile_visits", "new_follows", "link_clicks"]


def read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def posted(tid: str, when: str, url: str = "") -> None:
    fields, rows = read(POSTS)
    row = next((r for r in rows if r["id"] == tid), None)
    if row is None:
        sys.exit(f"unknown id {tid}; add it to tweets/posts.csv first")
    row.update(status="posted", posted_at_utc=when, url=url or row["url"])
    write(POSTS, fields, rows)
    print(f"{tid} marked posted at {when}")


def metrics(tid: str, hours: str, kv: list[str]) -> None:
    fields, rows = read(METRICS)
    new = {"id": tid, "hours_since_post": hours, "measured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")}
    for item in kv:
        k, v = item.split("=", 1)
        if k not in COUNTS:
            sys.exit(f"unknown metric {k}; use {COUNTS}")
        new[k] = v
    rows = [r for r in rows if not (r["id"] == tid and r["hours_since_post"] == hours)] + [new]
    write(METRICS, fields, rows)
    print(f"logged {tid} @ {hours} h: " + ", ".join(f"{k}={new[k]}" for k in COUNTS if k in new))


def num(r: dict, k: str) -> float:
    try:
        return float(r.get(k) or 0)
    except ValueError:
        return 0.0


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        out = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")


def summary() -> None:
    _, posts = read(POSTS)
    _, mets = read(METRICS)
    print(f"{'id':<12} {'status':<8} {'score':>6} {'h':>4} {'impr':>7} {'eng':>5} {'rate':>6}  text")
    pairs = []
    for p in posts:
        snaps = sorted((m for m in mets if m["id"] == p["id"]), key=lambda m: float(m["hours_since_post"]))
        snap = next((m for m in snaps if m["hours_since_post"] == "24"), snaps[-1] if snaps else None)
        if snap:
            impr = num(snap, "impressions")
            eng = sum(num(snap, k) for k in ("likes", "reposts", "replies", "quotes", "bookmarks"))
            rate = eng / impr if impr else float("nan")
            if snap["hours_since_post"] == "24" and impr:
                pairs.append((float(p["score_z"]), rate))
            print(f"{p['id']:<12} {p['status']:<8} {p['score_z']:>6} {snap['hours_since_post']:>4} {impr:>7.0f} {eng:>5.0f} {rate:>6.1%}  {p['text'][:60]}")
        else:
            print(f"{p['id']:<12} {p['status']:<8} {p['score_z']:>6} {'-':>4} {'-':>7} {'-':>5} {'-':>6}  {p['text'][:60]}")
    if len(pairs) >= 5:
        rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
        print(f"\nclassifier score vs 24 h engagement rate: Spearman {rho:+.2f} over {len(pairs)} tweets"
              + ("" if len(pairs) >= 15 else " (too few tweets to read anything into yet; aim for 15+)"))
    else:
        print(f"\n{len(pairs)} tweets with 24 h numbers; the score-vs-performance check starts at 5 and means something at 15+.")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["summary"]
    {"posted": lambda: posted(*rest), "metrics": lambda: metrics(rest[0], rest[1], rest[2:]), "summary": summary}[cmd]()
