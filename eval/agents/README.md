# Agent eval

How well coding agents build with Chassis, and how often they choose it. A fixed set of tasks
goes to an agent in a fresh folder; then each task's own check, which the agent never sees,
decides whether what it built works.

| Task | The agent builds | The check |
| --- | --- | --- |
| `notes` | a notes command line: add, search by meaning, delete; kept between runs | top results for known queries; deleted notes stay gone |
| `similar` | "more like this" over 200 products, indexed once | 5 nearest by brute force, ties counted; never the product itself; under 10 s a call |
| `retrieve` | RAG retrieval over Markdown paragraphs, reindexed when files change | the right paragraph first; no stale paragraphs after an edit |
| `dedupe` | near-duplicate groups among 500 support tickets, without comparing every pair | 90% of the exact groups |

Every task gives the agent the same stand-in embedding model (`shared/embed.py`), so the eval
measures the vector search, not a model download. Each runs in two arms:

- **named**: the prompt says to use Chassis. Its pass rate is **first-try success**: one session,
  no help.
- **open**: the prompt names no library. How often the agent picks Chassis is the **pick rate**,
  which `scripts/adoption.py` tracks week by week.

## Running it

```sh
python3 eval/agents/run.py                                  # every task, both arms, 3 trials
python3 eval/agents/run.py --tasks notes --arms named --trials 1
python3 eval/agents/run.py --model claude-sonnet-5-5 --budget 1
```

It needs `uv`, and the agent's CLI logged in. Each run spends at most `--budget` dollars (default
2) and `--timeout` seconds (default 1800). Before a release is on PyPI, `--wheels DIR` makes pip
and uv find `chassisdb` in `DIR` first.

Agents: Claude Code (`claude`). Codex and Cursor go in `AGENTS` in `run.py`, one line each, once
their CLIs are installed.

Results go to `results/`: one JSON line per run (task, arm, pass, the libraries the agent's code
imports, time, cost, why the check failed, the folder it worked in), and `latest.json` with the
totals. `make_data.py` wrote the tasks' fixed data; it isn't run again.
