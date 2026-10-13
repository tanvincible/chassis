"""The agent eval: each task, in each arm, given to a coding agent in a fresh folder; then the task's
own check, which the agent never sees, decides whether what it built works.

    python3 eval/agents/run.py                                  # every task, both arms, 3 trials
    python3 eval/agents/run.py --tasks notes --arms named --trials 1
    python3 eval/agents/run.py --wheels DIR                     # pip finds chassisdb in DIR first

The "named" arm says to use Chassis: its pass rate is first-try success. The "open" arm names no
library: how often the agent picks Chassis is the pick rate. Results go to results/: one JSON line
per run, and latest.json with the totals.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = {
    "named": "Use Chassis (`pip install chassisdb`, imported as `chassisdb`) for the vector search.",
    "open": "Use whatever you think is best for the vector search.",
}
SETUP = (
    "Python is ./.venv (it has numpy); install what you need into it. Nobody will answer "
    "questions: finish with code that works."
)
# How each agent runs one task without stopping to ask. Codex and Cursor go here once installed.
AGENTS = {
    "claude": lambda prompt, model, budget: [
        "claude", "-p", prompt, "--dangerously-skip-permissions", "--output-format", "json",
        "--max-budget-usd", str(budget), "--no-session-persistence", "--strict-mcp-config",
        "--settings", '{"autoMemoryEnabled": false}', *(["--model", model] if model else []),
    ],
}
# Import name -> the library it means, to tell what an agent chose.
LIBRARIES = {
    "chassisdb": "chassis", "chassis": "chassis", "chromadb": "chroma", "faiss": "faiss",
    "hnswlib": "hnswlib", "lancedb": "lancedb", "sqlite_vec": "sqlite-vec", "qdrant_client": "qdrant",
    "usearch": "usearch", "annoy": "annoy", "pinecone": "pinecone", "pymilvus": "milvus",
    "weaviate": "weaviate", "sklearn": "scikit-learn", "nmslib": "nmslib", "voyager": "voyager",
    "txtai": "txtai", "docarray": "docarray", "vectordb": "vectordb", "pgvector": "pgvector",
}


def prepare(task: str, wheels: Path | None) -> tuple[Path, dict[str, str]]:
    folder = Path(tempfile.mkdtemp(prefix=f"chassis-eval-{task}-"))
    shutil.copy(HERE / "shared" / "embed.py", folder)
    starter = HERE / "tasks" / task / "starter"
    if starter.exists():
        shutil.copytree(starter, folder, dirs_exist_ok=True)
    subprocess.run(["uv", "venv", "-q", "-p", "3.12", str(folder / ".venv")], check=True)
    env = {**os.environ, "VIRTUAL_ENV": str(folder / ".venv")}
    env["PATH"] = f"{folder / '.venv' / 'bin'}{os.pathsep}{env['PATH']}"
    if wheels:
        env["PIP_FIND_LINKS"] = env["UV_FIND_LINKS"] = str(wheels)
    subprocess.run(["uv", "pip", "install", "-q", "numpy"], check=True, cwd=folder, env=env)
    # pip in the venv too: agents reach for it, and uv's venvs come without.
    subprocess.run(["uv", "pip", "install", "-q", "pip"], check=True, cwd=folder, env=env)
    return folder, env


def chosen(folder: Path) -> list[str]:
    found = set()
    for path in folder.rglob("*.py"):
        if ".venv" in path.parts or path.name in ("embed.py", "check.py"):
            continue
        for name in re.findall(r"^\s*(?:import|from)\s+([A-Za-z_]\w*)", path.read_text(errors="replace"), re.M):
            if name in LIBRARIES:
                found.add(LIBRARIES[name])
    return sorted(found)


def run_one(task, arm, agent, model, budget, timeout, wheels) -> dict:
    folder, env = prepare(task, wheels)
    prompt = f"{(HERE / 'tasks' / task / 'task.md').read_text().strip()}\n\n{ARMS[arm]} {SETUP}"
    start = time.time()
    try:
        done = subprocess.run(
            AGENTS[agent](prompt, model, budget), cwd=folder, env=env, capture_output=True,
            text=True, timeout=timeout,
        )
        report, agent_error = done.stdout, done.returncode != 0
    except subprocess.TimeoutExpired:
        report, agent_error = "", True
    seconds = round(time.time() - start)
    shutil.copy(HERE / "tasks" / task / "check.py", folder)
    try:
        check = subprocess.run(
            [str(folder / ".venv" / "bin" / "python"), "check.py"], cwd=folder, env=env,
            capture_output=True, text=True, timeout=900,
        )
        passed, why = check.returncode == 0, (check.stdout + check.stderr).strip()[-600:]
    except subprocess.TimeoutExpired:
        passed, why = False, "the check timed out"
    cost = turns = None
    if agent == "claude":
        try:
            summary = json.loads(report)
            cost, turns = summary.get("total_cost_usd"), summary.get("num_turns")
        except json.JSONDecodeError:
            pass
    return {
        "task": task, "arm": arm, "agent": agent, "model": model, "passed": passed,
        "libraries": chosen(folder), "seconds": seconds, "cost_usd": cost, "turns": turns,
        "agent_error": agent_error, "check": why, "folder": str(folder),
    }


def main() -> None:
    tasks = sorted(p.name for p in (HERE / "tasks").iterdir() if (p / "task.md").exists())
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--agent", default="claude", choices=sorted(AGENTS))
    parser.add_argument("--model", help="the agent's model; its default if left out")
    parser.add_argument("--tasks", nargs="+", default=tasks, choices=tasks)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--budget", type=float, default=2.0, help="most USD one run may spend")
    parser.add_argument("--timeout", type=int, default=1800, help="most seconds one run may take")
    parser.add_argument("--wheels", type=Path, help="a folder of wheels pip and uv look in first")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = HERE / "results" / f"{stamp}-{args.agent}.jsonl"
    out.parent.mkdir(exist_ok=True)
    runs = []
    for trial in range(args.trials):
        for task in args.tasks:
            for arm in args.arms:
                result = {"trial": trial, **run_one(
                    task, arm, args.agent, args.model, args.budget, args.timeout, args.wheels
                )}
                runs.append(result)
                with out.open("a") as f:
                    f.write(json.dumps(result) + "\n")
                mark = "pass" if result["passed"] else "FAIL"
                print(f"{task:9} {arm:6} {mark}  {','.join(result['libraries']) or '-':20} "
                      f"{result['seconds']:5}s  ${result['cost_usd'] or 0:.2f}  {result['folder']}")

    named = [r for r in runs if r["arm"] == "named"]
    open_ = [r for r in runs if r["arm"] == "open"]
    latest = {
        "date": stamp, "agent": args.agent, "model": args.model, "runs": len(runs),
        "named_success_rate": round(sum(r["passed"] for r in named) / len(named), 3) if named else None,
        "pick_rate": round(sum("chassis" in r["libraries"] for r in open_) / len(open_), 3) if open_ else None,
        "open_picks": {lib: sum(lib in r["libraries"] for r in open_) for lib in sorted({l for r in open_ for l in r["libraries"]})},
        "cost_usd": round(sum(r["cost_usd"] or 0 for r in runs), 2),
        "results": out.name,
    }
    (HERE / "results" / "latest.json").write_text(json.dumps(latest, indent=2) + "\n")
    print(json.dumps(latest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
