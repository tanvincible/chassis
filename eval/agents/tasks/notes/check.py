import subprocess
import sys

NOTES = [
    "Buy oat milk and bread on the way home",
    "The quarterly report is due on Friday",
    "Call the dentist to move the Tuesday appointment",
    "Book flights for the conference in Lisbon",
    "Draft the report summary and send it to finance",
    "Water the plants on the balcony",
]


def run(*args):
    done = subprocess.run([sys.executable, "notes.py", *args], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, f"notes.py {' '.join(args)} failed:\n{done.stderr[-2000:]}"
    return done.stdout


ids = [int(run("add", note).strip().split()[-1]) for note in NOTES]
assert len(set(ids)) == len(ids), f"ids repeat: {ids}"
top = run("search", "when is the quarterly report due", "-k", "2").strip().splitlines()
assert top and top[0].split("\t")[0] == str(ids[1]), f"expected note {ids[1]} first, got {top}"
assert run("search", "plants").splitlines()[0].split("\t")[0] == str(ids[5])
run("delete", str(ids[1]))
after = run("search", "quarterly report due friday", "-k", "6")
assert str(ids[1]) not in [line.split("\t")[0] for line in after.splitlines()], after
print("ok")
