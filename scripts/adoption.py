"""How many people use Chassis, as one CSV row: installs, not stars.

    python3 scripts/adoption.py            # a summary
    python3 scripts/adoption.py --csv      # the CSV row, as the weekly workflow appends it

PyPI counts leave out mirrors (pypistats' without_mirrors); crates.io can't, so its counts include
mirrors and bots, which are most of a new crate's downloads. "Used by" is GitHub's dependents page.
"""

import argparse
import json
import re
import urllib.request
from datetime import date, timedelta

REPO = "tanvincible/chassis"
PYPI = ["chassisdb", "chassisdb-mcp", "langchain-chassisdb"]
CRATES = ["chassisdb", "chassis-core"]  # chassis-core is the crate's name before v0.8.0
FIELDS = [
    "date",
    *(f"pypi_{name}_week" for name in PYPI),
    *(f"pypi_{name}_total" for name in PYPI),
    *(f"crates_{name}_90_days" for name in CRATES),
    *(f"crates_{name}_total" for name in CRATES),
    "github_used_by_repositories",
    "github_used_by_packages",
    "github_stars",
]


def fetch(url: str) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": f"{REPO} adoption tracking"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:  # Not published yet
            return None
        raise


def pypi(name: str) -> tuple[int, int]:
    """Downloads without mirrors: the last 7 days, and every day pypistats keeps (180)."""
    body = fetch(f"https://pypistats.org/api/packages/{name}/overall?mirrors=false")
    if body is None:
        return 0, 0
    days = {row["date"]: row["downloads"] for row in json.loads(body)["data"]}
    week = {(date.today() - timedelta(days=n)).isoformat() for n in range(1, 8)}
    return sum(n for day, n in days.items() if day in week), sum(days.values())


def crates(name: str) -> tuple[int, int]:
    body = fetch(f"https://crates.io/api/v1/crates/{name}")
    if body is None:
        return 0, 0
    crate = json.loads(body)["crate"]
    return crate["recent_downloads"] or 0, crate["downloads"]


def used_by() -> tuple[int, int]:
    """GitHub's "Used by": the repositories and packages that depend on this one."""
    page = fetch(f"https://github.com/{REPO}/network/dependents").decode()
    counts = []
    for kind in ("Repositor", "Package"):
        found = re.search(rf"([\d,]+)\s*{kind}", page)
        counts.append(int(found.group(1).replace(",", "")) if found else 0)
    return counts[0], counts[1]


def stars() -> int:
    return json.loads(fetch(f"https://api.github.com/repos/{REPO}"))["stargazers_count"]


def row() -> dict[str, object]:
    values: dict[str, object] = {"date": date.today().isoformat()}
    for name in PYPI:
        values[f"pypi_{name}_week"], values[f"pypi_{name}_total"] = pypi(name)
    for name in CRATES:
        values[f"crates_{name}_90_days"], values[f"crates_{name}_total"] = crates(name)
    values["github_used_by_repositories"], values["github_used_by_packages"] = used_by()
    values["github_stars"] = stars()
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", action="store_true", help="print the CSV row")
    parser.add_argument("--header", action="store_true", help="print the CSV header first")
    args = parser.parse_args()
    values = row()
    if args.csv:
        if args.header:
            print(",".join(FIELDS))
        print(",".join(str(values[field]) for field in FIELDS))
        return
    width = max(map(len, FIELDS))
    for field in FIELDS:
        print(f"{field:<{width}}  {values[field]}")


if __name__ == "__main__":
    main()
