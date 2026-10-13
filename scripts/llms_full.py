"""Prints llms-full.txt: llms.txt, then the guides, the references and an example as one page of
Markdown, with relative links made absolute so they work wherever the page is read.

    python3 scripts/llms_full.py > site/llms-full.txt
"""

import posixpath
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOB = "https://github.com/tanvincible/chassis/blob/main/"
PAGES = [
    "README.md",
    "docs/src/guide/getting-started.md",
    "docs/src/guide/basic-usage.md",
    "chassisdb/README.md",
    "docs/src/guide/api-reference.md",
    "docs/src/guide/c-api.md",
    "docs/src/guide/errors.md",
    "docs/src/guide/agents.md",
]
EXAMPLE = "chassisdb/examples/notes_search.py"


def absolute(text: str, page: str) -> str:
    base = posixpath.dirname(page)

    def fix(link: re.Match) -> str:
        target = link.group(2)
        if re.match(r"[a-z]+:|#|/", target):
            return link.group(0)
        path, hash, anchor = target.partition("#")
        return f"{link.group(1)}({BLOB}{posixpath.normpath(posixpath.join(base, path))}{hash}{anchor})"

    return re.sub(r"(\])\(([^)\s]+)\)", fix, text)


def main() -> None:
    parts = [(ROOT / "llms.txt").read_text().rstrip()]
    for page in PAGES:
        parts.append(f"<!-- {page} -->\n\n{absolute((ROOT / page).read_text().rstrip(), page)}")
    parts.append(f"<!-- {EXAMPLE} -->\n\n```python\n{(ROOT / EXAMPLE).read_text().rstrip()}\n```")
    print("\n\n---\n\n".join(parts))


if __name__ == "__main__":
    main()
