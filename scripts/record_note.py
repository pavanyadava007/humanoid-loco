"""Append a failure / deviation note to results/failures.json (so the report can list it).

Usage: python scripts/record_note.py --what "..." --error "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hloco.common import RESULTS, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True)
    ap.add_argument("--error", required=True)
    args = ap.parse_args()
    p = RESULTS / "failures.json"
    items = json.loads(p.read_text())["items"] if p.exists() else []
    items.append({"what": args.what, "error": args.error})
    write_json(p, {"items": items})


if __name__ == "__main__":
    main()
