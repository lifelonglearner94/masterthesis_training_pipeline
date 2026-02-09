#!/usr/bin/env python3
"""Prepend a YYYYMMDD timestamp to Markdown files in this folder.

Usage:
  python rename_md_with_timestamp.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def main() -> None:
    docs_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d")
    prefix_re = re.compile(r"^\d{8}_")

    for md_file in docs_dir.glob("*.md"):
        if prefix_re.match(md_file.name):
            continue
        new_name = f"{timestamp}_{md_file.name}"
        md_file.rename(md_file.with_name(new_name))


if __name__ == "__main__":
    main()
