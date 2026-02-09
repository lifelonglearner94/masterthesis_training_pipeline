#!/usr/bin/env python3
"""Prepend a YYYYMMDD creation-date timestamp to Markdown files in this folder.

The creation date is determined from git history (date of the first commit that
introduced the file).  Falls back to today's date for untracked files.

Usage:
  python rename_md_with_timestamp.py          # normal run
  python rename_md_with_timestamp.py --fix    # strip wrong prefix first, then re-apply
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
import re

PREFIX_RE = re.compile(r"^(\d{8})_")


def git_creation_date(filepath: Path) -> str:
    """Return YYYYMMDD of the first commit that introduced *filepath*.

    Tries the current name first; if the file was renamed, uses
    ``git log --follow --diff-filter=A`` to track back to the original add.
    Falls back to today's date if the file has no git history (untracked).
    """
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--follow",
                "--diff-filter=A",
                "--format=%ai",
                "--",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            cwd=filepath.parent,
        )
        lines = result.stdout.strip().splitlines()
        if lines:
            # last line = earliest (first) commit that added the file
            date_str = lines[-1].split()[0]  # "YYYY-MM-DD"
            return date_str.replace("-", "")
    except FileNotFoundError:
        pass
    # fallback
    return datetime.now().strftime("%Y%m%d")


def strip_prefix(docs_dir: Path) -> None:
    """Remove an existing YYYYMMDD_ prefix from every .md file."""
    for md_file in sorted(docs_dir.glob("*.md")):
        m = PREFIX_RE.match(md_file.name)
        if m:
            original_name = md_file.name[len(m.group(0)) :]
            md_file.rename(md_file.with_name(original_name))
            print(f"  stripped: {md_file.name} -> {original_name}")


def add_prefix(docs_dir: Path) -> None:
    """Add YYYYMMDD_ prefix (from git creation date) to every .md file."""
    for md_file in sorted(docs_dir.glob("*.md")):
        if PREFIX_RE.match(md_file.name):
            print(f"  skipped (already prefixed): {md_file.name}")
            continue
        timestamp = git_creation_date(md_file)
        new_name = f"{timestamp}_{md_file.name}"
        md_file.rename(md_file.with_name(new_name))
        print(f"  renamed: {md_file.name} -> {new_name}")


def main() -> None:
    docs_dir = Path(__file__).resolve().parent
    fix_mode = "--fix" in sys.argv

    if fix_mode:
        print("== Stripping existing timestamps ==")
        strip_prefix(docs_dir)

    print("== Adding creation-date timestamps ==")
    add_prefix(docs_dir)
    print("Done.")


if __name__ == "__main__":
    main()
