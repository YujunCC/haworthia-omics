"""Fail when Git tracks private data, model assets, or likely secrets."""

import re
import subprocess
import sys
from pathlib import Path


FORBIDDEN_DIRECTORIES = {
    "local_images",
    "backups",
    "dist",
    ".agents",
    ".idea",
    "__pycache__",
}
FORBIDDEN_FILENAMES = {"apppp.py"}
FORBIDDEN_SUFFIXES = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}
MAX_FILE_SIZE = 25 * 1024 * 1024
TEXT_SUFFIXES = {
    "",
    ".bat",
    ".example",
    ".gitignore",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_PATTERNS = {
    "workspace path": re.compile(
        r"(?:[A-Za-z]:[\\/](?:Users[\\/](?!(?:your-name|USERNAME|user)[\\/])"
        r"[^\\/]+|PycharmProjects)[\\/])",
        re.IGNORECASE,
    ),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
    "credential assignment": re.compile(
        r"(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{8,}",
        re.IGNORECASE,
    ),
}


def tracked_files():
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print("Release audit requires an initialized Git repository.", file=sys.stderr)
        raise SystemExit(2)
    return [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]


def main():
    failures = []
    files = tracked_files()
    for path in files:
        lowered_parts = {part.lower() for part in path.parts}
        if lowered_parts & FORBIDDEN_DIRECTORIES:
            failures.append(f"forbidden directory: {path}")
        if path.name.lower() in FORBIDDEN_FILENAMES:
            failures.append(f"legacy file: {path}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden data/model extension: {path}")
        if path.name == ".env":
            failures.append(f"local environment file: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_FILE_SIZE:
            failures.append(f"file exceeds 25 MB: {path} ({size} bytes)")
        if size <= 2 * 1024 * 1024 and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in PRIVATE_PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{label}: {path}")

    if failures:
        print("Public release audit failed:", file=sys.stderr)
        for failure in sorted(set(failures)):
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Public release audit passed for {len(files)} tracked files.")


if __name__ == "__main__":
    main()
