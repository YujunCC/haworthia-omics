"""Download segmentation weights from the upstream rembg release."""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path


MODELS = {
    "isnet-general-use.onnx": {
        "url": (
            "https://github.com/danielgatis/rembg/releases/download/"
            "v0.0.0/isnet-general-use.onnx"
        ),
        "sha256": "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
    },
    "u2net.onnx": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        "sha256": "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
    },
}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name, metadata, destination):
    target = destination / name
    if target.exists() and file_hash(target) == metadata["sha256"]:
        print(f"Already verified: {target}")
        return

    temporary = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {name} from the upstream rembg release...")
    try:
        urllib.request.urlretrieve(metadata["url"], temporary)
        actual = file_hash(temporary)
        if actual != metadata["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: expected {metadata['sha256']}, got {actual}"
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Installed: {target}")


def main():
    default_dir = Path(os.getenv("U2NET_HOME", "~/.u2net")).expanduser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=default_dir)
    args = parser.parse_args()
    destination = args.model_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    print("The downloaded files are third-party model assets, not part of this project license.")
    for name, metadata in MODELS.items():
        download(name, metadata, destination)
    print("Segmentation model setup is complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
