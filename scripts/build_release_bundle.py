"""Build the public core application bundle without model or user data."""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [
    ".env.example",
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "README.md",
    "DATA_POLICY.md",
    "RELEASE_GUIDE.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    "app.py",
    "main.py",
    "database.py",
    "dataset.py",
    "engine.py",
    "model_package.py",
    "models.py",
    "segmentation.py",
    "run_haworthia.py",
    "install_haworthia.bat",
    "start_haworthia.bat",
    "MODEL_PACKAGE_FORMAT.md",
    "scripts/check_setup.py",
    "scripts/download_segmentation_models.py",
    "scripts/audit_release.py",
    "scripts/build_release_bundle.py",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_public_files(destination):
    for relative_name in PUBLIC_FILES:
        source = ROOT / relative_name
        if not source.is_file():
            raise FileNotFoundError(f"Missing public file: {source}")
        target = destination / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def build_archive(staging_root, archive_path):
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(staging_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging_root.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = f"haworthia-omics-v{args.version}-core"
    archive_path = output_dir / f"{bundle_name}.zip"

    with tempfile.TemporaryDirectory(prefix="haworthia_bundle_") as temporary:
        staging_root = Path(temporary) / bundle_name
        staging_root.mkdir()
        copy_public_files(staging_root)
        bundle_manifest = {
            "bundle_version": args.version,
            "bundle_type": "core_application_without_model",
            "release_ready": True,
            "license": "Apache-2.0",
            "model_included": False,
            "prototype_catalog_included": False,
            "images_included": False,
            "training_checkpoint_included": False,
        }
        (staging_root / "BUNDLE_MANIFEST.json").write_text(
            json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if archive_path.exists():
            archive_path.unlink()
        build_archive(staging_root, archive_path)

    archive_hash = sha256(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(
        f"{archive_hash}  {archive_path.name}\n", encoding="ascii"
    )
    print(f"Bundle: {archive_path}")
    print(f"SHA-256: {archive_hash}")
    print(f"Release ready: {bundle_manifest['release_ready']}")


if __name__ == "__main__":
    main()
