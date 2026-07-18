# Release Guide

## Public Contents

The public Git repository and GitHub Release contain the Apache-2.0 application and
documentation only. Do not publish any model, model manifest, prototype database, user
database, image, checkpoint, backup, or third-party ONNX weight.

## Local Git Setup

```powershell
git init -b main
git config user.name "雨筠"
git config user.email "142278530+Yujun8Q@users.noreply.github.com"
git add .
git status --short
python scripts/audit_release.py
git commit -m "Initial open-source release"
```

Create an empty public repository named `haworthia-omics` under `Yujun8Q`, then push:

```powershell
git remote add origin https://github.com/Yujun8Q/haworthia-omics.git
git push -u origin main
```

## Build Core Application

```powershell
python scripts/build_release_bundle.py --version 0.1.0
```

The builder creates `dist/haworthia-omics-v0.1.0-core.zip` and its checksum. The embedded
`BUNDLE_MANIFEST.json` must report Apache-2.0, `release_ready: true`, and false for model,
prototype catalog, images, and training checkpoint inclusion.

## GitHub Release

```powershell
git tag -a v0.1.0 -m "Haworthia OMICS v0.1.0"
git push origin v0.1.0
```

Create a GitHub Release from the tag and upload only the core ZIP and matching `.sha256`.
The automatic GitHub source archives are expected because this is an open-source repository.
Never upload anything under `dist/private/`, `local_images/`, `backups/`, or any `.pth`,
`.db`, `.onnx`, or checkpoint file.
