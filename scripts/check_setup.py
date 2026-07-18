"""Check the local runtime without reading or changing user images."""

import importlib.util
import os
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "fastapi",
    "matplotlib",
    "multipart",
    "networkx",
    "numpy",
    "onnxruntime",
    "pandas",
    "PIL",
    "pydantic",
    "requests",
    "scipy",
    "sklearn",
    "streamlit",
    "torch",
    "torchvision",
    "uvicorn",
]


def main():
    missing_modules = [
        name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None
    ]
    model_dir = Path(os.getenv("U2NET_HOME", "~/.u2net")).expanduser()
    missing_segmentation = [
        name
        for name in ("isnet-general-use.onnx", "u2net.onnx")
        if not (model_dir / name).is_file()
    ]
    trained_model = Path(
        os.getenv("HAWORTHIA_MODEL_PATH", "model_base.pth")
    ).expanduser()

    print(f"Python: {sys.version.split()[0]}")
    print(f"Dependencies: {'OK' if not missing_modules else 'missing ' + ', '.join(missing_modules)}")
    print(
        "Segmentation: "
        + ("OK" if not missing_segmentation else "missing " + ", ".join(missing_segmentation))
    )
    model_status = "not trained yet"
    if trained_model.is_file():
        model_status = "available local model"
    print(f"Trained phenotype model: {model_status}")

    if missing_modules or missing_segmentation:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
