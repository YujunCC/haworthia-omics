# Third-Party Components

Haworthia OMICS is licensed under Apache-2.0. Its Python dependencies remain governed by
their own upstream licenses. The core ZIP does not bundle those packages; the user installs
the pinned versions from the configured Python package index using `requirements.txt`.

Major dependencies include PyTorch, torchvision, FastAPI, Streamlit, ONNX Runtime, NumPy,
SciPy, pandas, scikit-learn, Pillow, Matplotlib, NetworkX, Requests, Pydantic, Uvicorn, and
python-multipart. Before redistributing a compiled application that embeds these packages,
collect and ship the exact license and notice files from the installed versions.

The optional files `isnet-general-use.onnx` and `u2net.onnx` are downloaded separately from
the upstream rembg GitHub Release by `scripts/download_segmentation_models.py`. The script
pins their URLs and verifies SHA-256. These model assets are not included in this repository
or core release, and they are not licensed under this project's Apache-2.0 license. Users
must review their upstream terms before downloading or using them.

Upstream references:

- https://github.com/danielgatis/rembg
- https://github.com/xuebinqin/U-2-Net
- https://github.com/xuebinqin/DIS
