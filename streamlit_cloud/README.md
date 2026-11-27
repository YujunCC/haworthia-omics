# Streamlit Community Cloud deployment

This directory contains the public, read-only hosted demo. It exposes inference and
numeric phenotype analysis only. It has no training, database maintenance, image
gallery, model import, or model export interface.

The app downloads `model_base.pth`, `catalog.db`, and `ASSET_MANIFEST.json` from a
private Hugging Face model repository at runtime. The catalog is sanitized and contains
no image records or image paths. Uploaded images are processed transiently and are not
added to the training database.

## Required Streamlit secrets

Configure the following in **App settings > Secrets**. Never commit a real token:

```toml
HF_MODEL_REPO_ID = "YujunCC/haworthia-omics-private-assets"
HF_TOKEN = "hf_..."
```

The token should be fine-grained, read-only, and limited to the one private model
repository. Select Python 3.12 when creating the app and use this main file:

```text
streamlit_cloud/streamlit_app.py
```

The first image request downloads and validates the two public segmentation models.
The app uses a low-memory mode that keeps at most one ONNX session resident at a time,
so this first request and later image requests may be slower on the free CPU instance.

Result images are sent as inline PNG previews over the existing Streamlit connection,
so they do not depend on temporary `/media/` URLs being reachable. Previews have a
maximum edge of 1600 pixels to bound the transfer size; this display limit does not
change segmentation, model inputs, similarity scores, or attention computation.
