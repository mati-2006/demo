<<<<<<< HEAD
=======
# Signature Detect & Verify — Demo

Local Gradio demo that:

1. **Detects** signatures on a document (Mels22 YOLO11s — `detector_yolo_1cls.pt`)
2. **Verifies** a detected crop against a specimen signature (Mels22 Siamese verifier)

## Setup

```bash
cd signature
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the demo

```bash
python app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

### Demo flow

1. Tab **Detect** — upload a document, click **Detect signatures**
2. Tab **Verify** — pick a detected crop, add a specimen via **Draw** (600×200 pad), **Upload**, or **Clipboard**, click **Verify**

First run downloads models from Hugging Face (needs internet).
>>>>>>> b8357fe (Initial commit)
