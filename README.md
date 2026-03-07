# Diabetes Detection from Iris Images

Deep learning system that detects diabetes from paired iris images using a CNN ensemble with 5-fold cross-validation.

---

## What it does

Takes left and right iris image pairs and classifies patients as diabetic or control. The model processes each image through multiple color channels (RGB, Grayscale, HSV, LAB) with spatial attention masks — 40 input channels total — and averages predictions across 5 trained folds for the final result.

---

## Dataset

- 127 patients — 52 control, 75 diabetic
- 2 images per patient (left eye, right eye)
- Split: 60% train / 20% val / 20% test (patient-level, no leakage)
- Images resized to 128×128

---

## Project Structure

```
eye_project/
├── run_eye_project.bat
├── config.py
├── requirements.txt
├── data_split_info.json
├── dataset/
│   ├── data/
│   │   ├── control/          # patients 1–52
│   │   └── diabetic/         # patients 53–127
│   ├── masks/
│   └── pancreatic_masks/
├── dataset_backup/
├── realdata/
│   ├── images/
│   ├── masks/
│   └── pancreatic_masks/
├── models/
│   ├── best_iris_model_3class.pth
│   ├── best_f1_model_fold_1.pth
│   └── ... (fold 2–5)
└── src/
    ├── cnntrain.py
    ├── cnnpredict.py
    ├── metrices.py
    ├── predict_realdata.py
    ├── predict_realdata_interactive.py
    ├── process_new_dataset.py
    ├── generate_masks.py
    ├── data_manager.py
    ├── visualize_results.py
    ├── gradcam_generate.py
    └── gradcam_montage.py
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Or just double-click `run_eye_project.bat` — it handles everything.

---

## Workflow

**1. Process dataset**
Put raw images in `dataset_backup/data/control/` and `/diabetic/`. Name them `XL.*.jpg` / `XR.*.jpg`. Run option 1 — it applies sequential numbering and handles unpaired files.

**2. Train**
Option 2 generates masks, splits data, and trains 5 models. Checkpoints saved to `models/`.

**3. Evaluate**
Option 3 runs ensemble prediction on the held-out test set using `data_split_info.json`. Outputs `evaluation_results.csv`.

**4. Predict on new data**
Drop images in `realdata/images/` and run option 4. No ground truth needed. Results in `realdata_predictions.csv`.

---

## Model

```
40-channel input (20 per eye)
→ Conv blocks with SE attention and GroupNorm
→ Global average pooling
→ FC 512→256→1
```

Trained with Focal Loss (handles class imbalance), early stopping (patience=8), tuned threshold per fold.

---

## Config

Key parameters in `config.py`:

```python
IMAGE_SIZE = 128
N_FOLDS = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 8
```

---

## Common Issues

- **No masks during training** — run option 1 first; option 2 auto-generates them
- **No images found in realdata** — place directly in `realdata/images/`, no subfolders
- **CUDA OOM** — lower `BATCH_SIZE` in config or force `device='cpu'`
- **Orphaned files warning** — each patient needs both L and R images

---

## Notes

Images are gitignored (`.jpg`, `.jpeg`, `.png`). Folder structure is preserved with `.gitkeep` files. Always use `data_split_info.json` for evaluation — never re-split after training.

---

Python 3.8+ · PyTorch 2.0+ · March 2026
