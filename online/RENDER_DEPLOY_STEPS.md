# Render Deployment Steps (Docker)

## 1. Push changes to GitHub
Make sure this folder exists in your repo:
- online/Dockerfile
- online/start.sh

## 2. Create Render Web Service
- New + -> Web Service
- Connect your GitHub repo
- Environment: Docker

## 3. Docker settings
- Dockerfile Path: online/Dockerfile
- Root Directory: leave empty (repository root)

Why: the Dockerfile copies the full repository (`COPY . /app`).
If Root Directory is set to `online`, app files like `webapp/` and `src/` will not be copied.

## 4. Instance and region
- Start with Standard or Starter plan
- Choose region closest to users

## 5. Environment variables
Add these in Render dashboard:

Required basics:
- PYTHONUNBUFFERED=1
- GUNICORN_WORKERS=1
- GUNICORN_THREADS=4
- GUNICORN_TIMEOUT=900

Optional (for auto model download):
- MODEL_BASE_URL=https://<your-model-storage-base-url>
- MODEL_FILES=best_f1_model_fold_1.pth,best_f1_model_fold_2.pth,best_f1_model_fold_3.pth,best_f1_model_fold_4.pth,best_f1_model_fold_5.pth,best_iris_model_3class.pth,best_iris_model_2class.pth

If model files are already committed in repo under /models, you can skip MODEL_BASE_URL.

## 6. Deploy
- Click Create Web Service
- Wait for build and deploy
- Open service URL

## 7. Verify quickly
- Open homepage
- Upload a small test set
- Run prediction and watch progress
- Download CSV
- Calculate measures

## 8. Common fixes
- "No module named src": ensure Root Directory is empty (repo root)
- OpenCV runtime errors: Dockerfile already installs libgl1 and libglib2.0-0
- Slow inference: CPU deployment is expected to be slower than local GPU
