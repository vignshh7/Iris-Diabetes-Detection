# Deployable Full Stack App (Frontend + Backend)

This folder contains a split deployment setup:

- `frontend/` - static website for uploads and results UI
- `backend/` - Flask API that runs your existing prediction pipeline
- `render.yaml` - legacy Render Blueprint, kept for reference only

## Backend

The backend reuses your existing model/pipeline from the repository root:

- uploads files
- creates isolated job workspace under `temp/webapp_runs/<jobId>/`
- runs `src.predict_realdata.predict_realdata()`
- streams progress and returns results/CSV

### Local run

From repository root:

```bash
pip install -r deployable-app/backend/requirements.txt
python deployable-app/backend/app.py
```

Backend URL: `http://127.0.0.1:5000`

## Frontend

### Local run

Use any static server from `deployable-app/frontend`.

Set API base in `frontend/config.js`:

```js
window.__API_BASE__ = "http://127.0.0.1:5000";
```

If empty (`""`), frontend assumes same-origin API.

## Recommended Free Deployment

Use this stack:

- Frontend: Vercel static site
- Backend: Hugging Face Spaces Docker app

### 1) Deploy backend to Hugging Face Spaces

1. Create a new Space on Hugging Face.
2. Choose `Docker` as the SDK.
3. Use the repository root as the Space repo so it includes `src/`, `models/`, `config.py`, and `requirements.txt`.
4. Make sure the repo-root `Dockerfile` exists.
5. Let Spaces build and start the API.

### 2) Deploy frontend to Vercel

1. Create a new Vercel project from this GitHub repo.
2. Set the root directory to `deployable-app/frontend`.
3. Set `API_BASE_URL` in Vercel Environment Variables to your Hugging Face backend URL.
4. Deploy the static site.

### 3) Production CORS

Set the backend environment variable:

```text
CORS_ORIGINS=https://your-vercel-app.vercel.app
```

If you are testing, `*` works, but a specific origin is better.

## Notes

- Your model files in `models/` are required in the same repository.
- Prediction can take time; timeout is configured for long-running jobs.
- Supported upload file types: `.jpg`, `.jpeg`.
- The frontend build step writes `config.js` from `API_BASE_URL` before Vercel serves the site.
- The repo-root Dockerfile is designed for Spaces and runs Gunicorn on the platform port.
