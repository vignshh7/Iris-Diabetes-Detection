# Deployable Full Stack App (Frontend + Backend)

This folder contains a split deployment setup:

- `frontend/` - static website for uploads and results UI
- `backend/` - Flask API that runs your existing prediction pipeline
- `render.yaml` - Render Blueprint to deploy both services

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

## Render Deployment

1. Push this repository to GitHub.
2. In Render, create Blueprint and select `deployable-app/render.yaml`.
3. Deploy backend service first.
4. Copy backend public URL.
5. In static frontend service env vars, set `API_BASE_URL` to backend URL.
   Example: `https://iris-diabetes-backend.onrender.com`
6. Optional but recommended: set backend `CORS_ORIGINS` to the frontend URL instead of `*`.

## Notes

- Your model files in `models/` are required in the same repository.
- Prediction can take time; timeout is configured for long-running jobs.
- Supported upload file types: `.jpg`, `.jpeg`.
