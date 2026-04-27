FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r deployable-app/backend/requirements.txt

EXPOSE 7860

CMD ["sh", "-c", "gunicorn --chdir deployable-app/backend app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 900"]
