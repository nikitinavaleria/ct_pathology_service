FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Amsterdam

WORKDIR /app

# + git для зависимостей из репо, + libgl1 для cv2, + libglib2.0-0 для загрузчиков
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Ставим зависимости
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Кладём исходники (сохраняем путь backend.app.main:app)
COPY backend /app/backend

EXPOSE 8000
ENV API_PREFIX=/api MAX_UPLOAD_MB=200

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

