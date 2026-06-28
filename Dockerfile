FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first so this layer is cached independently of
# source-code changes. Re-running pip only happens when requirements.txt
# itself changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and pre-trained model artifacts.
# models/ is listed in .gitignore (weights are not version-controlled)
# but intentionally absent from .dockerignore — the image ships with a
# ready-to-serve checkpoint so the container starts without training.
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
