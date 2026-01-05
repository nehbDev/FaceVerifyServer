FROM python:3.10-slim

# Install system dependencies for OpenCV and DeepFace 
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies [cite: 3]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Expose port and start gunicorn 
# "app:app" refers to the 'app' object inside 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers=1", "--timeout=120", "app:app"]