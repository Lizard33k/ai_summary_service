# -----------------------------
# Base Python image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Environment variables
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# Install system-level dependencies
# -----------------------------

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy Python dependencies
# -----------------------------
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy app and utils folders
# -----------------------------
COPY ./app /app/app
COPY ./utils /app/utils
COPY ./secret_keys /app/secret_keys
# -----------------------------
# Expose port for FastAPI
# -----------------------------
EXPOSE 8828

# -----------------------------
# Run the FastAPI server
# -----------------------------
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8828", "--reload"]
