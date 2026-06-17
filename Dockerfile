# Use a lightweight official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOST=0.0.0.0

# Set the working directory in the container
WORKDIR /app

# Install compiler dependencies (needed for certain C-extensions if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Set working directory to backend so python finds the app package
WORKDIR /app/backend

# Expose port 7860
EXPOSE 7860

# Command to run the application using factory pattern
CMD ["sh", "-c", "uvicorn app:create_app --factory --host $HOST --port $PORT"]
