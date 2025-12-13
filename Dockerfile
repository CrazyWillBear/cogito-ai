# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Prevents Python from writing .pyc files and enables unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory to project root
WORKDIR /app

# Install system deps if needed (add packages here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app

# If your server listens on a port, expose it (adjust as needed)
EXPOSE 50051

# Run the server from project root
CMD ["python", "main_server.py"]
