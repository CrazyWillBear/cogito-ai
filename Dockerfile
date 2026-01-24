# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Prevents Python from writing .pyc files and enables unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory to project root
WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app

# Expose server listening port
EXPOSE 50051

# Run the server from project root
CMD ["python", "cogito_server.py"]
