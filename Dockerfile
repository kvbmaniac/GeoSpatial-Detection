# Use Python base image
FROM python:3.10-slim

# Install system-level dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your project
COPY . .

# Expose Flask port
EXPOSE 5000

# Run your app
CMD ["python", "app.py"]
