# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify Tesseract installation and find the correct tessdata path
RUN tesseract --version && \
    find /usr/share -name "eng.traineddata" -type f | head -1 | xargs -I {} dirname {} > /tmp/tessdata_path.txt && \
    echo "Tessdata found at: $(cat /tmp/tessdata_path.txt)"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy diagnostic script first for debugging
COPY diagnose_tesseract.py .

# Run diagnostic to verify Tesseract setup (optional - can be removed in production)
RUN python diagnose_tesseract.py || true

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Dynamically set TESSDATA_PREFIX based on where the files actually are
# This approach works regardless of Tesseract version (4.x or 5.x)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5.00/tessdata

# Alternative: Set multiple possible paths and let Tesseract find the correct one
# ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata:/usr/share/tesseract-ocr/5.00/tessdata:/usr/share/tesseract-ocr/4.00/tessdata

# Create a startup script that verifies Tesseract before starting the app
RUN echo '#!/bin/bash\n\
# Find and export the correct TESSDATA_PREFIX\n\
for dir in /usr/share/tesseract-ocr/tessdata /usr/share/tesseract-ocr/*/tessdata /usr/share/tessdata; do\n\
    if [ -f "$dir/eng.traineddata" ]; then\n\
        export TESSDATA_PREFIX=$(dirname "$dir")\n\
        echo "Found tessdata at: $TESSDATA_PREFIX"\n\
        break\n\
    fi\n\
done\n\
\n\
# Verify Tesseract works\n\
tesseract --list-langs || echo "Warning: Tesseract language check failed"\n\
\n\
# Start the application\n\
exec gunicorn --bind 0.0.0.0:5000 app:app\n' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 5000

# Run the application using our startup script
CMD ["/app/start.sh"]