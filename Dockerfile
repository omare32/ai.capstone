# AAVAIL Revenue Prediction API - Lightweight Docker Container
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install minimal dependencies
RUN pip install --no-cache-dir \
    flask \
    pandas \
    numpy \
    scikit-learn \
    requests \
    matplotlib \
    seaborn

# Create necessary directories
RUN mkdir -p logs models data/processed tests

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/

# Set environment variables
ENV FLASK_APP=src/model_api.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run application
CMD ["python", "src/model_api.py"]
