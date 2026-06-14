FROM python:3.12-slim

# Install system-level dependencies required for PDF/Image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libx11-xcb1 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy your requirements file first (for caching)
COPY requirements.txt .

# 4. Install system dependencies (required for some Python packages like PyMuPDF)
RUN apt-get update && apt-get install -y gcc g++ libffi-dev

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Download the Spacy language model required for PII masking
RUN python -m spacy download en_core_web_sm

# 7. Copy the rest of your backend code
COPY . .

# 8. Expose the port FastAPI runs on
EXPOSE 8000

# 9. Start the Uvicorn server (0.0.0.0 allows external connections to the container)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]