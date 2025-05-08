# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy and install system packages
COPY packages.txt .
RUN apt-get update && \
    xargs -a packages.txt apt-get install -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy fonts if necessary
COPY fonts/ /usr/share/fonts/truetype/custom/
COPY dejavu-fonts-ttf-2.37/ /usr/share/fonts/truetype/dejavu/
RUN fc-cache -fv

# Copy app files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Streamlit-specific settings
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app_1.5_flash.py", "--server.port=8501", "--server.enableCORS=false"]
