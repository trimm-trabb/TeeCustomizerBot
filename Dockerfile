# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY requirements.txt ./
COPY app.py ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the required port for Hugging Face Spaces
EXPOSE 7860

# Run Chainlit on startup
CMD chainlit run app.py 
