# Use an official PyTorch image with CUDA support or Python base image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the contents of your project to /app in the container
COPY . .

# Set the PYTHONPATH to include the src directory
ENV PYTHONPATH=/app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port Gradio will run on (default: 7860)
EXPOSE 7860

# Command to run the application
CMD ["python", "./src/main.py"]