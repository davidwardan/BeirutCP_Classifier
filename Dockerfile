# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /BeirutCP_Classifier

# Copy the current directory contents into the container
COPY . /BeirutCP_Classifier

# Install necessary system dependencies for PyTorch, h5py, and opencv-python
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    libz-dev \
    libaec-dev \
    zlib1g-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    && apt-get clean

# Update pip
RUN pip install --upgrade pip

# Install PyTorch and Torchvision with appropriate CUDA (CPU-only as default)
# If GPU support is required, replace 'cpu' with 'cu118' or the desired CUDA version
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --use-feature=2020-resolver

# Expose port 8080
EXPOSE 8080

# Run the application
CMD ["python", "Classification_UI.py"]