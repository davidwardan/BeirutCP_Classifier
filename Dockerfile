# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /BeirutCP_Classifier

# Copy the current directory contents into the container
COPY . /BeirutCP_Classifier

# Install necessary system dependencies for h5py and opencv-python
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
    && apt-get clean

# Update pip
RUN pip install --upgrade pip

# Install a pre-built h5py wheel to avoid building from source
RUN pip install h5py --no-binary=h5py

# Install Python packages specified in requirements.txt (skip h5py, already installed)
RUN pip install --no-cache-dir -r requirements.txt --use-feature=2020-resolver

# Expose port 8080
EXPOSE 8080

# Run the application
CMD ["python", "Classification_UI.py"]
