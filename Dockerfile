# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from buffering output and to set the Flask environment
ENV PYTHONUNBUFFERED 1
ENV FLASK_ENV=production

# Set the working directory in the Docker container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy the rest of the application code into the container at /app
COPY . /app

# Expose the port that the Flask app runs on
ENV PORT=5000
EXPOSE 5000

# Define the command to run the Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
