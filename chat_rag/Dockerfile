# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 6006 available to the world outside this container
EXPOSE 6006

# Define environment variable
ENV STREAMLIT_SERVER_PORT=6006

WORKDIR /app/src
# Run flask when the container launches
CMD ["streamlit", "run", "--server.port", "6006", "app.py"]
