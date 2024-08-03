# Use the official lightweight Python image.
FROM python:3.9-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the current directory contents into the container at /app.
COPY . /app

# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on.
EXPOSE 5000

# Run the application.
CMD ["python", "src/app.py"]
