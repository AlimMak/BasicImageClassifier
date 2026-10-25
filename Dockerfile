# Start from a small Linux box that already has Python 3.11
FROM python:3.11-slim

# Work inside /app in the container
WORKDIR /app

# CPU version of PyTorch: much smaller, and the API doesn't need a GPU
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in only what the API needs: the model code, the API, and the trained weights
COPY model.py app.py model.pt ./

# Start the server when the container runs
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]