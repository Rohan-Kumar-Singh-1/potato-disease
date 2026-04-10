FROM python:3.10

# Install system dependencies (important for TensorFlow + PIL)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD ["python", "main.py"]