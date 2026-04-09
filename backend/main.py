from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
from PIL import Image
import io
import json

app = FastAPI(title="Potato Disease Prediction API", version="1.0.0")

# Allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ← change this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TensorFlow Serving endpoint
TF_SERVING_URL = "http://localhost:8501/v1/models/plant_disease_model:predict"

# Class labels — update to match your model's output order
CLASS_NAMES = [
    "Potato___Early_Blight",
    "Potato___Late_Blight",
    "Potato___healthy",
]

# Friendly display names and advice
CLASS_INFO = {
    "Potato___Early_Blight": {
        "label": "Early Blight",
        "emoji": "🍂",
        "severity": "moderate",
        "color": "#f59e0b",
        "description": "Caused by Alternaria solani fungus. Appears as dark brown spots with concentric rings.",
        "treatment": "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves and avoid overhead watering.",
    },
    "Potato___Late_Blight": {
        "label": "Late Blight",
        "emoji": "🦠",
        "severity": "severe",
        "color": "#ef4444",
        "description": "Caused by Phytophthora infestans. Appears as water-soaked lesions turning dark brown/black.",
        "treatment": "Apply copper-based fungicides immediately. Destroy severely infected plants. Improve drainage and air circulation.",
    },
    "Potato___healthy": {
        "label": "Healthy",
        "emoji": "✅",
        "severity": "none",
        "color": "#22c55e",
        "description": "No disease detected. Your potato plant looks healthy!",
        "treatment": "Continue regular care: water at soil level, rotate crops yearly, monitor for pests.",
    },
}

IMG_SIZE = 256  # Change if your model uses a different input size


def preprocess_image(image_bytes: bytes) -> list:
    """Resize and normalize image for model input."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    return img_array.tolist()  # shape: [256, 256, 3]


@app.get("/")
def root():
    return {"message": "Potato Disease Prediction API is running 🥔"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    try:
        input_data = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process image: {str(e)}")

    payload = {"instances": [input_data]}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(TF_SERVING_URL, json=payload)
            response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to TensorFlow Serving. Make sure Docker is running.",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TensorFlow Serving returned error: {e.response.text}",
        )

    tf_result = response.json()
    predictions = tf_result.get("predictions", [[]])[0]

    if len(predictions) != len(CLASS_NAMES):
        raise HTTPException(
            status_code=500,
            detail=f"Model returned {len(predictions)} classes, expected {len(CLASS_NAMES)}.",
        )

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    # Build full result with all class probabilities
    all_predictions = [
        {
            "class_key": CLASS_NAMES[i],
            "label": CLASS_INFO[CLASS_NAMES[i]]["label"],
            "probability": round(float(predictions[i]) * 100, 2),
            "color": CLASS_INFO[CLASS_NAMES[i]]["color"],
        }
        for i in range(len(CLASS_NAMES))
    ]
    all_predictions.sort(key=lambda x: x["probability"], reverse=True)

    info = CLASS_INFO[predicted_class]

    return {
        "predicted_class": predicted_class,
        "label": info["label"],
        "confidence": round(confidence, 2),
        "severity": info["severity"],
        "color": info["color"],
        "emoji": info["emoji"],
        "description": info["description"],
        "treatment": info["treatment"],
        "all_predictions": all_predictions,
    }
