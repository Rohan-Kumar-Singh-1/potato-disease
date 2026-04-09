import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import json
import uvicorn


app = FastAPI(title="Potato Disease Prediction API", version="1.0.0")

# Allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TensorFlow Serving endpoint
model = tf.keras.models.load_model("fixed_model.keras", compile=False)
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

IMG_SIZE = 224  # Change if your model uses a different input size


def preprocess_image(image_bytes: bytes) -> list:
    """Resize and normalize image for model input."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    return image.astype(np.float32) 


@app.get("/")
def root():
    return {"message": "Potato Disease Prediction API is running 🥔"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = preprocess_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)

    prediction = model.predict(image_batch, verbose=0)[0]
    print(prediction)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    info = CLASS_INFO[predicted_class]

    all_predictions = []
    for i, prob in enumerate(prediction):
        key = CLASS_NAMES[i]
        meta = CLASS_INFO[key]

        all_predictions.append({
            "class_key": key,
            "label": meta["label"],
            "probability": float(prob * 100),
            "color": meta["color"]
    })

    return {
        "class": predicted_class,
        "label": info["label"],
        "emoji": info["emoji"],
        "severity": info["severity"],
        "color": info["color"],
        "description": info["description"],
        "treatment": info["treatment"],
        "confidence": float(confidence * 100),
        "all_predictions": all_predictions
}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)