import tensorflow as tf
import numpy as np

# Load your fixed model
model = tf.keras.models.load_model("backend/fixed_model.keras", compile=False)

print("✅ Model loaded successfully!")

# Test prediction
dummy = np.random.rand(1, 224, 224, 3).astype("float32")
pred = model.predict(dummy)

print("✅ Prediction works!")
print(pred)