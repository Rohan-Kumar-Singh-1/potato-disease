import tensorflow as tf

# Load old model
model = tf.keras.models.load_model("backend/potato_model.keras", compile=False)

# Save in new compatible format
model.save("fixed_model.keras")

print("Model fixed and saved!")