import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import os
import shutil

KERAS_MODEL_PATH = "/home/rohan/workspace/DL/Model/plant_disease_model.keras"
EXPORT_DIR = "serving_model/1"

if os.path.exists("serving_model"):
    shutil.rmtree("serving_model")


model = tf.keras.models.load_model(KERAS_MODEL_PATH)

model.export(EXPORT_DIR)

print(f"Model exported to: {EXPORT_DIR}")