from ultralytics import YOLO
import os

MODEL_DIR = ''
OUTPUT_DIR = ''

model = YOLO(MODEL_DIR)

model.predict(
    source=OUTPUT_DIR,
    save_txt=True,
    name="predictions"
)