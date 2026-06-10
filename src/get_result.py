from ultralytics import YOLO

MODEL_DIR = ''
INPUT_DIR = ''

model = YOLO(MODEL_DIR, task='segment')

results = model(INPUT_DIR)

results[0].save(filename='result.jpg')