import yaml
import os
import torch
from ultralytics import YOLO

DATASET_DIR = ''
YAML_DIR = ''

# yaml config
data_config = {
    'path': DATASET_DIR, 
    'train': 'images/train', 
    'val': 'images/val', 
    'nc': 1,
    'names': {0: 'danger_zone'}
}

# write to yaml file
with open(YAML_DIR, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)



# get YOLO instance seg model
model = YOLO('yolo26n-seg.pt')

# detect cpu or gpu for kaggle t4x2 setup
device = [0,1] if torch.cuda.is_available() else 'cpu'

# train model
results = model.train(
    data=YAML_DIR,
    epochs=150,
    imgsz=640,
    save_period=5,
    batch=32,
    patience=10,
    device=device,
    project='runs',
    name='yolo_model'
)