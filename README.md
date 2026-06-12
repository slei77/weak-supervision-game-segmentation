# Weak Supervision Game Segmentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Example Result](#2-example-result)
3. [Motivation](#3-motivation)
4. [Pipeline](#4-pipeline)
5. [Dataset](#5-dataset)
6. [Evaluation](#6-evaluation)
7. [Key Findings](#7-key-findings)
8. [Failure Cases](#8-failure-cases)
9. [Future Work](#9-future-work)
10. [Repository Structure](#10-repository-structure)

## 1. Project Overview
This project explores weakly supervised segmentation in bullet-hell game environments. Because obtaining ground-truth projectile annotations from game footage is difficult, OpenCV-based HSV contour heuristics were used to generate pseudo-labels for training a YOLO segmentation model. The project investigates how label noise affects learned representations and evaluates model behavior against manually corrected annotations.

Initially, the task was defined as bullet instance segmentation. However, due to limitations in heuristic label generation, discrete projectiles were frequently merged. As a result, the task was reframed as hazard-region segmentation, where labels represent spatial regions of projectile rather than individual bullets. 

## 2. Example Result
| Frame | Prediction |
| --- | --- |
|<img src="https://github.com/user-attachments/assets/0e7e9e14-912a-4ea2-bef4-c41ebf4bb14f" />|<img src="https://github.com/user-attachments/assets/a56bc281-ecf8-4543-830c-89fae9a236cd" />|


| Auto Label | Manual Label |
| --- | --- |
|<img src="https://github.com/user-attachments/assets/5629afc2-2d3c-4721-a9b9-0235256f4f9a" />|<img src="https://github.com/user-attachments/assets/cbd2ab8b-33a4-40c3-948e-952843a7d3dd" />|

## 3. Motivation

Bullet-hell scenes contain hundreds of projectiles on-screen at once, making manual instance segmentation incredibly time-consuming. Pseudo-labeling solves this by accelerating dataset generation, producing thousands of labels in the time it would take to manually annotate just a few samples.

## 4. Pipeline

1. Game Frames
2. HSV Thresholding with Median Blur and Erosion
3. Contour Extraction and Filtering
4. YOLO Segmentation Training
5. Predictions
6. Evaluation

## 5. Dataset

Training Images: 1399
Validation Images: 278
Manual and Auto Evaluation Images: 20

## 6. Evaluation

### Label Agreement:

| Comparison | Mean IoU |
| --- | --- |
| Auto vs Manual | 0.81 |
| Manual vs Auto | 0.65 |

The asymmetric mean IoU indicates that the heuristic auto-labeler causes over-segmentation, which is also evident in the failure cases section. 

### Model Evaluation:

| Comparison | Mean IoU |
| --- | --- |
| Prediction vs Auto | 0.52 |
| Prediciton vs Manual | 0.60 |

Model predictions aligned more closely with manually refined annotations than with original pseudo-labels.

## 7. Key Findings

- HSV contour heuristics enabled rapid pseudo-label generation but has trouble separating nearby projectiles, at least without heavy curation. 
- On a manually corrected evaluation subset, model predictions exhibited higher agreement with human-refined annotations than with the original pseudo-labels.
- Dataset quality appeared to be the primary limitation for precise projectile instance segmentation rather than model capacity.

## 8. Failure Cases

| Prediction | Auto Label |
| --- | --- |
|<img src="https://github.com/user-attachments/assets/a56bc281-ecf8-4543-830c-89fae9a236cd" />|<img src="https://github.com/user-attachments/assets/5629afc2-2d3c-4721-a9b9-0235256f4f9a" />|

The model mistakenly detects the glow around the projectile. This stems from an issue with the auto-labeler, which struggles to differentiate between the projectile itself and its glow. Consequently, the model is training on these weak labels and learning to include the glow in its predictions. 


| Auto Label | Prediction |
| --- | --- |
|<img width="2560" height="1600" alt="s4_frame_0452_visualization" src="https://github.com/user-attachments/assets/4cb317a8-d452-4975-ba82-0fd5fb608855" />|<img width="2560" height="1600" alt="predicted_frame" src="https://github.com/user-attachments/assets/32100c37-1e32-49f2-b3dc-9ec552dac805" />|

The auto-labeler captured background noise around the boss despite the HSV filter being limited to magenta and purple. This discrepancy is likely caused by the blur and erosion preprocessing steps, which accidentally smeared adjacent red aura around the boss into the target magenta spectrum before filtering.


| Auto Label | Prediction |
| --- | --- |
|<img width="2560" height="1600" alt="s4_frame_0030_visualization" src="https://github.com/user-attachments/assets/2c0b32a8-816c-49a5-a169-9aecfbadbbb5" />|<img width="2560" height="1600" alt="predicted_frame" src="https://github.com/user-attachments/assets/bab30082-6ba6-4a34-a538-6526be6e8228" />|

The auto-labeler struggles to detect parts of the circle due to its dark purple hue and extremely low Value (V) channel. Compounded by the overall darkness of the scene, the labeler also mistakenly captured parts of the faint aura. While the model's predictions suffer from the same dark-color limitations, it interestingly managed to suppress some of that faint aura.

## 9. Future Work

Future work would focus on extracting ground-truth projectile and indicator data directly from game memory or engine state, allowing for large-scale perfectly labeled datasets and more precise instance segmentation.

## 10. Repository Structure

```
weak-supervision-game-segmentation/
├── model/
│   ├── best.onnx
│   └── best.pt
├── results/
├── src/
├── README.md
└── dataset.txt
```