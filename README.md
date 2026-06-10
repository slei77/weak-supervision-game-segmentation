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

## 1. Project Overview
This project explores weakly supervised segmentation in bullet-hell game environments. Because obtaining ground-truth projectile annotations from game footage is difficult, OpenCV-based HSV contour heuristics were used to generate pseudo-labels for training a YOLO segmentation model. The project investigates how label noise affects learned representations and evaluates model behavior against manually corrected annotations.

Initially, the task was defined as bullet instance segmentation. However, due to limitations in heuristic label generation, discrete projectiles were frequently merged. As a result, the task was reframed as hazard-region segmentation, where labels represent spatial regions of projectile rather than individual bullets. 

## 2. Example Result



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



## 9. Future Work

Future work would focus on extracting ground-truth projectile and indicator data directly from game memory or engine state, allowing for large-scale perfectly labeled datasets and more precise instance segmentation.
