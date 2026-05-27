**Developed by Group MAGNIFICAT**

# 🍎 Fruit Ripeness Detection & Classification (Google Colab Pipeline)

A complete deep learning pipeline designed to execute inside Google Colab. This project solves the complex challenge of merging multiple independent fruit datasets exported from Roboflow to train both Object Detection (YOLOv8) and lightweight Image Classification (MobileNetV2) models.

## 📌 Project Overview

**Supported Fruits (5):** Avocado, Guava, Mango, Pineapple, Banana  
**Ripeness Stages (3):** Unripe, Ripe, Overripe  
**Total Classes:** 15  

### Supported Architectures
* **YOLOv8s:** Primary object detection model. Responsible for identifying fruit locations (bounding boxes) and initial classification.
* **MobileNetV2 (α=0.5 & α=1.0):** Secondary classification models optimized for edge hardware. Evaluates isolated fruit crops to provide high-speed ripeness validation. Includes TensorFlow Lite (`.tflite`) conversion.

---

## 📦 Model Weights (Download Links)

Due to GitHub file size limits, the trained heavy `.keras` weights are hosted externally on Google Drive.

* [Download MobileNetV2 Alpha 0.5 (.keras)](INSERT_YOUR_ALPHA_05_GOOGLE_DRIVE_LINK_HERE)
* [Download MobileNetV2 Alpha 1.0 (.keras)](INSERT_YOUR_ALPHA_10_GOOGLE_DRIVE_LINK_HERE)

*(Note: The optimized, deployment-ready `.tflite` edge models and the trained YOLOv8s weights are included directly in this repository).*

---

## 🚀 How to Run (YOLOv8s Workflow)

The `Fruit_Ripeness_Pipeline.ipynb` is structured to run sequentially.

### Part 1: Setup, Data Preprocessing & Training
* **[CELL 1] — Setup & Dependencies:** Installs `ultralytics` and necessary plotting/vision libraries.
* **[CELL 2] — Configuration & Mapping:** Defines the core 15-class Global Mapping system.
* **[CELL 3] — Storage Mounting:** Mounts Google Drive and verifies dataset paths.
* **[CELL 4] — Automated Dataset Integration:** Extracts datasets, utilizes string-matching to fix overlapping local IDs, and merges validated data into a centralized folder.
* **[CELL 5] — Master Configuration:** Writes the final merged `data.yaml` file required by YOLO.
* **[CELL 6] — Distribution Audit:** Generates class balance visualizations.
* **[CELL 6.5] — Automated Splitting:** Partitions 15% Validation and 10% Testing data.
* **[CELL 7] — Model Training:** Fine-tunes `yolov8s.pt` utilizing built-in early stopping and automatic augmentations.

### Part 2: Evaluation & Inference
* **[CELL 8] — Reload Environment:** Safety net for Colab runtime disconnects.
* **[CELL 9] — Evaluation:** Computes final mAP50, Precision, and Recall.
* **[CELL 10] — Confusion Matrix:** Visualizes classification errors.
* **[CELL 11-12] — Inference:** Translates raw outputs into colored bounding boxes and generates a final visual predictions grid.

---

## 🚀 How to Run (MobileNetV2 Workflow)

Run either `MobileNetV2_Alpha05_Pipeline.ipynb` or `MobileNetV2_Alpha10_Pipeline.ipynb`. These notebooks repurpose the YOLO bounding boxes to train a standard Keras classifier.

### Part 1: Keras Dataset Formatting
* **[CELL 1-3] — Setup:** Loads TensorFlow, Keras, and mounts storage.
* **[CELL 4-5] — Base Integration:** Re-runs the global mapping to clean raw YOLO datasets.
* **[CELL 6] — YOLO to Classification Conversion:** Reads YOLO bounding box coordinates, crops the exact pixel area of every fruit, resizes to `224x224`, and saves them into strict Keras class-separated folders (`clf_dataset/train/ClassName`).
* **[CELL 7] — ImageDataGenerators:** Applies real-time data augmentation (rotation, zoom, brightness shifts).

### Part 2: Two-Phase Transfer Learning
* **[CELL 8] — Architecture:** Initializes MobileNetV2 (α=0.5 or 1.0) with frozen ImageNet weights and attaches a custom GlobalAveragePooling & Dropout classification head.
* **[CELL 9] — Phase 1 (Head Training):** Trains only the new top layers while the base remains completely frozen.
* **[CELL 10] — Phase 2 (Fine-Tuning):** Unfreezes the top 30 layers of the base model and trains at a significantly reduced learning rate for precision adjustment.

### Part 3: Evaluation & Edge Export
* **[CELL 11-14] — Metrics:** Plots continuous loss/accuracy curves across both training phases, prints the sklearn Classification Report, and generates dual-pane Confusion Matrices.
* **[CELL 15] — Export:** Saves the `.keras` model and automatically compiles a highly compressed `.tflite` model for edge deployment.
* **[CELL 16] — Visual Grid:** Runs final inference on 15 random test crops.

---

## 🛠️ Key Technical Features
1. **Automated Error Correction:** The pipeline utilizes robust logic to fix incorrect `data.yaml` labels dynamically during the integration phase.
2. **Keras Format Conversion:** Automatically bridges the gap between Object Detection datasets and Image Classification architectures by translating bounding box coordinates into physical image crops.
3. **Edge Readiness:** Supports explicit Alpha scaling (0.5/1.0) and TensorFlow Lite optimization for low-power hardware constraints.
