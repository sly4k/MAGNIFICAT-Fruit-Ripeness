**Developed by Group MAGNIFICAT**

# 🍎 YOLOv8 Fruit Ripeness Detection (Google Colab Version)

A complete, cell-by-cell deep learning pipeline designed to execute inside Google Colab. This project detects and classifies fruits and their specific ripeness stages using **YOLOv8s** via Transfer Learning.

## 📌 Project Overview
This project solves the complex challenge of merging multiple independent fruit datasets exported from Roboflow into a single, unified deep learning pipeline. 

**Supported Fruits (5):** Avocado, Guava, Mango, Pineapple, Banana  
**Ripeness Stages (3):** Unripe, Ripe, Overripe  
**Total Classes:** 15  
**Architecture:** YOLOv8s (Ultralytics)

---

## 🚀 How to Run (Cell by Cell Workflow)

The codebase is structured to be run sequentially in Google Colab. Below is the exact breakdown of what each cell accomplishes.

### Part 1: Setup, Data Preprocessing & Training
*This phase handles the automated extraction, cleaning, mapping, and training.*

* **[CELL 1] — Setup & Dependencies:** Installs `ultralytics` and necessary plotting/vision libraries.
* **[CELL 2] — Configuration & Mapping:** Defines the core 15-class Global Mapping system to ensure consistency across all fruit datasets.
* **[CELL 3] — Storage Mounting:** Mounts Google Drive, sets file paths, and runs a sanity check to ensure the datasets actually exist.
* **[CELL 4] — Automated Dataset Integration:** 
  * Extracts each fruit dataset.
  * **Smart Mapping:** Reads internal `data.yaml` files and automatically translates overlapping local IDs (e.g., Avocado `0` and Guava `0`) into the unified Global 15-Class ID system.
  * Filters out corrupted images and merges everything into a centralized folder.
* **[CELL 5] — Master Configuration:** Writes the final merged `data.yaml` file required by YOLOv8.
* **[CELL 6] — Distribution Audit:** Generates a bar chart to verify class balance and total dataset volume.
* **[CELL 6.5] — Automated Splitting:** Takes the merged training data and strictly partitions 15% for Validation and 10% for Testing to prevent overfitting.
* **[CELL 7] — Model Training:** Fine-tunes `yolov8s.pt` over 50 epochs utilizing built-in early stopping, cosine learning rate decay, and automatic augmentations.

### Part 2: Evaluation, Inference & Explainability
*This phase runs after the training is complete to test the model's accuracy.*

* **[CELL 8] — Reload Environment:** A safety net to re-import libraries and paths in case the Colab runtime disconnects after the long training process.
* **[CELL 9] — Evaluation on Unseen Data:** Computes the final mAP50, Precision, and Recall against the separated test data.
* **[CELL 10] — Confusion Matrix:** Visualizes exactly which fruit types or ripeness stages the model is confusing (Major vs. Minor errors).
* **[CELL 11] — Real-World Inference:** The core prediction function. Translates raw model outputs back into human-readable bounding boxes colored by fruit type, with confidence scores.
* **[CELL 12] — Visual Predictions Grid:** Generates a massive grid of randomly sampled test predictions for a quick, final visual sanity check.

---

## 🛠️ Key Technical Features
1. **Transfer Learning:** Leverages a backbone pre-trained on COCO to achieve high accuracy quickly without requiring massive datasets.
2. **Automated Error Correction:** The Integration Cell (Cell 4) uses robust string-matching logic to fix incorrect or overlapping labels in the raw dataset files.
3. **End-to-End Execution:** Entirely self-contained pipeline from raw `.zip` extraction to deployed visual inference.
