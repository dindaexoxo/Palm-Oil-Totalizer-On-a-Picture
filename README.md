
# Palm Oil Tree Detection & Counting

This project focuses on detecting and counting palm oil trees in drone images using the YOLOv5 object detection model. The dataset used is sourced from [RoboFlow Oil Palm Detection Dataset](https://universe.roboflow.com/manfred-michael/oil-palm-detection/dataset/6), and the annotations were converted to YOLO format for training.

## Table of Contents

-   [Overview](#overview)
-   [Dataset Structure](#dataset-structure)
-   [Pipeline](#pipeline)
    -   [1. Dataset Conversion](#1-dataset-conversion)
    -   [2. Training](#2-training)
    -   [3. Inference](#3-inference)
-   [Dependencies](#dependencies)
-   [Acknowledgments](#acknowledgments)
-   [License](#license)

## Overview

The goal of this project is to:

1.  Detect palm oil trees in drone images.
2.  Count the number of trees in each image.
3.  Provide bounding boxes around each detected tree.

This project uses:

-   **YOLOv5** for object detection.
-   **Test-Time Augmentation (TTA)** to improve detection accuracy, especially for small or distant objects.

## Dataset Structure

The dataset follows this directory structure after downloading and preparation:
```bash
Palm_Trees_dataset/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
├── train/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── val/
│   ├── imgX.jpg
│   ├── imgY.jpg
│   └── ...
└── test/
    ├── imgA.jpg
    ├── imgB.jpg
    └── ...
```
-   **annotations/**: Contains COCO JSON files for training, validation, and testing sets.
-   **train/**: Contains the training images.
-   **val/**: Contains the validation images.
-   **test/**: Contains the test images (if applicable).

## Pipeline
### 1. Dataset Conversion

To convert the COCO annotations into YOLO format:

1.  Download the dataset in COCO JSON format.
2.  Organize the dataset as shown above.
3.  Use the `coco_to_yolo.py` script to convert COCO annotations to YOLO format:
	```bash
	python coco_to_yolo.py
	```
4. After running, the YOLO-format labels will be saved in `labels/train`, `labels/val`, and `labels/test`.

### 2. Training

The YOLOv5 model is trained using the converted dataset. Key parameters include:

-   **Input image size**: `640 x 640`.
-   **Batch size**: `4` (adjust for memory limitations).
-   **Number of epochs**: `10`.

The training process saves the best model weights in the `runs/train/exp` directory.

For detailed training instructions, refer to [README_Training.md](README_Training.md).

### 3. Inference

The `detect_with_tta` script is used for inference:

-   Takes an input image.
-   Detects and counts palm oil trees.
-   Displays the annotated image in Jupyter Notebook with bounding boxes.

For detailed inference instructions, refer to [README_Inference.md](README_Inference.md).

## Dependencies

Ensure the following dependencies are installed:

-   Python 3.8+
-   PyTorch
-   YOLOv5 (clone the repository or install via `pip install ultralytics`)
-   OpenCV
-   Matplotlib
-   tqdm

Install the required Python libraries:
```bash
pip install -r requirements.txt
```

## Acknowledgments

-   **Dataset**: [RoboFlow Oil Palm Detection Dataset](https://universe.roboflow.com/manfred-michael/oil-palm-detection/dataset/6).
-   **YOLOv5**: The Ultralytics team for their exceptional object detection framework.
-   **ChatGPT**: Assisted with code structuring, debugging, and documentation.

## Acknowledgments

-   **Ultralytics/YOLOv5**: Core detection framework.
-   **ChatGPT**: Assisted in generating code snippets and documentation structure.
-   **Community**: Many online resources and blog posts that guide object detection tasks.

## License

This project is open-sourced under the MIT License. Portions of this repository’s content, including code snippets and explanations, were generated using **ChatGPT** by OpenAI.

```sql
MIT License

Copyright (c) 2025 

Permission is hereby granted, free of charge, to any person obtaining a copy   
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights   
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      
copies of the Software, and to permit persons to whom the Software is          
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in     
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING        
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER           
DEALINGS IN THE SOFTWARE.
```
