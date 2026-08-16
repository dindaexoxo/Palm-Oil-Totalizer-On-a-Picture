# YOLOv5 Training for Palm Oil Tree Detection

This document records the original workflow used to prepare the dataset, convert annotations, and train a YOLOv5 model for detecting palm oil trees. Local paths and dependency versions may require adjustment before the workflow can be reproduced in a new environment.

## Table of Contents

-   [Dataset Preparation](#dataset-preparation)
-   [Dataset Conversion (COCO to YOLO)](#dataset-conversion-coco-to-yolo)
-   [Training the Model](#training-the-model)
-   [Validation Notes](#validation-notes)
-   [Key Notes](#key-notes)

## Dataset Preparation

1.   **Download the Dataset**:
    
	    -   The palm oil tree dataset can be downloaded from [RoboFlow Oil Palm Detection Dataset](https://universe.roboflow.com/manfred-michael/oil-palm-detection/dataset/6).
2.   **Organize the Dataset**:
    
	    -   After downloading the dataset in COCO JSON format, organize it as follows:
		    ``` kotlin
			Palm_Trees_dataset/
			â”œâ”€â”€ annotations/
			â”‚   â”œâ”€â”€ instances_train.json
			â”‚   â”œâ”€â”€ instances_val.json
			â”œâ”€â”€ images/
			â”‚   â”œâ”€â”€ train/
			â”‚   â”œâ”€â”€ val/
		    ```

## Dataset Conversion (COCO to YOLO)
Since YOLOv5 requires annotations in YOLO format, use the following script to convert COCO annotations to YOLO format.
### Conversion Script: `coco_to_yolo.py`
```python 
import json
import os
from tqdm import tqdm

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0  # Center x
    y = box[1] + box[3] / 2.0  # Center y
    w = box[2]
    h = box[3]
    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    return (x, y, w, h)

datasets = ['train', 'val']
root_path = 'G:/Palm_Trees_dataset'

for dataset in datasets:
    json_path = os.path.join(root_path, 'annotations', f'instances_{dataset}.json')
    labels_dir = os.path.join(root_path, 'labels', dataset)
    os.makedirs(labels_dir, exist_ok=True)
    
    with open(json_path) as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    
    for ann in tqdm(data['annotations'], desc=f'Converting {dataset} annotations'):
        img = images[ann['image_id']]
        bbox_yolo = convert((img['width'], img['height']), ann['bbox'])
        class_id = ann['category_id'] - 1
        
        label_path = os.path.join(labels_dir, os.path.splitext(img['file_name'])[0] + '.txt')
        with open(label_path, 'a') as f:
            f.write(f"{class_id} {' '.join(map(str, bbox_yolo))}\n")
```
### Steps to Run the Script
1. Place the dataset in the structure shown above.
2. Run the script:
	```bash
	python coco_to_yolo.py
	```
3. After running, the dataset structure will include YOLO annotations
	``` kotlin
	Palm_Trees_dataset/
	â”œâ”€â”€ annotations/
	â”œâ”€â”€ images/
	â”‚   â”œâ”€â”€ train/
	â”‚   â”œâ”€â”€ val/
	â”œâ”€â”€ labels/
	â”‚   â”œâ”€â”€ train/
	â”‚   â”œâ”€â”€ val/
	```
## Training the Model

The following script trains the YOLOv5 model using the converted dataset.

### Training Script
```python
# Import necessary modules
import torch
import yaml
from yolov5 import train

# Adjust arguments for training
args = {
    'img_size': 640,
    'batch_size': 4,
    'epochs': 10,
    'data': 'palm_tree.yaml',
    'weights': 'yolov5n.pt',
    'project': 'runs/train',
    'name': 'exp',
    'device': 'cpu',
}

# Run training
train.run(**args)
```
### Validation Notes

1. Update the validation command so that `weights` points to the trained model (`best.pt`).
2. Execute the validation script:
	```bash
	python val.py
	```
3. Validation results (precision, recall, and mAP) will be displayed in the console and saved under the `runs/val` directory.

The recorded results in this repository come from the validation split. A separately documented final test-set evaluation is not currently included, and the configuration should not be interpreted as systematically optimized.

## Key Notes

1.   **Batch Size**:
	    -   Reduce `batch_size` if you encounter memory issues, especially on CPUs.
2.   **Epochs**:
	    -   Start with fewer epochs (e.g., 10) and gradually increase if the model underfits.
3.   **Input Size**:
	    -   The `img_size` parameter controls the input image resolution. Use `640` for a balance between speed and accuracy.
