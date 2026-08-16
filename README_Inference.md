
# Experimental YOLOv5 Inference for Palm Oil Tree Detection

This document preserves the prototype inference workflow developed for detecting and counting palm oil trees in aerial images. It is an experimental record rather than a validated production implementation. The threshold settings, bounding-box coordinate handling, and counting accuracy require further verification before operational use. Despite the original function name, the current forward call does not establish that Test-Time Augmentation (TTA) was activated.

## Table of Contents

-   [Overview](#overview)
-   [Detection Script](#detection-script)
-   [Usage Instructions](#usage-instructions)
-   [Examples](#examples)
    -   [Experimental Settings for Small Objects](#experimental-settings-for-small-objects)
    -   [Experimental Settings for Larger Objects](#experimental-settings-for-larger-objects)
-   [Key Parameters](#key-parameters)
-   [Current Limitations](#current-limitations)
-   [Acknowledgments](#acknowledgments)

## Overview

The prototype inference script:

1.  Loads the trained YOLOv5 model.
2.  Reads an input image (`image_path`).
3.  Contains an attempted **Test-Time Augmentation (TTA)** setting that is not passed to the model's forward call.
4.  Displays the annotated image in the Jupyter Notebook with bounding boxes for each detected palm oil tree.
5.  Prints the total count of detected trees to the console.

## Detection Script

Below is the inference script used to detect palm oil trees:
```python
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# YOLOv5/Ultralytics imports
from pathlib import Path
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression

def detect_with_tta(
    image_path,
    model_path,
    conf_threshold=0.25,
    iou_threshold=0.45,
    img_size=640,
    class_filter=None,
    device='cpu'
):
    """
    Runs the original experimental YOLOv5 inference workflow.
    
    Args:
    - image_path (str): Path to input image.
    - model_path (str): Path to YOLOv5 weights.
    - conf_threshold (float): Minimum confidence threshold for detection.
    - iou_threshold (float): IoU threshold for Non-Max Suppression.
    - img_size (int): Image size for inference.
    - class_filter (list): List of class IDs to detect (None = detect all classes).
    - device (str): 'cpu' or 'cuda'.
    
    Returns:
    - annotated_image (np.ndarray): Annotated image with bounding boxes.
    - count (int): Total number of detected objects.
    """

    # Load the model
    model = DetectMultiBackend(model_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    model.augment = True  # This assignment alone does not establish that TTA is used
    
    # Load and preprocess the image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = letterbox(img_rgb, img_size, stride=stride, auto=True)[0]
    img_resized = img_resized.transpose((2, 0, 1))[None]  # Convert to NCHW
    img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
    
    # Run inference
    with torch.no_grad():
        pred = model(img_tensor)[0]
    dets = non_max_suppression(pred, conf_threshold, iou_threshold, classes=class_filter)[0]
    
    # Draw bounding boxes
    annotated = img_rgb.copy()
    count = 0
    if dets is not None and len(dets):
        for *box, conf, cls_id in dets:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            count += 1
    
    # Display results
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated)
    plt.axis("off")
    plt.title(f"Detected {count} objects")
    plt.show()
    
    return annotated, count
```

## Usage Instructions
1.   **Prepare the Environment**:
		- Install YOLOv5 and its dependencies::

			```bash
			pip install ultralytics
			```
2. **Update the Script**:
    
	 -   Set the `image_path` to the path of your test image.
	-   Set the `model_path` to the path of your trained YOLOv5 weights (`best.pt`).
3. ****Run the Script**:**
     -   Execute the script in a Jupyter Notebook or Python environment.

## Examples
### Experimental Settings for Small Objects

The following settings were explored for images containing small objects. They were selected heuristically and were not calibrated against a labeled holdout set:
```python
image_path = "G:/Palm_Trees_dataset/images/val/ai_assignment_20241202_count.jpeg"
model_path = "runs/train/exp4/weights/best.pt"

annotated_img, detections_count = detect_with_tta(
    image_path=image_path,
    model_path=model_path,
    conf_threshold=0.001,  # Lower threshold for small objects
    iou_threshold=0.45,
    img_size=1920,         # Higher resolution for small objects
    device='cpu'
)

print(f"Detected {detections_count} palm oil trees.")
```

### Experimental Settings for Larger Objects
The following alternative settings were explored for images where trees appear larger:
```python
image_path = "G:/Palm_Trees_dataset/images/val/test.jpg"
model_path = "runs/train/exp4/weights/best.pt"

annotated_img, detections_count = detect_with_tta(
    image_path=image_path,
    model_path=model_path,
    conf_threshold=0.1,  # Higher threshold for larger objects
    iou_threshold=0.45,
    img_size=1024,       # Lower resolution for larger objects
    device='cpu'
)

print(f"Detected {detections_count} palm oil trees.")
```

## Key Parameters

1.  **`conf_threshold`**:
    
    -   A very low value such as `0.001` retains many candidate detections and can substantially increase false positives.
    -   A higher value such as `0.1` filters more low-confidence detections.
2.  **`img_size`**:
    
    -   Larger image sizes were explored for small-object detection, but their effect was not measured in a controlled comparison.
    -   Image size also changes processing time and memory requirements.
3.   **`iou_threshold`**:
    
	    -   Controls the strictness of Non-Maximum Suppression (default: `0.45`).

## Current Limitations

- The repository does not contain a controlled comparison of inference with and without TTA.
- The current `model(img_tensor)` call does not pass `augment=True`, so the saved notebook does not establish that TTA was executed.
- The prototype's bounding-box coordinates should be checked against the original image after letterbox resizing.
- Confidence and image-size settings are heuristic rather than calibrated.
- Counting performance has not been compared with manually verified counts; the 300-detection example also reaches YOLOv5's default detection ceiling.
- The example has not been reconstructed and tested from a clean environment.

## Acknowledgments

-   **Dataset**: [RoboFlow Oil Palm Detection Dataset](https://universe.roboflow.com/manfred-michael/oil-palm-detection/dataset/6).
-   **YOLOv5**: The Ultralytics team for their exceptional object detection framework.
-   **ChatGPT**: Assisted in refining and documenting the inference pipeline.
