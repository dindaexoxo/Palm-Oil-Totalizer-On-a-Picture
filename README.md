# Palm Oil Tree Detection and Counting with YOLOv5

This portfolio project explores one-class oil palm tree detection in aerial imagery using a pretrained YOLOv5 model. The repository documents the workflow used to convert annotations, train and validate the detector, and experiment with counting trees from retained bounding-box detections.

## Project Scope

The project demonstrates:

- conversion of COCO bounding-box annotations into YOLO format;
- fine-tuning of a pretrained YOLOv5n detector;
- evaluation with object-detection metrics on a validation split; and
- experimental image inference in which the number of retained detections is used as the estimated tree count.

This is a learning and portfolio project. It has not been validated as a production system or as an operational forestry inventory tool.

## Full Inference Notebook

The complete notebook with embedded inference and counting outputs is available here:

**[Open `Palm-oil-tree-totalizer-on-a-picture.ipynb` on Google Drive](https://drive.google.com/file/d/17sTeQFwqQBKdBTkpao7Du49WFuImjL2Q/view?usp=sharing)**

The notebook is hosted externally because its embedded outputs make the file approximately 105 MB. Its experimental results should be interpreted together with the limitations documented below.

## Dataset

The project uses version 6 of the [Oil Palm Detection dataset on Roboflow Universe](https://universe.roboflow.com/manfred-michael/oil-palm-detection/dataset/6). It contains aerial images with bounding-box annotations for a single oil palm class.

The source dataset provides separate training, validation, and test directories. The workflow in this repository converts the available COCO annotations into YOLO text labels.

## Repository Contents

| File | Description |
| --- | --- |
| `coco_to_yolo.py` | Converts COCO bounding boxes into normalized YOLO labels. |
| `Train-Code-YOLOv5.ipynb` | Records the original model configuration, training, and validation workflow. |
| `best.pt` | Saved model weights from the training experiment. |
| `model performance.JPG` | Screenshot of the recorded YOLOv5 validation output. |
| `README_Training.md` | Additional notes about the original training workflow. |
| `README_Inference.md` | Notes and prototype code for experimental inference and counting. |
| [Full inference and counting notebook](https://drive.google.com/file/d/17sTeQFwqQBKdBTkpao7Du49WFuImjL2Q/view?usp=sharing) | Original `Palm-oil-tree-totalizer-on-a-picture.ipynb` notebook hosted on Google Drive because the file is approximately 105 MB. |

## Workflow

### 1. Annotation Conversion

The `coco_to_yolo.py` script converts each COCO bounding box from `[x_min, y_min, width, height]` into normalized YOLO coordinates:

```text
class_id x_center y_center width height
```

The paths in the original script reflect the local environment used for the experiment and must be changed before reuse.

### 2. Model Training

The recorded notebook uses the following configuration:

| Parameter | Recorded value |
| --- | --- |
| Base model | YOLOv5n pretrained weights |
| Input size | 640 x 640 pixels |
| Batch size | 4 |
| Epochs | 10 |
| Device | CPU |
| Classes | 1 (`palm-oil`) |

These settings document the completed experiment; they are not presented as an optimized configuration.

### 3. Recorded Validation Results

The saved validation output reports results on the dataset's validation split of **813 images** containing **1,951 annotated instances**:

| Metric | Recorded value |
| --- | ---: |
| Precision | 0.843 |
| Recall | 0.890 |
| mAP@0.5 | 0.876 |
| mAP@0.5:0.95 | 0.519 |

![Recorded YOLOv5 validation output](model%20performance.JPG)

These are object-detection metrics, not a general "detection accuracy" score. The repository does not currently contain a separately recorded evaluation on an independent test set.

### 4. Experimental Counting

The prototype inference workflow estimates a tree count by counting bounding boxes retained after confidence filtering and non-maximum suppression. This demonstrates the mechanics of detection-based counting, but counting quality has not been evaluated against manually verified image-level counts using metrics such as mean absolute error.

`README_Inference.md` contains the experimental inference notes. The [full inference and counting notebook](https://drive.google.com/file/d/17sTeQFwqQBKdBTkpao7Du49WFuImjL2Q/view?usp=sharing) is hosted externally because its embedded outputs make the file approximately 105 MB.

The notebook records examples returning 300 and 8 detections. The 300-detection result reaches YOLOv5's default `max_det=300` limit and therefore cannot be interpreted as the complete number of trees in the image. Neither example is compared with a manually verified ground-truth count.

Although the inference function is named `detect_with_tta` and assigns `model.augment = True`, its forward call does not pass `augment=True`. In the YOLOv5 `DetectMultiBackend` interface, the forward argument defaults to `False`; consequently, the current notebook does not establish that TTA was executed.

## Limitations

- The recorded metrics come from the validation split rather than a separately documented final test evaluation.
- Generalization to imagery from different plantations, cameras, resolutions, seasons, or flight conditions has not been measured.
- The current inference call does not provide evidence that Test-Time Augmentation was activated.
- Confidence thresholds were explored heuristically and were not calibrated on an independent dataset.
- Detection-based counts may include false positives, missed trees, or duplicate detections; one example also reaches the default 300-detection ceiling, and counting error has not been quantified.
- The original environment has not yet been reconstructed from a clean installation, so dependency compatibility may require adjustment.

## Documentation Revision

Earlier project descriptions used the phrase "90%+ detection accuracy" and attributed an approximately 15% improvement to Test-Time Augmentation. Those statements were removed in August 2026 because the saved evidence supports the validation metrics reported above, while the reviewed inference call does not establish that TTA was activated or that counting accuracy was measured. This revision corrects the documentation; it does not represent a new model-training run.

## Acknowledgments

- [Oil Palm Detection dataset](https://universe.roboflow.com/manfred-michael/oil-palm-detection/dataset/6) published on Roboflow Universe.
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the object-detection framework and pretrained weights.
- ChatGPT was used to assist with code structuring, debugging, and documentation. The author executed the project and is responsible for the repository's final content and stated limitations.

## License

This repository is released under the [MIT License](LICENSE).
