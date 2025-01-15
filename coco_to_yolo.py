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
