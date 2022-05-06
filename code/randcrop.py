import os
import cv2
from mmseg.datasets import ODOCDataset

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomCrop', crop_size=(512, 512))
]
img_dir='train/images/all'
ann_dir='train/ann'
data_root='../data/FOVCrop-padding/REFUGE-FOVCrop-padding'
output_dir = 'img'

os.makedirs(output_dir, exist_ok=True)

dataset = ODOCDataset(pipeline=train_pipeline, img_dir=img_dir, img_suffix='.jpg', ann_dir=ann_dir, data_root=data_root)

for i in range(len(dataset)):
    result = dataset[i]
    img = result['img']
    filename = result['ori_filename']
    cv2.imwrite(os.path.join(output_dir, filename), img)