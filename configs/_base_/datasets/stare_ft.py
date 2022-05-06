# dataset settings

dataset_type = 'vesselDataset'
data_root = '../data/FOVCrop-padding/STARE-FOVCrop-padding'

img_norm_cfg = dict(
    mean=[145.820, 79.810, 25.016], std=[89.677, 47.413, 19.985], to_rgb=True)

image_scale = (512, 512)

classes = ('background', 'vessel')

palette = [
    [0, 0, 0],
    [128, 0, 0],
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=image_scale, ratio_range=(0.8, 1.2)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate', prob=1, degree=(-15, -11.25, -7.5, -3.75, 0, 3.75, 7.5, 11.25, 15)),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_DRIVE_test = dict(
    img_dir='test/images',
    ann_dir='test/ann-1st',
    data_root='../data/FOVCrop-padding/DRIVE-FOVCrop-padding',
    type=dataset_type,
    pipeline=test_pipeline,
    classes=classes,
    palette=palette,
    img_suffix='.tif')

dataset_STARE_train = dict(
    img_dir='images/train',
    ann_dir='ann-ah/train',
    data_root=data_root,
    type=dataset_type,
    pipeline=train_pipeline,
    classes=classes,
    palette=palette)

dataset_STARE_test = dict(
    img_dir='images/test',
    ann_dir='ann-ah/test',
    data_root=data_root,
    type=dataset_type,
    pipeline=test_pipeline,
    classes=classes,
    palette=palette)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dataset_STARE_train,
    val=dataset_STARE_test,
    test=dataset_STARE_test)
