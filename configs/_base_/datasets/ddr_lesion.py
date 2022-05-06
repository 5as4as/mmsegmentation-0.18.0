# dataset settings
dataset_type = 'LesionDataset'
data_root = '../data/FOVCrop-padding/DDR-FOVCrop-padding'

img_norm_cfg = dict(
    mean=[119.003, 74.719, 31.546], std=[67.947, 44.335, 20.600], to_rgb=True)
crop_size = (1024, 1024)
img_scale = (1024, 1024)


palette = [
    [0, 0, 0],
    [128, 0, 0],  # EX: red
    [0, 128, 0],  # HE: green
    [128, 128, 0],  # SE: yellow
    [0, 0, 128]  # MA: blue
]
classes = ['bg', 'EX', 'HE', 'SE', 'MA']
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=1, degree=(0, 90, 180, 270)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # TODO
        img_scale=img_scale,
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        img_dir='train/images',
        ann_dir='train/ann',
        data_root=data_root,
        classes=classes,
        palette=palette,
        type=dataset_type,
        pipeline=train_pipeline),
    val=dict(
        img_dir='val/images',
        ann_dir='val/ann',
        data_root=data_root,
        classes=classes,
        palette=palette,
        type=dataset_type,
        pipeline=test_pipeline),
    test=dict(
        img_dir='test/images',
        ann_dir='test/ann',
        data_root=data_root,
        classes=classes,
        palette=palette,
        type=dataset_type,
        pipeline=test_pipeline))