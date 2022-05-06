# dataset settings

img_scale = (512, 512)
crop_size = (512, 512)
img_norm_cfg_ddr = dict(
    mean=[119.003, 74.719, 31.546], std=[67.947, 44.335, 20.600], to_rgb=True)

source_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomRotate',
        prob=1,
        degree=(-15, -11.25, -7.5, -3.75, 0, 3.75, 7.5, 11.25, 15)),
    # dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize', **img_norm_cfg_ddr),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

target_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=img_scale),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomRotate',
        prob=1,
        degree=(-15, -11.25, -7.5, -3.75, 0, 3.75, 7.5, 11.25, 15)),
    # dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize', **img_norm_cfg_ddr),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
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
            dict(type='Normalize', **img_norm_cfg_ddr),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_DDR_train = dict(
    img_dir='train/images',
    ann_dir='train/ann',
    data_root='../data/FOVCrop-padding/DDR-FOVCrop-padding',
    type='LesionDataset',
    pipeline=source_pipeline)

dataset_IDRID_train = dict(
    imgs_root='../data/FOVCrop-padding/IDRiD-FOVCrop-padding/train/images',
    type='GANDataset',
    pipeline=target_pipeline
)

dataset_IDRID_test = dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='../data/FOVCrop-padding/IDRiD-FOVCrop-padding',
    type='LesionDataset',
    pipeline=test_pipeline)

dataset_DDR_test = dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='../data/FOVCrop-padding/DDR-FOVCrop-padding',
    type='LesionDataset',
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train_source=dataset_DDR_train,
    train_target=dataset_IDRID_train,
    val=[
        dataset_IDRID_test,
        dataset_DDR_test
    ],
    test=dataset_IDRID_test
)
