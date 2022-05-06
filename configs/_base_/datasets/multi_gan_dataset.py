# dataset settings

img_scale = (512, 512)
crop_size = (512, 512)
img_norm_cfg_refuge = dict(
    mean=[69.563, 41.422, 20.689], std=[46.321, 26.691, 13.643], to_rgb=True)

train_pipeline = [
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
        type='Normalize', **img_norm_cfg_refuge),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
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
            dict(type='Normalize', **img_norm_cfg_refuge),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_REFUGE_train = dict(
    img_dir='train/images/all',
    ann_dir='train/ann',
    data_root='../data/FOVCrop-padding/REFUGE-FOVCrop-padding',
    type='ODOCDataset',
    pipeline=train_pipeline)

dataset_DDR_train = dict(
    img_dir='train/images',
    ann_dir='train/ann',
    data_root='../data/FOVCrop-padding/DDR-FOVCrop-padding',
    type='LesionDataset',
    pipeline=train_pipeline
)

dataset_refuge_test = dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='../data/FOVCrop-padding/REFUGE-FOVCrop-padding',
    type='ODOCDataset',
    pipeline=test_pipeline)

dataset_ddr_test=dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='../data/FOVCrop-padding/DDR-FOVCrop-padding',
    type='LesionDataset',
    pipeline=test_pipeline)

dataset_IDRID_test = dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='../data/FOVCrop-padding/IDRiD-FOVCrop-padding',
    type='LesionDataset',
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train_source=dataset_REFUGE_train,
    train_target=dataset_DDR_train,
    val=[
        dataset_refuge_test,
        dataset_ddr_test
    ],
    test=dataset_IDRID_test)
