# dataset settings

img_scale = (512, 512)
crop_size = (512, 512)
img_norm_cfg_aria = dict(
    mean=[103.422, 37.727, 9.108], std=[85.485, 36.581, 10.273], to_rgb=True)
img_norm_cfg_refuge = dict(
    mean=[69.563, 41.422, 20.689], std=[46.321, 26.691, 13.643], to_rgb=True)
img_norm_cfg_ddr = dict(
    mean=[119.003, 74.719, 31.546], std=[67.947, 44.335, 20.600], to_rgb=True)
img_norm_cfg_idrid = dict(
    mean=[121.037, 58.632, 16.929], std=[78.433, 40.494, 13.205], to_rgb=True)

vessel_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.8, 1.2)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomRotate',
        prob=1,
        degree=(-15, -11.25, -7.5, -3.75, 0, 3.75, 7.5, 11.25, 15)),
    dict(
        type='Normalize', **img_norm_cfg_aria),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
refuge_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.8, 1.2)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomRotate',
        prob=1,
        degree=(-15, -11.25, -7.5, -3.75, 0, 3.75, 7.5, 11.25, 15)),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize', **img_norm_cfg_refuge),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
ddr_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=1, degree=(0, 90, 180, 270)),
    dict(
        type='Normalize', **img_norm_cfg_ddr),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
idrid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomRotate',
        prob=1,
        degree=(-45, 45)),
    dict(
        type='Normalize', **img_norm_cfg_idrid),
    dict(type='Pad', size=img_scale, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomRotate',
        prob=1,
        degree=(-45, 45)),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg_refuge),
    dict(type='Instance_Normalize'),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]


aria_test_pipeline = [
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
            dict(type='Normalize', **img_norm_cfg_aria),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

refuge_test_pipeline = [
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

ddr_test_pipeline = [
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

idrid_test_pipeline = [
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
            dict(type='Normalize', **img_norm_cfg_idrid),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
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
            # dict(type='Normalize', **img_norm_cfg_idrid),
            dict(type='Instance_Normalize'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


dataset_ARIA_train = dict(
    type='RepeatDataset',
    times=3,
    dataset=dict(
        img_dir='images',
        ann_dir='ann-BDP',
        data_root='../data/FOVCrop-padding/ARIA-FOVCrop-padding',
        type='vesselDataset',
        pipeline=train_pipeline,
        img_suffix='.tif')
)

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
dataset_IDRID_train = dict(
    img_dir='train/images',
    ann_dir='train/ann',
    data_root='../data/FOVCrop-padding/IDRiD-FOVCrop-padding',
    type='LesionDataset',
    pipeline=train_pipeline
)

dataset_DRIVE_test = dict(
    img_dir='test/images',
    ann_dir='test/ann-1st',
    data_root='../data/FOVCrop-padding/DRIVE-FOVCrop-padding',
    type='vesselDataset',
    pipeline=test_pipeline,
    img_suffix='.tif',
    use_sigmoid=True)

dataset_STARE_test = dict(
    img_dir='images/test',
    ann_dir='ann-ah/test',
    data_root='../data/FOVCrop-padding/STARE-FOVCrop-padding',
    type='vesselDataset',
    pipeline=test_pipeline,
    use_sigmoid=True)

dataset_refuge_test = dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='../data/FOVCrop-padding/REFUGE-FOVCrop-padding',
    type='ODOCDataset',
    pipeline=test_pipeline,
    use_sigmoid=False)

dataset_ddr_test=dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='../data/FOVCrop-padding/DDR-FOVCrop-padding',
    type='LesionDataset',
    pipeline=test_pipeline)

dataset_idrid_test=dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root='../data/FOVCrop-padding/IDRiD-FOVCrop-padding',
    type='LesionDataset',
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=[
        dataset_ARIA_train,
        dataset_REFUGE_train,
        dataset_DDR_train,
        dataset_IDRID_train
    ],
    val=[
        dataset_DRIVE_test,
        dataset_STARE_test,
        dataset_refuge_test,
        dataset_ddr_test
    ],
    test=[
        dataset_DRIVE_test,
        dataset_STARE_test,
        dataset_refuge_test,
        dataset_ddr_test
    ])


