_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/refuge.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    type='EncoderDecoder_multi',
    use_sigmoid=True,
    pretrained='pretrain/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=2, loss_decode=dict(type='BinaryLoss', loss_weight=1.0, loss_type='dice', smooth=1e-5)),
    auxiliary_head=dict(in_channels=384, num_classes=2, loss_decode=dict(type='BinaryLoss', loss_weight=0.4, loss_type='dice', smooth=1e-5)))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU

# dataset settings

dataset_type = 'ODOCDataset'
data_root = '../data/FOVCrop-padding/REFUGE-FOVCrop-padding'
#img_norm_cfg = dict(mean=[102.994, 69.034, 56.513], std=[70.871, 50.111, 40.183], to_rgb=True)
img_norm_cfg = dict(mean=[69.563, 41.422, 20.689], std=[46.321, 26.691, 13.643], to_rgb=True)
#img_norm_cfg = dict(mean=[131.844, 76.464, 38.214], std=[53.581, 35.279, 17.195], to_rgb=True)
image_scale = (1024, 1024)

classes = ('background', 'rim', 'optim cup', 'optim disk')

palette = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=image_scale,ratio_range=(0.8,1.2)),
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

dataset_REFUGE_test = dict(
    img_dir='test/images',
    ann_dir='test/ann',
    data_root=data_root,
    type=dataset_type,
    pipeline=test_pipeline,
    classes=classes,
    palette=palette,
    use_sigmoid=True)

dataset_REFUGE_val = dict(
    img_dir='val/images',
    ann_dir='val/ann',
    data_root=data_root,
    type=dataset_type,
    pipeline=test_pipeline,
    classes=classes,
    palette=palette,
    use_sigmoid=True)

dataset_REFUGE_train=dict(
    img_dir='train/images/all',
    ann_dir='train/ann',
    data_root=data_root,
    type=dataset_type,
    pipeline=train_pipeline,
    classes=classes,
    palette=palette,
    use_sigmoid=True)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dataset_REFUGE_train,
    val=dataset_REFUGE_val,
    test=dataset_REFUGE_test)
