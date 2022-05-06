_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/ddr_lesion.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    type='EncoderDecoder_multi',
    use_sigmoid = True,
    compute_aupr = True,
    pretrained='pretrain/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=4, loss_decode=dict(type='BinaryLoss', loss_weight=1.0, loss_type='dice', smooth=1e-5)),
    auxiliary_head=dict(in_channels=384, num_classes=4, loss_decode=dict(type='BinaryLoss', loss_weight=0.4, loss_type='dice', smooth=1e-5)))

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
    dict(type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=0),
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
