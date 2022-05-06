_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/drive_ft.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_5k.py'
]
model = dict(
    type='EncoderDecoder_multi',
    AUC = True,
    pretrained='pretrain/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=2),
    auxiliary_head=dict(in_channels=384, num_classes=2))

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
data = dict(samples_per_gpu=1)

train_pipeline = [
    _delete_=True,
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(540, 540), ratio_range=(0.8, 1.2)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate', prob=0.5, degree=(-15, -11.25, -7.5, -3.75, 3.75, 7.5, 11.25, 15)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
