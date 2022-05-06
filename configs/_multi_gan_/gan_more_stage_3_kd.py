_base_ = [
    '../_base_/datasets/multi_gan_dataset.py',
    '../_base_/models/multi_task_gan_kd.py',
    '../_base_/default_runtime_gan.py'
]

model = dict(
    generator=dict(
        backbone=dict(
            start_more_stage=1,
        )
    )
)

train_cfg = dict(_delete_=True, ODOC_lamda=0.01, ODOC_lamda_aux=0.002, lesion_lamda=0.01, lesion_lamda_aux=0.002)

# define optimizer
optimizer = dict(
    generator=dict(
        type='AdamW',
        lr=0.000006,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys={
                'absolute_pos_embed': dict(decay_mult=0.),
                'relative_position_bias_table': dict(decay_mult=0.),
                'norm': dict(decay_mult=0.)
            })),
    discriminator=dict(
        type='Adam', lr=0.0001, betas=(0.9, 0.99)),
    auxiliary_discriminator=dict(
        type='Adam', lr=0.0001, betas=(0.9, 0.99))
)

lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=0.0,
    by_epoch=False)

evaluation = dict(interval=2000, metric='mIoU', by_epoch=False)

checkpoint_config = dict(interval=2000, by_epoch=False)

total_iters = 20000

load_generator = 'work_dirs/upernet_swin_tiny_patch4_window7_512x512_40k_multi_pretrain_224x224_1K_more_stage_3/iter_40000.pth'
