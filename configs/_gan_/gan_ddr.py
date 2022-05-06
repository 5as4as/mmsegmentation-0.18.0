_base_ = [
    'datasets/gan_dataset.py',
    '../_base_/models/my_gan.py',
    '../_base_/default_runtime_gan.py'
]

num_classes = 4

model = dict(
    generator=dict(
         use_sigmoid=True,
         backbone=dict(
             embed_dims=96,
             depths=[2, 2, 6, 2],
             num_heads=[3, 6, 12, 24],
             window_size=7,
             use_abs_pos_embed=False,
             drop_path_rate=0.3,
             patch_norm=True),
         decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=num_classes),
         auxiliary_head=dict(in_channels=384, num_classes=num_classes)
    ),
    discriminator=dict(in_channels=num_classes),
    auxiliary_discriminator=dict(
        type='Discriminator',
        in_channels=num_classes)
)

train_cfg = dict(_delete_=True, lamda=0.01, lamda_aux=0.002, source='ddr')

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

load_generator = 'work_dirs/upernet_swin_tiny_patch4_window7_512x512_40k_ddr_pretrain_224x224_1K_512x512/iter_40000.pth'
