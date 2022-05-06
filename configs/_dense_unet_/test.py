_base_ = [
    '../_base_/models/dense_unet.py', '../_base_/datasets/idrid_lesion.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    type='EncoderDecoder_multi',
    use_sigmoid=True,
    compute_aupr=True,
    pretrained='pretrain/densenet161.pth',
    backbone=dict(arch='161'),
    decode_head=
        dict(in_channels=[384, 768, 2112, 2208],
             num_classes=4,
             channels=32,
             loss_decode=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[0.001, 0.1, 0.1, 0.1, 1.0])
        ),
    # auxiliary_head=dict(in_channels=384, num_classes=4, loss_decode=dict(type='BinaryLoss', loss_weight=0.4, loss_type='dice', smooth=1e-5))
)

test_cfg = dict(mode='slide', crop_size=512, stride=350)

optimizer = dict(lr=0.001)
lr_config = dict(policy='poly', power=0.9, min_lr=0, by_epoch=False)

