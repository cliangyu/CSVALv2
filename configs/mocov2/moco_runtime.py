_base_ = [
    '../_base_/schedules/medmnist_sgd_coslr_200e.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MoCo',
    queue_len=65536,
    feat_dim=128,
    momentum=0.001,
    backbone=dict(
        type='ResNet',
        depth=50,
        norm_cfg=dict(type='BN'),
        zero_init_residual=False),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2))

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
)

# only keeps the latest 3 checkpoints
default_hooks = dict(logger=dict(type='LoggerHook', interval=20),
                     checkpoint=dict(max_keep_ckpts=3))
custom_hooks = [dict(type='InfoNCEHook', priority='ABOVE_NORMAL')]


# # NOTE: `auto_scale_lr` is for automatically scaling LR
# # based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)