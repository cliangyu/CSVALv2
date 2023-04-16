# dataset settings
dataset_type = 'BloodMNIST'
N = 105
img_norm_cfg = dict(mean=[.5, .5, .5], std=[.5, .5, .5])


data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    to_rgb=True,
    **img_norm_cfg
    )

view_pipeline = [
    dict(type='RandomResizedCrop', scale=28, backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        prob=0.8),
    dict(type='RandomGrayscale', prob=0.2, keep_channels=True, channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='GaussianBlur', magnitude_range=(0.1, 2.0),
        magnitude_std='inf', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
]

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackInputs', meta_keys=['medmnist_idx', 'sample_idx', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction']),
]

# dataset summary

train_dataloader = dict(
    # batch_size=4096,
    batch_size=64,
    num_workers=8,
    drop_last=True,
    sampler=dict(type='RepeatAugSampler', num_repeats=N, shuffle=True),
    dataset=dict(
            type=dataset_type,
            data_prefix='data/medmnist',
            split='train',
            pipeline=train_pipeline,
        ))