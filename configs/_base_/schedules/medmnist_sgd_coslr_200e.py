# optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.48, weight_decay=1e-4, momentum=0.9))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        by_epoch=True,
        begin=5,
        end=200,
    )
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
