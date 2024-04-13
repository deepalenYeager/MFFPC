model = dict(
    type='MFFPC',
    backbone=dict(
        type='deformable_resnet18'
    ),
    neck=dict(
        type='FPN',
    ),
    detection_head=dict(
        type='MFFPC_head',
        config='config/MFFPC/nas-configs/MFFPC_base.config',
        pooling_size=9,
        dropout_ratio=0.1,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_emb=dict(
            type='EmbLoss',
            feature_dim=4,
            loss_weight=0.25
        )
    )
)
repeat_times = 10
data = dict(
    batch_size=16,
    train=dict(
        type='MFFPC_MSRA',
        split='train',
        is_transform=True,
        img_size=736,
        short_size=736,
        pooling_size=9,
        read_type='cv2',
        repeat_times=repeat_times
    ),
    test=dict(
        type='MFFPC_MSRA',
        split='test',
        short_size=736,
        read_type='pil'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=600 // repeat_times,
    optimizer='Adam',
    #pretrain='',
    
    save_interval=10 // repeat_times,
)
test_cfg = dict(
    min_score=0.89,
    min_area=250,
    bbox_type='rect',
    result_path='outputs/submit_msra/'
)