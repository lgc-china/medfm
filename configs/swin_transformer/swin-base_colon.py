# Only for evaluation
_base_ = [
    '../_base_/models/swin_transformer/base_384.py',
    '../_base_/datasets/colon.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py', '../_base_/custom_imports.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/lgc/pycharm_code/MedFM-main/pretrain/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=2),
)
