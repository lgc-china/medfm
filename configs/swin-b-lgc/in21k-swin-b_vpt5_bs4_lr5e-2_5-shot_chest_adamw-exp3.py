_base_ = [
    '/home/lgc/pycharm_code/MedFM-main/configs/_base_/datasets/chest.py',
    '/home/lgc/pycharm_code/MedFM-main/configs/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '/home/lgc/pycharm_code/MedFM-main/configs/_base_/default_runtime.py',
    '/home/lgc/pycharm_code/MedFM-main/configs/_base_/custom_imports.py',
]


lr = 5e-3
n = 1
vpl = 5
dataset = 'chest'
exp_num = 3
nshot = 5
run_name = f'in21k-swin-b_vpt-{vpl}_bs4_lr{lr}_{nshot}-shot_{dataset}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedSwinTransformer',
        prompt_length=vpl,
        arch='base',
        img_size=384,
        stage_cfgs=dict(block_cfgs=dict(window_size=12))),
    neck=None,
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=1024,
    ))
data = dict(
    samples_per_gpu=4,  # use 2 gpus, total 128
    train=dict(
        ann_file=
        f'../configs/data/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt'
    ),
    val=dict(
        ann_file=
        f'../configs/data/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
    test=dict(ann_file=f'../configs/data/MedFMC/{dataset}/test_WithLabel.txt'))

optimizer = dict(lr=lr)

log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])
load_from = 'pth'
work_dir = f'../configs/work_dirs/exp{exp_num}/{run_name}'

runner = dict(type='EpochBasedRunner', max_epochs=30)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
