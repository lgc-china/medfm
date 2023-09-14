import mmcv
import numpy as np
import os
import torch
from argparse import ArgumentParser
from mmcls.apis import inference_model, init_model
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


def inference_model(model, img):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
    return scores


def main(img_file, img_path, checkpoint, config, output_prediction):
    parser = ArgumentParser()
    # parser.add_argument('-img_file', required=False, default='/home/lgc/dataset/MedFMC/stage2/MedFMC_test/chest/test_WithoutLabel.txt', help='Names of test image files')
    # parser.add_argument('-img_path', required=False, default='/home/lgc/dataset/MedFMC/stage2/MedFMC_test/chest/images/', help='Path of test image files')

    parser.add_argument('-img_file', required=False,default=img_file,help='Names of test image files')
    parser.add_argument('-img_path', required=False,default=img_path,help='Path of test image files')

    #exp1
    # parser.add_argument('-config', required=False, default='/home/lgc/pycharm_code/MedFM-main/configs/swin-b_vpt/in21k-swin-b_vpt5_bs4_lr5e-2_10-shot_chest_adamw.py', help='Config file')
    # parser.add_argument('-checkpoint', required=False, default='/home/lgc/pycharm_code/MedFM-main/tools/work_dirs/exp1/in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest/best_mAP_epoch_17.pth', help='Checkpoint file')
    # parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument('-output-prediction', required=False,help='where to save prediction in csv file',default='/home/lgc/pycharm_code/MedFM-main/tools/result/exp1/chest_10-shot_submission.csv')

    #exp2
    # parser.add_argument('-config', required=False,default='/home/lgc/pycharm_code/MedFM-main/configs/swin-b_vpt/exp2/in21k-swin-b_vpt5_bs4_lr5e-2_10-shot_chest_adamw.py',help='Config file')
    # parser.add_argument('-checkpoint', required=False,default='/home/lgc/pycharm_code/MedFM-main/tools/work_dirs/exp2/in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest/best_mAP_epoch_20.pth',help='Checkpoint file')
    # parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument('-output-prediction', required=False, help='where to save prediction in csv file',default='/home/lgc/pycharm_code/MedFM-main/tools/result/exp2/chest_10-shot_submission.csv')

    #exp3
    # parser.add_argument('-config', required=False,default='/home/lgc/pycharm_code/MedFM-main/configs/swin-b_vpt/exp3/in21k-swin-b_vpt5_bs4_lr5e-2_10-shot_chest_adamw.py',help='Config file')
    # parser.add_argument('-checkpoint', required=False,default='/home/lgc/pycharm_code/MedFM-main/tools/work_dirs/exp3/in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest/best_mAP_epoch_20.pth',help='Checkpoint file')
    # parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument('-output-prediction', required=False, help='where to save prediction in csv file',default='/home/lgc/pycharm_code/MedFM-main/tools/result/exp3/chest_10-shot_submission.csv')

    #exp4
    # parser.add_argument('-config', required=False,default='/home/lgc/pycharm_code/MedFM-main/configs/swin-b_vpt/exp4/in21k-swin-b_vpt5_bs4_lr5e-2_10-shot_chest_adamw.py',help='Config file')
    # parser.add_argument('-checkpoint', required=False,default='/home/lgc/pycharm_code/MedFM-main/tools/work_dirs/exp4/in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest/best_mAP_epoch_20.pth',help='Checkpoint file')
    # parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument('-output-prediction', required=False, help='where to save prediction in csv file',default='/home/lgc/pycharm_code/MedFM-main/tools/result/exp4/chest_10-shot_submission.csv')

    #exp5
    parser.add_argument('-config', required=False,default=config,help='Config file')
    parser.add_argument('-checkpoint', required=False,default=checkpoint,help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('-output-prediction', required=False, help='where to save prediction in csv file',default=output_prediction)



    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a bundle of images
    if args.output_prediction:
        with open(args.output_prediction, 'w') as f_out:
            for line in open(args.img_file, 'r'):
                image_name = line.split('\n')[0]
                file = os.path.join(args.img_path, image_name)
                result = inference_model(model, file)[0]
                f_out.write(image_name)
                for j in range(len(result)):
                    f_out.write(',' + str(np.around(result[j], 8)))
                f_out.write('\n')


if __name__ == '__main__':
    #代码逻辑调试

    #模拟参数
    # a = '3'
    # idir = '/home/lgc/pycharm_code/MedFM-main/tools/work_dirs/exp3/in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest'
    # i = 'in21k-swin-b_vpt-5_bs4_lr0.05_10-shot_chest'
    # files = os.listdir(idir)
    # files.sort()
    #
    # dataset = i.split('_')[-1]
    # nshot = i.split('_')[-2]
    # n = nshot.split('-')[0]
    # #定义超残
    # img_file = '/home/lgc/dataset/MedFMC/stage2/MedFMC_test/{}/test_WithoutLabel.txt'.format(dataset)
    # img_path = '/home/lgc/dataset/MedFMC/stage2/MedFMC_test/{}/images/'.format(dataset)
    # checkpoint = os.path.join(idir, files[1])
    # config = os.path.join(idir, files[3])
    # output_prediction = '/home/lgc/pycharm_code/MedFM-main/tools/result/exp{}/{}_{}-shot_submission.csv'.format(a,dataset,n)
    # #打印超残
    # print('lgc')
    # print(img_file)
    # print(img_path)
    # print(checkpoint)
    # print(config)
    # print(output_prediction)


    #代码逻辑
    result = '../configs/result/'
    if os.path.isdir(result)==False:
        os.mkdir(result)

    for a in range(1, 6):
        dir = '../configs/work_dirs/exp{}/'.format(a)
        ins = os.listdir(dir)
        for i in ins:
            idir = os.path.join(dir, i)
            files = os.listdir(idir)
            inpy = None
            for f in files:
                if 'in21k' in f:
                    inpy = f
            #获取小参数
            dataset = inpy.split('_')[-2]
            nshot = inpy.split('_')[-3]
            n = nshot.split('-')[0]
            #定义超参
            img_file = '../configs/data/MedFMC_test/{}/test_WithoutLabel.txt'.format(dataset)
            img_path = '../configs/data/MedFMC_test/{}/images/'.format(dataset)

            pth = None
            for f in files:
                if 'best' in f:
                    pth = f
            checkpoint = os.path.join(idir, pth)
            config = os.path.join(idir, inpy)
            rdir = '../configs/result/exp{}/'.format(a)
            if os.path.isdir(rdir) == False:
                os.mkdir(rdir)
            output_prediction = '../configs/result/exp{}/{}_{}-shot_submission.csv'.format(a,dataset,n)

            main(img_file, img_path, checkpoint, config, output_prediction)

            # print(config)
            # print(inpy.split('_'))
            # 
            # print(img_file)
            # print(img_path)
            # print(checkpoint)
            # print(output_prediction)
            # print('-------')


    # main()
