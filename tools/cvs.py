import csv
import random
import os
import shutil
if __name__ == "__main__":
    name = 'in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest_adamw-exp2.py'
    n = name.split("_")
    print(n)


    # data = 'chest'
    # with open('/home/lgc/dataset/MedFMC/MedFMC_train_v2/MedFMC_train/{}/{}_train.csv'.format(data, data)) as f:
    #     fcsv = csv.reader(f)
    #
    #     headers = next(fcsv)
    #
    #     with open('/home/lgc/dataset/MedFMC/MedFMC_train_v2/MedFMC_train/{}/trainval.txt'.format(data), 'w') as tv:
    #         for row in fcsv:
    #             tv.write(row[1])
    #             # tv.write(' ')
    #             label = ' '
    #             # print(len(row))
    #             # print(len(row[2:]))
    #             # print(type(row[2]))
    #             # print(row[2][0])
    #             for a in row[3:]:
    #                 label = label + str(a[0]) + ' '
    #             # print(label)
    #             # print(label[:-1])
    #             # label.replace(',', ' ')
    #             tv.write(label[:-1])
    #             # print(label)
    #             tv.write('\n')
    #
    #
    #
    #
    # #
    # with open('/home/lgc/dataset/MedFMC/MedFMC_train_v2/MedFMC_train/{}/trainval.txt'.format(data), 'r') as f:
    #     lists = f.readlines()
    #     random.shuffle(lists)
    #     c = len(lists)
    #     train = lists[:int(c*0.8)]
    #     val = lists[int(c*0.8):]
    #     with open('/home/lgc/dataset/MedFMC/MedFMC_train_v2/MedFMC_train/{}/train_20.txt'.format(data), 'w') as tr:
    #         for t in train:
    #             tr.write(t)
    #             # tr.write('\n')
    #
    #     with open('/home/lgc/dataset/MedFMC/MedFMC_train_v2/MedFMC_train/{}/val_20.txt'.format(data), 'w') as va:
    #         for v in val:
    #             va.write(v)
    #             # va.write('\n')
    # d = '/home/lgc/pycharm_code/MedFM-main/configs/swin-b_vpt/exp1'
    # d2 = '/home/lgc/pycharm_code/MedFM-main/configs/swin-b_vpt2'
    # names = os.listdir(d)
    #
    #
    # for a in range(1, 6):
    #     for n in names:
    #         new = n.split('.')[0]
    #         new_n = new + '-' + 'exp' + str(a) + '.py'
    #         pt_old = os.path.join(d, n)
    #         pt_new = os.path.join(d2, new_n)
    #
    #         # pt_old = os.path.join(d, n)
    #         # pt_new = os.path.join(d, new)
    #         #
    #         shutil.copy(pt_old, pt_new)
    #
    # # curPath = os.path.abspath(os.path.dirname(__file__))
    # # rootPath = curPath[:curPath.find("MedFM-main") + len("MedFM-main")]  # 获取myProject，也就是项目的根路径
    # # print(rootPath)
    # curPath = os.path.abspath(os.path.dirname(__file__))
    # rootPath = curPath[:curPath.find("MedFM-main") + len("MedFM-main")]
    # print(rootPath)
    # dataset = 'chest'
    # exp_num = 1
    # nshot = 1
    # # a = os.path.join(rootPath, '/data/MedFMC/{}/{}_{}-shot_train_exp{}.txt'.format(dataset, dataset, nshot, exp_num))
    # a = rootPath + f'/data/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt'
    # print(a)
