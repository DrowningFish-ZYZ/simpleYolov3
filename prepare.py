"""
created: 2022-3-13
数据制作模块:
    - 制作指定 anchors
    - 制作指定 classes
    - 制作指定 train & val
"""
from xml.etree import ElementTree

import os
import numpy as np


def make_anchors(anchors: list):
    """ 将先验框存入文本 """
    with open('./infos/anchors.txt', 'w') as file:
        for i in range(len(anchors)):
            if i != len(anchors) - 1:
                info = str(anchors[i][0]) + ',' + str(anchors[i][1]) + \
                       ',' + ' ' * 2
            else:
                info = str(anchors[i][0]) + ',' + str(anchors[i][1])
            file.write(info)


def make_classes(classes: list):
    """ 将类别存入文本 """
    with open('./infos/classes.txt', 'w') as file:
        for c in classes:
            file.write(f'{c}\n')


def parse_conn(f_name, file):
    """ 解析 xml 文件, 存入 file 中 """
    with open(f'./dataset/annotations/{f_name}', 'r') as fp:
        tree = ElementTree.parse(fp)
        image_path = tree.find('path').text.split('\\')[-1]
        file.write(f'dataset/images/{image_path}')
        for obj in tree.getroot().iter('object'):
            difficult = obj.find('difficult').text
            c = obj.find('name').text
            if c not in classes_ or int(difficult) == 1: continue
            c_id = classes_.index(c)
            boxes = obj.find('bndbox')
            xmin = boxes.find('xmin').text
            ymin = boxes.find('ymin').text
            xmax = boxes.find('xmax').text
            ymax = boxes.find('ymax').text
            file.write(f' {xmin},{ymin},{xmax},{ymax},{c_id}')
        file.write('\n')



def make_dataset(train_percent=.8, val_percent=.2):
    """
    提取注解文件, 将其内容存入txt
    如: image_path x,y,w,h,label,
    将数据分为3份:
        - 训练集: train.txt
        - 测试集_1: val.txt
    """

    # 获取所有注解文件, 并且打乱, 切割为 2 份
    ann_files = os.listdir('./dataset/annotations')
    np.random.shuffle(ann_files)
    train_set = ann_files[:int(len(ann_files) * train_percent)]
    val_set = ann_files[int(len(ann_files) * train_percent):]

    with open('./infos/train.txt', 'w') as train_file:
        for train in train_set:
            parse_conn(train, train_file)

    with open('./infos/val.txt', 'w') as val_file:
        for val in val_set:
            parse_conn(val, val_file)



if __name__ == '__main__':
    """ 
    注意: 
        - 只支持 xml 文件的注解
        -  注解文件请放在 ./dataset/annotations/
        -  图片请放在 ./dataset/images/
    """

    # 大中小三组先验框
    anchors_ = [(10, 13), (16, 30), (33, 23),
                (30, 61), (62, 45), (59, 119),
                (116, 90), (156, 198), (373, 326)]

    # 类别合集
    classes_ = ['anime face']

    print('===================== 正在制作: anchors, classes, train, val =====================')
    make_anchors(anchors=anchors_)
    make_classes(classes=classes_)
    make_dataset()
    print('===================== [OK] =====================')
