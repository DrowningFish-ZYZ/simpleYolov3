"""
created: 2022-3-13
工具类模块:
    - anchors 的加载
    - classes 的加载
    - train, val 的加载
    - 数据集的制作
    - 模型的初始化创建
    - 训练可视化, 参数可视化
"""

from matplotlib import pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

import model
import numpy as np

# ==================================================== 配置加载 ====================================================
def load_config(
        anchors_path: str,
        classes_path: str,
        train_path: str,
        val_path: str) -> tuple:

    """
    加载配置
    :param anchors_path: 先验框的路径
    :param classes_path: 类别路径
    :param train_path: 训练集路径
    :param val_path: 测试集路径
    :return: anchors, classes, train_files, val_files
    """

    with open(anchors_path) as file:
        anchors = [float(anchor.strip()) for anchor in file.readline().split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    with open(classes_path, 'r', encoding='utf-8') as file:
        classes = [c.strip() for c in file.readlines()]

    with open(train_path) as file:
        trains = file.readlines()

    with open(val_path) as file:
        vals = file.readlines()

    return anchors, classes, trains, vals


def load_anchors(path: str) -> np.ndarray:
    with open(path) as file:
        anchors = [float(anchor.strip()) for anchor in file.readline().split(',')]
        return np.array(anchors).reshape(-1, 2)


def load_classes(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as file:
        return [c.strip() for c in file.readlines()]

# ==================================================== 数据集生成 ====================================================
def generator(
        annotation_lines: list,
        batch_size: int,
        input_shape: tuple,
        anchors: np.ndarray,
        num_classes: int) -> tuple:

    """
    制作数据集
    :param annotation_lines: 注解文件列表
    :param batch_size: 批次
    :param input_shape: 输入形状: (416, 416)
    :param anchors: 先验框
    :param num_classes: 类别数量
    """

    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image_data, box_data = list(), list()
            image, box = model.get_random_data(
                annotation_lines[i],
                input_shape,
                random=True)
            image_data.append(image)
            box_data.append(box)
            i += 1

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = model.preprocess_true_boxes(
            box_data,
            input_shape,
            anchors,
            num_classes)

        yield [image_data, *y_true], np.zeros(batch_size)


def generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """ 数据集过滤 """
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


# =================================================== 模型构建 ====================================================
def create_model(
        input_shape: tuple,
        anchors: np.ndarray,
        num_classes: int,
        load_model_path='./models/weights.h5',
        load_pretrained=False,
        freeze_body=False) -> Model:

    """
    构建yolo模型
    :param input_shape: 输入形状
    :param anchors: 先验框
    :param num_classes: 类别数
    :param load_model_path: 模型加载的位置
    :param load_pretrained: 是否加载模型权重
    :param freeze_body:
    """

    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = model.yolo_body(image_input, num_anchors // 3, num_classes)

    if load_pretrained: model_body.load_weights(load_model_path, by_name=True, skip_mismatch=True)

    model_loss = Lambda(model.yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})([*model_body.output, *y_true])
    return Model([model_body.input, *y_true], model_loss)


# =================================================== 可视化模型 ====================================================
def save_loss(yolo_model: Model, history, path: str):
    # 保存模型参数
    with open(path + 'weights.txt', 'w') as fp:
        for v in yolo_model.trainable_variables:
            fp.write(str(v.name) + '\n')  # 打印当前网络层名字
            fp.write(str(v.shape) + '\n')  # 当前网络层形状
            fp.write(str(v.numpy()) + '\n')  # 当前网络层的所有参数

    # 训练集 loss
    loss = history.history['loss']

    # 测试集 loss
    val_loss = history.history['val_loss']

    # 绘制训练集和验证集的 acc 和 loss 曲线
    plt.subplot(1, 2, 1)  # 将画布分割为 1 行 2 列, 当前在第 1 列
    plt.plot(loss, label='Train loss')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_loss, label='Test loss')
    plt.grid()
    plt.legend()

    plt.savefig(path + 'loss.jpg')