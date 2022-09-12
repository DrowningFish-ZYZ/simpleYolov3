# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:00:56 2018

@author: wmy
"""


from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input

import os
import cv2
import utils
import model
import colorsys
import numpy as np
import tensorflow as tf


class Predict:

    def __init__(
            self, classes: list, anchors: np.ndarray,
            model_path: str, font_path: str,
            iuo=0.4, score=0.25):
        """
        初始化预测器
        :param classes: 类别
        :param anchors: 先验框
        :param model_path: 模型权重路径
        :param font_path: 字体路径
        """
        # 兼容 tensorflow v1 和 v2
        tf.compat.v1.disable_v2_behavior()

        # 基础配置
        self.classes_names = classes
        self.anchors = anchors
        self.model_path = model_path
        self.font_path = font_path
        self.model_image_size = (416, 416)

        # 其它
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.iou = iuo
        self.score = score
        self.boxes, self.scores, self.classes, self.model, self.colors, self.input_image_shape = self._generate()

    def _generate(self):
        """
        生成器, 根据模型, classes_name, anchors
        生成: 对应的特征图像框, 准确率, classes_id
        :return: boxes, scores, classes, yolo_model, colors, input_image_shape
        """

        num_anchors = len(self.anchors)
        num_classes = len(self.classes_names)
        is_tiny_version = num_anchors == 6

        # 创建模型加载权重
        yolo_model = model.tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes) if is_tiny_version else model.yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
        yolo_model.load_weights(self.model_path)

        # 生成绘制边框的颜色
        hsv_tuples = [(x / len(self.classes_names), 1., 1.) for x in range(len(self.classes_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        # 固定种子, 打乱边框颜色
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)

        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = model.yolo_eval(
            yolo_model.output, self.anchors,
            len(self.classes_names), input_image_shape,
            score_threshold=self.score, iou_threshold=self.iou)

        return boxes, scores, classes, yolo_model, colors, input_image_shape

    def image(self, image: Image.Image) -> Image.Image:
        """
        对图像进行预测
        :param image: 图像
        :return: images
        """

        # 将图片缩放为 32 的倍数
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = model.letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = model.letterbox_image(image, new_image_size)

        # 归一化, 添加批次维度, 即: shape(w, h, 3) => shape(b, w, h, 3)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        # 得到预测框, 预测真实度, 预测类别
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # 添加字体和厚度
        font = ImageFont.truetype(
            font=self.font_path,
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32') )
        thickness = (image.size[0] + image.size[1]) // 300

        # 将得到的数组, 添加到原图上去
        for i, c in reversed(list(enumerate(out_classes))):
            # 边框和准确率展示
            predicted_class = self.classes_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # 标签展示
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            # 还原边框到原图的四个角
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image

    def video(self, video_path: str, out_path: str):
        """
        对视频进行图像识别: 将视频逐帧抽取, 通过open_cv工具库,
        送入网络识别, 再逐帧还原
        :param video_path: 视频路径
        :param out_path: 还原后的视频路径
        """
        out_path = './a.mkv'.split('/')
        out_path[-1] = out_path[-1].split('.')[0]
        out_path = '/'.join(out_path) + '.mp4'

        # 获取视频对象
        v = cv2.VideoCapture(video_path)
        # 获取视频字节码, fps, 宽高
        video_fourcc = int(v.get(cv2.CAP_PROP_FOURCC))
        video_fps = v.get(cv2.CAP_PROP_FPS)
        video_size = (int(v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # 输出对象
        out = cv2.VideoWriter(out_path, video_fourcc, video_fps, video_size)

        accum_time = 0
        prev_time = timer()
        fps = 'FPS: ??'
        curr_fps = 0
        # 逐帧制作
        print('============================ 正在逐帧制作, 请耐心等待 ============================')
        while True:
            return_value, name = v.read()
            if return_value:
                image = Image.fromarray(name)
                image = self.image(image)
                result = np.asarray(image)

                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps += 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = 'FPS: ' + str(curr_fps)
                    curr_fps = 0

                cv2.putText(result, text=fps, org=(3, 15),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=.5, color=(255, 0, 0), thickness=2)

                out.write(result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            else:
                break

        self.sess.close()

    def camera(self):
        """ 摄像头实时测 """
        v = cv2.VideoCapture(0)

        while True:
            revalue, name = v.read()
            if revalue:
                image = Image.fromarray(name)
                image = self.image(image)
                result = np.asarray(image)
                cv2.imshow("result", result)
                cv2.waitKey(1)

            else:
                break


# ============================================================= 预测功能 =============================================================
if __name__ == '__main__':

    images = os.listdir("./dataset/images/")
    np.random.shuffle(images)
    img = Image.open(f'./dataset/images/{images[0]}')

    anchors_ = utils.load_anchors('./infos/anchors.txt')
    classes_ = utils.load_classes('./infos/classes.txt')

    predict = Predict(
        classes=classes_, anchors=anchors_,
        model_path='./models_t/anime_weights.h5',
        font_path='./font/msyh.ttc')

    predict.image(img).save('b.jpg')  # 识别某个图片
    # predict.video('./2022-03-13 19-18-48.mkv', './a.mp4')  识别视频
    # predict.camera()  开启摄像头识别