# simpleYolov3
简单使用yolov3

## 1.环境安装
Anconda 
  ### 注意：具体的 cuda，cudnn 和 tensorflow  根据自己的显卡来选择对应版本！
  ```python
  dawdaw
  ```
  1. conda create --name TF2.1 python==3.7
  2. conda activate TF2.1
  3. conda install cudatoolkit=11
  4. conda install cudnn=7.6
  5. pip install tensorflow==2.6
  
 ## 2.使用方式
 1. 将图片数据(*.jpg)与标注数据(*.xml)分别放置在 dataset/images dataset/annotations 下
 2. 使用 prepare.py 构建你的训练集文件，测试集文件，类别文件，先验框文件，infos/(train.txt, text.txt, classes.txt, anchors.txt)
 3. 使用 train.py 里定义好的代码去训练你的模型, 具体配置都以写出, 你只需要对其数据微调
 4. 使用 predict.py 里的 Predict 类去对，图片，视频，摄像头识别
