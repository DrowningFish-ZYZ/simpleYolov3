from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

import utils

"""
# created: 2022-3-13
# 模型训练模块, 在这里, 你可以使用自己的数据集来训练模型
# 重新训练模型学习率建议调为: 0.001
# 加载权重训练模型, 学习率建议为: lr <= 0.0001
"""

if __name__ == '__main__':

    batch_size = 2
    input_shape_ = (416, 416)
    anchors, classes, trains, vals = utils.load_config(
        anchors_path='./infos/anchors.txt',
        classes_path='./infos/classes.txt',
        train_path='./infos/train.txt',
        val_path='./infos/val.txt'
    )

    model = utils.create_model(
        input_shape=input_shape_,
        anchors=anchors,
        num_classes=len(classes),
        load_pretrained=True,
        load_model_path='./models/weights.h5'
    )

    model.compile(
        optimizer=optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss={'yolo_loss': lambda y_true, y_pred: y_pred},
    )

    save_model = callbacks.ModelCheckpoint(
        './models/weights.h5',
        save_weights_only=True,
        save_best_only=True
    )

    history = model.fit(
        utils.generator_wrap(trains, batch_size, input_shape_, anchors, len(classes)),
        steps_per_epoch=max(1, len(trains) // batch_size),
        validation_data=utils.generator_wrap(vals, batch_size, input_shape_, anchors, len(classes)),
        validation_steps=max(1, len(vals) // batch_size),
        epochs=200,
        initial_epoch=0,
        callbacks=[save_model]
    )

    utils.save_loss(yolo_model=model, history=history, path='./models/')