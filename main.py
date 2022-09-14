from matplotlib import pyplot as plt
from model import get_model
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    TimeDistributed,
    Dropout,
    Input,
    Dense,
    BatchNormalization,
    GRU,
    Layer,
    Flatten,
    MaxPooling2D,
    concatenate,
    Lambda,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from keras import layers
from keras import models
from tensorflow.python.keras.layers import ConvLSTM2D

from data_loader import DataLoader


def main():
    gpu_available = tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )
    model_type = "2D"
    train_loader, val_loader = DataLoader(
        "train", squeeze=model_type == "2D"
    ).get_loaders()

    x, y = next(train_loader)

    print(f"\n gpu_available={gpu_available}")
    print(f"model_type={model_type}")
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    dil_rate = (2, 2)

    model = get_model(x.shape, model_type)

    from keras import backend as K

    # batch_size  x * y * n_channels
    def dice_coefficient(y_true, y_pred, smooth=0.0001):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        # print(y_true_f.shape, y_pred_f.shape)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth
        )

    def dice_coefficient_loss(y_true, y_pred):
        return 1.0 - dice_coefficient(y_true, y_pred)

    def iou_loss(y_true, y_pred):
        return 1 - iou(y_true, y_pred)

    def iou(y_true, y_pred):
        # ipdb.set_trace()
        try:
            intersection = K.sum(K.abs(y_true) * (y_true))
        except:
            ipdb.set_trace()
        sum_ = K.sum(K.square(y_true)) + K.sum(K.square(y_pred))
        jac = (intersection) / (sum_ - intersection)
        return jac

    LR = 5e-4
    optim = tf.keras.optimizers.Adam(learning_rate=LR)

    metrics = [iou, dice_coefficient, "binary_accuracy"]

    model.compile(optimizer=optim, loss=dice_coefficient_loss, metrics=metrics)
    model.summary()
    steps_per_epoch = 1
    val_steps_per_epoch = 10
    history = model.fit(
        train_loader,
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        verbose=1,
        validation_data=val_loader,
        validation_steps=val_steps_per_epoch,
        batch_size=10,
    )

    x, y = val_generator.__next__()

    y_pred = model.predict(x)
    plt.imshow(y_pred.squeeze()[0])

    plt.imshow(y.squeeze()[0])

    plt.imshow(x[0, :, :, -1])


if __name__ == "__main__":
    main()
