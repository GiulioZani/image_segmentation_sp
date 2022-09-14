from matplotlib import pyplot as plt
from data_loader import get_generators
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
from tensorflow.keras.optimizers import Adam
from keras import layers
from keras import models
from tensorflow.python.keras.layers import ConvLSTM2D


def main():
    gpu_available = tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )
    print(f"\n gpu_available={gpu_available}\n")
    train_generator, val_generator = get_generators()
    """
    x, y = train_generator.__next__()
    print(x.shape)
    print(y.shape)
    for i in range(0, 3):
        image = x[i, -1, :, :, 0]
        # mask = np.argmax(y[i], axis=2)
        mask = y[i, -1, :, :, 0]
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.show()

    x.shape

    y.shape

    val_generator = Gen(val_generator)
    x, y = val_generator.__next__()

    for i in range(0, 3):
        image = x[i, 0, :, :, 0]
        # mask = np.argmax(y[i], axis=2)
        mask = y[i, 0, :, :, 0]
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.show()

    # Define the model metrcis and load model.
    """
    # num_train_imgs = len(os.listdir("/gdrive/MyDrive/train/train_images/images"))
    # num_val_images = len(os.listdir("/gdrive/MyDrive/train/val_images/images"))
    # steps_per_epoch = num_train_imgs // batch_size
    # val_steps_per_epoch = num_val_images // batch_size
    """
    batch=x.shape[0]
    steps=x.shape[1]
    IMG_HEIGHT = x.shape[2]
    IMG_WIDTH  = x.shape[3]
    IMG_CHANNELS = x.shape[4]
    input_shape = (3, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    print(input_shape)
    """

    dil_rate = (2, 2)

    model = get_model()

    from keras import backend as K

    # batch_size  x * y * n_channels
    def dice_coefficient(y_true, y_pred, smooth=0.0001):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        # print(y_true_f.shape, y_pred_f.shape)
        try:
            intersection = K.sum(y_true_f * y_pred_f)
        except:
            ipdb.set_trace()
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
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        verbose=1,
        validation_data=val_generator,
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
