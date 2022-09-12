# https://github.com/kivijoshi/TimeDistributedImageDataGenerator
import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import types
import numpy as np
from tensorflow.keras.preprocessing.image import (
    array_to_img,
    img_to_array,
    load_img,
)
from tensorflow.keras.preprocessing.image import DirectoryIterator
from keras.preprocessing.image import ImageDataGenerator
from itertools import tee
from new_data_loader import DataLoader


class TimeDistributedImageDataGenerator(ImageDataGenerator):
    def __init__(
        self,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format="channels_last",
        validation_split=0.0,
        # interpolation_order=1,
        dtype="float32",
        time_steps=5,
    ):

        self.time_steps = time_steps

        super().__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=validation_split,
            # interpolation_order=interpolation_order,
            dtype=dtype,
        )

    """Takes the path to a directory & generates batches of augmented data.
        # Arguments
            directory: string, path to the target directory.
                It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM or TIF images
                inside each of the subdirectories directory tree
                will be included in the generator.
                See [this script](
                https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
                for more details.
            target_size: Tuple of integers `(height, width)`,
                default: `(256, 256)`.
                The dimensions to which all images found will be resized.
            color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
                Whether the images will be converted to
                have 1, 3, or 4 channels.
            classes: Optional list of class subdirectories
                (e.g. `['dogs', 'cats']`). Default: None.
                If not provided, the list of classes will be automatically
                inferred from the subdirectory names/structure
                under `directory`, where each subdirectory will
                be treated as a different class
                (and the order of the classes, which will map to the label
                indices, will be alphanumeric).
                The dictionary containing the mapping from class names to class
                indices can be obtained via the attribute `class_indices`.
            class_mode: One of "categorical", "binary", "sparse",
                "input", or None. Default: "categorical".
                Determines the type of label arrays that are returned:
                - "categorical" will be 2D one-hot encoded labels,
                - "binary" will be 1D binary labels,
                    "sparse" will be 1D integer labels,
                - "input" will be images identical
                    to input images (mainly used to work with autoencoders).
                - If None, no labels are returned
                  (the generator will only yield batches of image data,
                  which is useful to use with `model.predict_generator()`).
                  Please note that in case of class_mode None,
                  the data still needs to reside in a subdirectory
                  of `directory` for it to work correctly.
            batch_size: Size of the batches of data (default: 32).
            shuffle: Whether to shuffle the data (default: True)
                If set to False, sorts the data in alphanumeric order.
            seed: Optional random seed for shuffling and transformations.
            save_to_dir: None or str (default: None).
                This allows you to optionally specify
                a directory to which to save
                the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str. Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: One of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            follow_links: Whether to follow symlinks inside
                class subdirectories (default: False).
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataGenerator`.
            interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`,
                and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed,
                `"box"` and `"hamming"` are also supported.
                By default, `"nearest"` is used.
        # Returns
            A `DirectoryIterator` yielding tuples of `(x, y)`
                where `x` is a numpy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a numpy array of corresponding labels.
        """

    def flow_from_directory(
        self,
        directory,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
    ):

        return TimeDistributedDirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
        )


class TimeDistributedDirectoryIterator(DirectoryIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        TimeSteps = self.image_data_generator.time_steps
        batch_x = np.zeros(
            (len(index_array),) + (TimeSteps,) + self.image_shape, dtype=self.dtype,
        )  # KJ
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            for k in reversed(range(0, TimeSteps)):
                try:
                    img = load_img(
                        filepaths[j - k],
                        color_mode=self.color_mode,
                        target_size=self.target_size,
                        interpolation=self.interpolation,
                    )
                    x = img_to_array(img, data_format=self.data_format)
                except:
                    print("Unexpected error:", sys.exc_info())
                    pass
                # Pillow images should be closed after `load_img`,
                # but not PIL images.
                if hasattr(img, "close"):
                    img.close()
                if self.image_data_generator:
                    params = self.image_data_generator.get_random_transform(x.shape)
                    x = self.image_data_generator.apply_transform(x, params)
                    x = self.image_data_generator.standardize(x)
                batch_x[i][k] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = "{prefix}_{index}_{hash}.{format}".format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format,
                )
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == "input":
            batch_y = batch_x.copy()
        elif self.class_mode in {"binary", "sparse"}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == "categorical":
            batch_y = np.zeros(
                (len(batch_x), TimeSteps, len(self.class_indices)), dtype=self.dtype,
            )
            for i, n_observation in enumerate(index_array):
                for q in reversed(range(0, TimeSteps)):
                    batch_y[i, q, self.classes[n_observation - q]] = 1.0
        elif self.class_mode == "multi_output":
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == "raw":
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y[:, -1]
        else:
            return batch_x, batch_y[:, -1], self.sample_weight[index_array]


class OnlyIndexGenerator:
    def __init__(self, generator, index=2):
        self.generator = generator
        self.index = index

    def __iter__(self):
        return self

    def __next__(self):
        x, y = self.generator.__next__()
        y = y[:, self.index :]
        x = x.squeeze(-1).transpose(0, 2, 3, 1)
        y = y.squeeze(-1).transpose(0, 2, 3, 1)
        return x, y


def get_generators():
    train_img_path = "train/train_images"
    train_mask_path = "train/train_masks"

    val_img_path = "train/val_images"
    val_mask_path = "train/val_masks"

    seed = 42

    img_data_gen_args = dict(
        rescale=1.0 / 255,
        rotation_range=90,
        zoom_range=0.2,
        brightness_range=[0.3, 0.9],
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.5,
        time_steps=3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="constant",
    )

    mask_data_gen_args = dict(
        rotation_range=90,
        zoom_range=0.2,
        brightness_range=[0.3, 0.9],
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.5,
        time_steps=3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="constant",
        preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype),
    )  # Binarize the output again.

    image_data_generator = TimeDistributedImageDataGenerator(**img_data_gen_args)
    mask_data_generator = TimeDistributedImageDataGenerator(**mask_data_gen_args)

    img_data_gen_args_val = dict(rescale=1.0 / 255, time_steps=3)

    mask_data_gen_args_val = dict(
        preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype),
        time_steps=3,
    )  # Binarize the output again.

    image_data_generator_val = TimeDistributedImageDataGenerator(
        **img_data_gen_args_val
    )
    mask_data_generator_val = TimeDistributedImageDataGenerator(
        **mask_data_gen_args_val
    )

    batch_size = 3

    image_generator = image_data_generator.flow_from_directory(
        train_img_path,
        seed=seed,
        batch_size=batch_size,
        color_mode="grayscale",
        target_size=(256, 256),
        class_mode=None,
    )  # Very important to set this otherwise it returns multiple numpy arrays
    # thinking class mode is binary.

    mask_generator = mask_data_generator.flow_from_directory(
        train_mask_path,
        seed=seed,
        batch_size=batch_size,
        color_mode="grayscale",
        target_size=(256, 256),  # Read masks in grayscale
        class_mode=None,
    )

    valid_img_generator = image_data_generator_val.flow_from_directory(
        val_img_path,
        seed=seed,
        batch_size=batch_size,
        color_mode="grayscale",
        target_size=(256, 256),
        class_mode=None,
    )  # Default batch size 32, if not specified here
    valid_mask_generator = mask_data_generator_val.flow_from_directory(
        val_mask_path,
        seed=seed,
        batch_size=batch_size,
        target_size=(256, 256),
        color_mode="grayscale",  # Read masks in grayscale
        class_mode=None,
    )  # Default batch size 32, if not specified here

    train_generator, train_generator2 = tee(zip(image_generator, mask_generator))
    val_generator, val_generator2 = tee(zip(valid_img_generator, valid_mask_generator))
    train_generator2 = OnlyIndexGenerator(train_generator2)
    val_generator2 = OnlyIndexGenerator(val_generator2)
    # asserts that the generators are the same
    for (x1, y1), (x2, y2) in zip(train_generator, train_generator2):
        y1 = y1[:, 2:]
        y1 = y1.squeeze(-1).transpose(0, 2, 3, 1)
        x1 = x1.squeeze(-1).transpose(0, 2, 3, 1)
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
    for (x1, y1), (x2, y2) in zip(val_generator, val_generator2):
        y1 = y1[:, 2:]
        y1 = y1.squeeze(-1).transpose(0, 2, 3, 1)
        x1 = x1.squeeze(-1).transpose(0, 2, 3, 1)
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
    print("success!")


if __name__ == "__main__":
    get_generators()
