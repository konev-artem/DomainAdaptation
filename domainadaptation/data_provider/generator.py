import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DomainGenerator:
    # you can use image_data_generator_kwargs for augmentation
    def __init__(self, dataset_dir, **image_data_generator_kwargs):
        self.datagen = ImageDataGenerator(**image_data_generator_kwargs)
        self.domains = dict()
        with os.scandir(dataset_dir) as it:
            for entry in it:
                if entry.is_dir():
                    self.domains[entry.name] = entry.path

    def get_domain_names(self):
        return list(self.domains.keys())

    def make_generator(self, domain, **generator_kwargs):
        dir = self.domains[domain]
        return self.datagen.flow_from_directory(dir, **generator_kwargs)


class MaskedGenerator:

    def __init__(self, dataset, mask, batch_size, preprocess_input=lambda x: x / 255.0,
                 flip_horizontal=False, random_crop=(224, 224)):
        """
        Arguments:

            :param dataset: (Dataset) dataset of images to sample from
            :param mask: (array) array of bools with the same len as the dataset
            :param batch_size: (int) batch size
            :param preprocess_input: (lambda) additional function to preprocess input
        """

        assert len(dataset) == len(mask), 'Dataset and mask should have the same length'

        self.dataset = dataset
        self.mask = mask
        self.batch_size = batch_size
        self.preprocess_input = preprocess_input

        self.flip_horizontal = flip_horizontal
        self.random_crop = random_crop

    def set_mask(self, mask):

        assert len(self.mask) == len(mask), 'Wrong length'

        self.mask = mask

    def _transform(self, x_batch, y_batch):

        x_batch, y_batch = np.concatenate(x_batch, axis=0), np.array(y_batch)

        if self.preprocess_input is not None:
            x_batch = self.preprocess_input(x_batch)

        x_batch, y_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32), tf.convert_to_tensor(y_batch, dtype=tf.int32)

        if self.flip_horizontal:
            x_batch = tf.image.random_flip_left_right(x_batch)

        if self.random_crop is not None:
            x_batch = tf.map_fn(lambda img: tf.image.random_crop(img, [*self.random_crop, 3]), x_batch)

        return x_batch, y_batch

    def get_batch(self, classes):
        """ Get batch of given classes according to mask """

        samples_per_batch = self.batch_size // len(classes)

        x_batch, y_batch = [], []

        for i, class_ in enumerate(classes):
            mask = self.dataset.class_to_mask[class_] * self.mask
            indices = np.argwhere(mask).flatten()
            indices = np.random.choice(indices, size=min(samples_per_batch, len(indices)), replace=False)

            for index in indices:
                img, target = self.dataset[index]
                x_batch.append(img[np.newaxis, ...])
                y_batch.append(target)

        return self._transform(x_batch, y_batch)

    def __iter__(self):
        """ Iterates through the whole dataset """

        x_batch, y_batch = [], []

        for ind in range(len(self.dataset)):

            img, target = self.dataset[ind]
            x_batch.append(img[np.newaxis, ...])
            y_batch.append(target)

            if len(x_batch) >= self.batch_size:
                yield self._transform(x_batch, y_batch)
                x_batch, y_batch = [], []

        yield self._transform(x_batch, y_batch)
