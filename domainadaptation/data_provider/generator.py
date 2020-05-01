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


class MaskedDataLoader:

    def __init__(self, dataset, mask, batch_size, preprocess_input=lambda x: x / 255.0):
        """
        Arguments:

            :param dataset: (Dataset) dataset of images to sample from
            :param mask: (array) array of bools with the same len as the dataset
            :param batch_size: (int) batch size
            :param preprocess_input: (lambda) additional function to preprocess input
            :param TODO
        """

        assert len(dataset) == len(mask), 'Dataset and mask should have the same length'

        self.dataset = dataset
        self.mask = mask
        self.batch_size = batch_size
        self.preprocess_input = preprocess_input

    def set_mask(self, mask):

        assert len(self.mask) == len(mask), 'Wrong length'

        self.mask = mask

    def get_batch(self, classes):

        samples_per_batch = self.batch_size // len(classes)

        x_batch, y_batch = [], []

        for i, class_ in enumerate(classes):
            indices = self.dataset.class_to_indices[class_]
            indices = np.random.permutation(indices)

            for index in indices:

                if self.mask[index]:
                    img, target = self.dataset[index]
                    x_batch.append(img[np.newaxis, ...])
                    y_batch.append(target)

                if len(x_batch) >= (i + 1) * samples_per_batch:
                    break

        return x_batch, y_batch

    def __call__(self):

        x_batch, y_batch = [], []

        for ind in range(len(self.dataset)):

            if self.mask[ind]:
                img, target = self.dataset[ind]
                x_batch.append(img[np.newaxis, ...])
                y_batch.append(target)

            if len(x_batch) >= self.batch_size:
                x_batch, y_batch = np.concatenate(x_batch, axis=0), np.array(y_batch)

                if self.preprocess_input is not None:
                    x_batch = self.preprocess_input(x_batch)

                yield tf.convert_to_tensor(x_batch), tf.convert_to_tensor(y_batch)
                x_batch, y_batch = [], []

        # TODO