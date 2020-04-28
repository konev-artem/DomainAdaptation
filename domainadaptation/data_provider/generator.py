import os

import numpy as np
import tensorflow as tf

from PIL import Image
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


class Dataset:

    def __init__(self, root, img_size, store_in_ram):
        """
        Arguments:

            :param root: (str) path to root
            :param img_size: (int) to what size to resize the images
            :param store_in_ram: (bool) whether to store in RAM the data
        """

        self.root = root
        self.img_size = img_size
        self.store_in_ram = store_in_ram
        self.classes = sorted(os.listdir(root))
        self.class_map = dict(zip(self.classes, range(len(self.classes))))

        self._dataset = []
        self._classes = []

        for class_ in self.classes:
            for filename in sorted(os.listdir(os.path.join(root, class_))):

                path = os.path.join(self.root, class_, filename)

                if self.store_in_ram:
                    img = self._read_img(path=path, img_size=self.img_size)
                    self._dataset.append(img)
                else:
                    self._dataset.append(path)

                self._classes.append(self.class_map[class_])

    @staticmethod
    def _read_img(path, img_size):
        """
        Reads image from path, resizes to (img_size x img_size)
        and convert to numpy array

        Arguments:

            :param path: (str) path to image
            :param img_size: (int) to what size to resize the images
        """

        img = Image.open(path).resize((img_size, img_size))
        return np.array(img)

    def __len__(self):

        return len(self._dataset)

    def __getitem__(self, idx):

        assert idx < len(self)

        img = self._dataset[idx] if self.store_in_ram else \
            self._read_img(self._dataset[idx], self.img_size)

        return img, self._classes[idx]


class MaskedDataLoader:

    def __init__(self, dataset, mask, batch_size, preprocess_input=lambda x: x / 255.0):
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

    def update_mask(self, mask):

        assert len(self.mask) == len(mask), 'Wrong length'

        self.mask = mask

    def __iter__(self):

        indices = np.random.permutation(len(self.dataset))
        x_batch, y_batch = [], []

        for ind in indices:

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
