import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import numpy as np

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

class VariableLengthIterator:
    def __init__(self, image_iterator, class_id, class_cnt):
        self.image_iterator = image_iterator
        self.class_id = class_id
        self.class_cnt = class_cnt

    def next(self, count):
        res_x = []
        for i in range(count):
            step = self.image_iterator.next()
            res_x.append(step[0])
        return np.concatenate(res_x), to_categorical(np.full(len(res_x), self.class_id), self.class_cnt)

class ClassSpecificGenerator(DomainGenerator):
    def __init__(self, dataset_dir, **image_data_generator_kwargs):
        super().__init__(dataset_dir, **image_data_generator_kwargs)

        self.class_indices = self.get_class_indices()

    def get_class_indices(self):
        g = self.make_generator(self.get_domain_names()[0])
        return g.class_indices

    def get_class_iterator(self, domain, class_name, **generator_kwargs):
        generator_kwargs['batch_size'] = 1
        generator_kwargs['classes'] = [class_name]
        dir = self.domains[domain]
        gen = self.datagen.flow_from_directory(dir, **generator_kwargs)
        return VariableLengthIterator(gen, self.class_indices[class_name], len(self.class_indices))
