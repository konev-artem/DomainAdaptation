import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DomainGenerator:
    # you can use image_data_generator_kwargs for augmentation
    def __init__(self, dataset_dir, preprocessing_function, **image_data_generator_kwargs):
        self.datagen = ImageDataGenerator(preprocessing_function=preprocessing_function, **image_data_generator_kwargs)
        self.datagen_no_augm = ImageDataGenerator(preprocessing_function=preprocessing_function)
        self.domains = dict()
        with os.scandir(dataset_dir) as it:
            for entry in it:
                if entry.is_dir():
                    self.domains[entry.name] = entry.path

    def get_domain_names(self):
        return list(self.domains.keys())

    def make_generator(self, domain, use_augmentation=True, **generator_kwargs):
        dir = self.domains[domain]
        dg = self.datagen if use_augmentation else self.datagen_no_augm
        return dg.flow_from_directory(dir, **generator_kwargs)

