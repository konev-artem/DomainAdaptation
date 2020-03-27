from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

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
