from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DomainGenerator:
    def __init__(self, source_dir, target_dir, target_size=(100, 100)):
        self.datagen = ImageDataGenerator()
        self.target_size = target_size
        self.source_gen = self.datagen.flow_from_directory(source_dir)
        self.target_gen = self.datagen.flow_from_directory(target_gen)
        
    def source_generator():
        return self.source_gen
    
    def target_generator():
        return self.target_gen
