from os.path import join, exists
import tensorflow as tf

from tensorflow.keras.layers import Flatten
from tensorflow.keras import Sequential

class BaseFabric:
    def __init__(self, model_name=None, model_fname='basemodel', path='.', flatten=True):
        self._path = path
        self._model_fname = model_fname
        self.need_to_save = False
        self.model = None
        self.model_name = model_name
        self.flatten = flatten

    def get_model(self, *args, **kwargs):
        if "include_top" not in kwargs.keys():
            kwargs['include_top'] = False

        if self.model is not None:
            return self.model
        if exists(join(self._path, self._model_fname)):
            self.model = tf.keras.models.load_model(join(self._path, self._model_fname))
        else:
            self.model = self.model_name(*args, **kwargs)
            self.need_to_save = True

        if self.flatten:
            temp = self.model
            self.model = Sequential()
            self.model.add(temp)
            self.model.add(Flatten())
        return self.model

    def get_path(self):
        return self._path


    def __del__(self):
        if self.need_to_save:
            self.model.save(join(self._path, self._model_fname))


class Resnet101Fabric(BaseFabric):
    def __init__(self, **kwargs):
        super().__init__(model_name=tf.keras.applications.ResNet101, model_fname='resnet101', **kwargs)

class VGG19Fabric(BaseFabric):
    def __init__(self, **kwargs):
        super().__init__(model_name=tf.keras.applications.VGG19, model_fname='vgg19', **kwargs)

class Resnet50Fabric(BaseFabric):
    def __init__(self, **kwargs):
        super().__init__(model_name=tf.keras.applications.ResNet50, model_fname='resnet50', **kwargs)

