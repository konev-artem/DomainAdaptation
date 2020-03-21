from os.path import join, exists
import tensorflow as tf


class Resnet101:
    def __init__(self, path='.', model_fname='resnet101.h5'):
        self._path = path
        self._model_fname = model_fname
        self.need_to_save = False
        self.model = None

    def get_model(self):
        if self.model is not None:
            return self.model
        if exists(join(self._path, self._model_fname)):
            self.model = tf.keras.models.load_model(join(self._path, self._model_fname))
        else:
            self.model = tf.keras.applications.ResNet101()
            self.need_to_save = True
        return self.model

    def __del__(self):
        if self.need_to_save:
            self.model.save(join(self._path, self._model_fname))