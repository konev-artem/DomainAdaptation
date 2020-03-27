from os.path import join, exists
import tensorflow as tf


class BaseFabric:
    def __init__(self, model_name=None, model_fname='basemodel.h5', path='.'):
        self._path = path
        self._model_fname = model_fname
        self.need_to_save = False
        self.model = None
        self.model_name = model_name

    def get_model(self, *args, **kwargs):
        if self.model is not None:
            return self.model
        if exists(join(self._path, self._model_fname)):
            self.model = tf.keras.models.load_model(join(self._path, self._model_fname))
        else:
            self.model = self.model_name(*args, **kwargs)
            self.need_to_save = True
        return self.model

    def __del__(self):
        if self.need_to_save:
            self.model.save(join(self._path, self._model_fname))


class Resnet101Fabric(BaseFabric):
    def __init__(self, **kwargs):
        super().__init__(model_name=tf.keras.applications.ResNet101, model_fname='resnet101.h5', **kwargs)

