from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from domainadaptation.trainer import Trainer
from domainadaptation.visualizer import Visualizer
from domainadaptation.data_provider import DomainGenerator


class Experiment:
    class BackboneType(str, Enum):
        ALEXNET = "alexnet"
        VGG16 = "vgg16"
        RESNET50 = "resnet50"
        RESNET101 = "resnet101"

        def __str__(self):
            return self.value

    def __init__(self, config):
        self.config = config

        self._kwargs_for_backbone = {
            'include_top': False,
            'weights': config['backbone']['weights'],
            'input_shape': (*config['backbone']['img-size'], 3),
            'pooling': config['backbone']['pooling'],
        }

        if config["backbone"]["type"] == self.BackboneType.ALEXNET:
            raise NotImplementedError
        elif config["backbone"]["type"] == self.BackboneType.VGG16:
            self._backbone_class = keras.applications.vgg16.VGG16
            preprocess_input = keras.applications.vgg16.preprocess_input
        elif config["backbone"]["type"] == self.BackboneType.RESNET50:
            self._backbone_class = keras.applications.resnet.ResNet50
            preprocess_input = keras.applications.resnet.preprocess_input
        elif config["backbone"]["type"] == self.BackboneType.RESNET101:
            self._backbone_class = keras.applications.resnet.ResNet101
            preprocess_input = keras.applications.resnet.preprocess_input
        else:
            raise ValueError("Not supported backbone type")

        self.domain_generator = DomainGenerator(config["dataset"]["path"],
                                                preprocessing_function=preprocess_input,
                                                **config["dataset"]["augmentations"])

    def _get_new_backbone_instance(self, **kwargs):
        if kwargs:
            new_kwargs = self._kwargs_for_backbone.copy()
            new_kwargs.update(kwargs)
            instance = self._backbone_class(**new_kwargs)
        else:
            instance = self._backbone_class(**self._kwargs_for_backbone)

        assert self.config['backbone']['num_trainable_layers'] >= -1
        assert type(self.config['backbone']['num_trainable_layers']) == int

        if self.config['backbone']['num_trainable_layers'] != -1:
            num_non_trainable_layers = len(instance.layers) - self.config['backbone']['num_trainable_layers']
            for layer in instance.layers[:num_non_trainable_layers]:
                layer.trainable = False

        return instance

    @staticmethod
    def _cross_entropy(model, x_batch, y_batch, head):

        assert head in [0, 1], "wrong head number"

        logits = model(x_batch)[head]
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_batch, logits))

    @staticmethod
    def _domain_wrapper(generator, domain=0):
        """ Changes y_batch from class labels to source/target domain label """

        assert domain in [0, 1], "wrong domain number"

        for X_batch, _ in generator:
            y_batch = np.zeros(shape=(X_batch.shape[0], 2))
            y_batch[:, domain] = 1
            yield X_batch, tf.convert_to_tensor(y_batch)

    @staticmethod
    def _get_classifier_head(num_classes):

        return keras.models.Sequential([
            keras.layers.Dense(units=1024, activation='relu'),
            keras.layers.Dense(units=1024, activation='relu'),
            keras.layers.Dense(units=num_classes)
        ])
