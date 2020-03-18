from enum import Enum

import tensorflow as tf
import tensorflow.keras as keras

from ..trainer import Trainer
from ..models import Alexnet, Vgg16, Resnet50
from ..data_provider import DomainGenerator


class Experiment:
    class BackboneType(str, Enum):
        ALEXNET = "alexnet"
        VGG16 = "vgg16"
        RESNET50 = "resnet50"

        def __str__(self):
            return self.value

    def __init__(self, config):
        self.config = config

        self.img_width = config["dataset"]["img_width"]
        self.img_height = config["dataset"]["img_height"]

        input_size = (self.img_width, self.img_height, 3)
        if config["backbone"]["type"] == self.BackboneType.ALEXNET:
            self.backbone = Alexnet(input_size)
        elif config["backbone"]["type"] == self.BackboneType.VGG16:
            self.backbone = Vgg16(input_size)
        elif config["backbone"]["type"] == self.BackboneType.RESNET50:
            self.backbone = Resnet50(input_size)
        else:
            raise ValueError("Not supported backbone type")

        self.domain_generator = DomainGenerator(
            source_dir=config["sourse_dataset"],
            target_dir=config["target_dataset"],
            target_size=(self.img_width, self.img_height)
        )


class DANNExperiment(Experiment):
    """
    Domain-Adversarial Training of Neural Networks
    link: https://arxiv.org/abs/1505.07818
    """

    class GradReverse(keras.layers.Layer):  # куда этот слой ???
        """
        Gradient reversal layer
        """

        def __init__(self):
            super().__init__()

        @staticmethod
        @tf.custom_gradient
        def grad_reverse(x):
            y = tf.identity(x)

            def custom_grad(dy):
                return -dy

            return y, custom_grad

        def call(self, x, **kwargs):
            return self.grad_reverse(x)

    def __init__(self, config):
        super().__init__(config)

        # -- create model --
        classifier_head = keras.layers.Dense(units=config["classes"])
        domain_head = keras.layers.Dense(units=2)
        gradient_reversal_layer = self.GradReverse()

        self.model = keras.Model(
            inputs=self.backbone.inputs,
            outputs=[
                classifier_head(self.backbone.outputs),
                domain_head(gradient_reversal_layer(self.backbone.outputs))
            ]
        )

    def __call__(self):
        def cross_entropy(model, x_batch, y_batch):  # куда эту функцию ?
            # y_batch one_hot or number of classes ???
            return tf.nn.softmax_cross_entropy_with_logits(y_batch, model(x_batch))

        # training strategy
        trainer = Trainer()
        trainer.train(
            model=self.model,
            compute_loss=cross_entropy,
            optimizer=keras.optimizers.Adam(),
            train_generator=self.domain_generator.source_generator(),
            steps=self.config["steps"],
            callbacks='???'  # какие колбэки используем, откуда их берем, где создаем ???
        )

        # evaluation
        # ???

        # something else
        # ???
