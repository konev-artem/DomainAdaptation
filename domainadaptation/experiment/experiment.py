from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from ..trainer import Trainer
from ..tester import Tester
from ..data_provider import DomainGenerator
from ..models import Alexnet, Vgg16, Resnet50, Resnet101, GradientReversal


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

        self.img_width = config["dataset"]["img_width"]
        self.img_height = config["dataset"]["img_height"]

        if config["backbone"]["type"] == self.BackboneType.ALEXNET:
            self.backbone = Alexnet().get_model()
        elif config["backbone"]["type"] == self.BackboneType.VGG16:
            self.backbone = Vgg16().get_model()
        elif config["backbone"]["type"] == self.BackboneType.RESNET50:
            self.backbone = Resnet50().get_model()
        elif config["backbone"]["type"] == self.BackboneType.RESNET101:
            self.backbone = Resnet101().get_model()
        else:
            raise ValueError("Not supported backbone type")

        self.domain_generator = DomainGenerator(
            source_dir=config["source_dataset"],
            target_dir=config["target_dataset"],
            target_size=(self.img_width, self.img_height)
        )

    @staticmethod
    def _get_classifier_head(num_classes):
        return keras.layers.Dense(units=num_classes)


class DANNExperiment(Experiment):
    """
    Domain-Adversarial Training of Neural Networks
    link: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, config):
        super().__init__(config)

        # -- create model --
        classifier_head = self._get_classifier_head(num_classes=config["classes"])
        domain_head = self._get_classifier_head(num_classes=2)
        gradient_reversal_layer = GradientReversal(self._get_lambda())

        self.model = keras.Model(
            inputs=self.backbone.inputs,
            outputs=[
                classifier_head(self.backbone.outputs),
                domain_head(gradient_reversal_layer(self.backbone.outputs))
            ]
        )

    def __call__(self):
        def cross_entropy(model, x_batch, y_batch):
            y_one_hot = tf.one_hot(y_batch, depth=self.config["classes"])
            logits = model(x_batch)
            return tf.nn.softmax_cross_entropy_with_logits(y_one_hot, logits)

        # -- training strategy --
        trainer = Trainer()
        adam = keras.optimizers.Adam()
        for _ in range(self.config["epochs"]):
            trainer.train(model=self.model, compute_loss=cross_entropy, optimizer=adam,  # callbacks=''
                          train_generator=self.domain_generator.source_generator(), steps=self.config["steps"])

        # -- evaluation --
        tester = Tester()
        tester.test(self.model, self.domain_generator.target_generator())

        # -- visualization --
        pass

    @staticmethod
    def _get_lambda(p=0):
        """ original lambda scheduler """
        return 2 / (1 + np.exp(-10 * p)) - 1
