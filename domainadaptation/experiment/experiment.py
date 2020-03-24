from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from ..tester import Tester
from ..trainer import Trainer
from ..visualizer import Visualizer
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

        self.domain_generator = DomainGenerator(config["dataset"]["path"],
                                                **config["dataset"]["augmentations"])

    @staticmethod
    def _cross_entropy(model, x_batch, y_batch):
        logits = model(x_batch)
        return tf.nn.softmax_cross_entropy_with_logits(y_batch, logits)

    @staticmethod
    def _domain_wrapper(generator, domain=0):
        """ Changes y_batch from class labels to source/target domain label """

        assert domain in [0, 1], "wrong domain number"

        while True:
            for X_batch, _ in generator:
                y_batch = np.zeros(shape=(X_batch.shape[0], 2))
                y_batch[:, domain] = 1
                yield X_batch, tf.convert_to_tensor(y_batch)

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
        classifier_head = self._get_classifier_head(num_classes=config["dataset"]["classes"])
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

        # -- initialization --
        optimizer = keras.optimizers.Adam()
        source_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]
        )
        target_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["target"],
            batch_size=self.config['batch_size'],
            target_size=self.config["backbone"]["img-size"]
        )

        # -- training strategy --
        trainer = Trainer()
        for _ in range(self.config["epochs"]):
            # TODO add callbacks

            for generator in [source_generator,
                              self._domain_wrapper(source_generator, domain=0),
                              self._domain_wrapper(target_generator, domain=1)]:
                trainer.train(
                    model=self.model,
                    compute_loss=self._cross_entropy,
                    optimizer=optimizer,  # the same optimizer ?
                    train_generator=generator,
                    steps=self.config["steps"]
                )

                # TODO somehow increase lambda in grad_rev_layer

        # -- evaluation --
        tester = Tester()
        tester.test(self.model, target_generator)

        # -- visualization --
        # visualizer = Visualizer(embeddings=X, domains=domains, labels=y)
        # visualizer.visualize(size=75)

    @staticmethod
    def _get_lambda(p=0):
        """ Original lambda scheduler """
        return 2 / (1 + np.exp(-10 * p)) - 1
