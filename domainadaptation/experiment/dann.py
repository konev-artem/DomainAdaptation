import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from domainadaptation.tester import Tester
from domainadaptation.models import GradientReversal
from domainadaptation.experiment import Experiment
from domainadaptation.visualizer import Visualizer
from domainadaptation.utils import get_features_and_labels

import time

class DANNExperiment(Experiment):
    """
    Domain-Adversarial Training of Neural Networks
    link: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, config):
        super().__init__(config)

    def __call__(self, train_domain_head=True):

        #  --- create model ---
        backbone = self._get_new_backbone_instance()
        classifier_head = self._get_classifier_head(num_classes=self.config["dataset"]["classes"])
        domain_head = self._get_classifier_head(num_classes=2)

        lambda_ = tf.Variable(initial_value=0., trainable=False, dtype=tf.float32)
        gradient_reversal_layer = GradientReversal(lambda_)

        dann_model = keras.Model(inputs=backbone.inputs, outputs=[
            classifier_head(backbone.outputs[0]),
            domain_head(gradient_reversal_layer(backbone.outputs[0]))
        ])

        #  --- create generators ---
        source_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img_size"]
        )

        domain_0_generator = self._domain_wrapper(
            self.domain_generator.make_generator(
                domain=self.config["dataset"]["source"],
                batch_size=self.config["batch_size"],
                target_size=self.config["backbone"]["img_size"]
            ),
            domain=0
        )

        domain_1_generator = self._domain_wrapper(
            self.domain_generator.make_generator(
                domain=self.config["dataset"]["target"],
                batch_size=self.config["batch_size"],
                target_size=self.config["backbone"]["img_size"]
            ),
            domain=1
        )

        #  --- train dann model ---
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        steps_per_epoch = len(source_generator)

        for epoch_num in range(self.config["epochs"]):
            for step_during_epoch in range(steps_per_epoch):
                with tf.GradientTape() as tape:
                    x_batch, y_batch = next(source_generator)
                    classification_loss = self._cross_entropy(dann_model, x_batch, y_batch, head=0)

                    if train_domain_head:
                        x_batch, y_batch = next(domain_0_generator)
                        domain_loss = self._cross_entropy(dann_model, x_batch, y_batch, head=1)

                        x_batch, y_batch = next(domain_1_generator)
                        domain_loss += self._cross_entropy(dann_model, x_batch, y_batch, head=1)

                    if train_domain_head:
                        total_loss = classification_loss + (domain_loss / 2.)
                    else:
                        total_loss = classification_loss

                    grads = tape.gradient(total_loss, dann_model.trainable_variables)

                    optimizer.apply_gradients(zip(grads, dann_model.trainable_variables))

                    p_ = (steps_per_epoch * epoch_num + step_during_epoch) / (steps_per_epoch * self.config["epochs"])
                    lambda_.assign(DANNExperiment._get_lambda(p=p_))

                    if step_during_epoch % 50 == 0:
                        print('Progress: {}\nmean total loss:{}, lambda: {}'.format(p_, total_loss, lambda_.numpy()))
                        if train_domain_head:
                            print('classification loss: {}, domain_loss: {}'.format(classification_loss, domain_loss))

        #  --- test dann model ---
        classification_model = keras.Model(
            inputs=dann_model.inputs,
            outputs=dann_model.outputs[0]
        )

        tester = Tester()
        tester.test(classification_model, source_generator)
        
        target_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["target"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img_size"]
        )
        tester = Tester()
        tester.test(classification_model, target_generator)
        
        
        #  --- visualize features from the last layer of backbone ---
        source_features, source_labels = get_features_and_labels(backbone, source_generator, 50)
        target_features, target_labels = get_features_and_labels(backbone, target_generator, 50)
        
        visualizer = Visualizer(
            embeddings=np.vstack((source_features,
                                  target_features)),
            labels=np.hstack((source_labels,
                              target_labels)),
            domains=np.hstack((np.zeros(source_features.shape[0], dtype=int),
                               np.ones(target_features.shape[0], dtype=int))),
            **self.config['visualizer']
        )
        visualizer.visualize(**self.config['visualize'])

    @staticmethod
    def _get_lambda(p=0):
        """ Original lambda scheduler """
        return 2 / (1 + np.exp(-10 * p)) - 1

class DANNExperimentOffice31(Experiment):
    """
    Domain-Adversarial Training of Neural Networks
    link: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, config):
        super().__init__(config)

    def __call__(self, train_domain_head=True):

        #  --- create model ---
        backbone = self._get_new_backbone_instance()
        classifier_head = self._get_classifier_head(num_classes=self.config["dataset"]["classes"])
        domain_head = self._get_classifier_head(num_classes=2)

        lambda_ = tf.Variable(initial_value=0., trainable=False, dtype=tf.float32)
        gradient_reversal_layer = GradientReversal(lambda_)

        dann_model = keras.Model(inputs=backbone.inputs, outputs=[
            classifier_head(backbone.outputs[0]),
            domain_head(gradient_reversal_layer(backbone.outputs[0]))
        ])

        #  --- create generators ---
        source_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img_size"]
        )

        domain_0_generator = self._domain_wrapper(
            self.domain_generator.make_generator(
                domain=self.config["dataset"]["source"],
                batch_size=self.config["batch_size"],
                target_size=self.config["backbone"]["img_size"]
            ),
            domain=0
        )

        domain_1_generator = self._domain_wrapper(
            self.domain_generator.make_generator(
                domain=self.config["dataset"]["target"],
                batch_size=self.config["batch_size"],
                target_size=self.config["backbone"]["img_size"]
            ),
            domain=1
        )

        #  --- test dann model ---
        classification_model = keras.Model(
            inputs=dann_model.inputs,
            outputs=dann_model.outputs[0]
        )

        #  --- train dann model ---
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        steps_per_epoch = len(source_generator)

        for epoch_num in range(self.config["epochs"]):
            print(f"Starting epoch {epoch_num}", time.ctime())
            mean_domain_loss = keras.metrics.MeanTensor()
            mean_class_loss = keras.metrics.MeanTensor()
            mean_total_loss = keras.metrics.MeanTensor()
            for step_during_epoch in range(steps_per_epoch):
                with tf.GradientTape() as tape:
                    x_batch, y_batch = next(source_generator)
                    classification_loss = self._cross_entropy(dann_model, x_batch, y_batch, head=0)
                    mean_class_loss.update_state(classification_loss)

                    if train_domain_head:
                        x_batch, y_batch = next(domain_0_generator)
                        domain_loss = self._cross_entropy(dann_model, x_batch, y_batch, head=1)

                        x_batch, y_batch = next(domain_1_generator)
                        domain_loss += self._cross_entropy(dann_model, x_batch, y_batch, head=1)

                        mean_domain_loss.update_state(domain_loss)

                    if train_domain_head:
                        total_loss = classification_loss + domain_loss
                    else:
                        total_loss = classification_loss

                    mean_total_loss.update_state(total_loss)

                    grads = tape.gradient(total_loss, dann_model.trainable_variables)

                    optimizer.apply_gradients(zip(grads, dann_model.trainable_variables))

                    p_ = (steps_per_epoch * epoch_num + step_during_epoch) / (steps_per_epoch * self.config["epochs"])
                    lambda_.assign(DANNExperiment._get_lambda(p=p_))

                    # if step_during_epoch % 50 == 0:
                    #     print('Mean total loss:{}, lambda: {}'.format(total_loss, lambda_.numpy()))
                    #     if train_domain_head:
                    #         print('classification loss: {}, domain_loss: {}'.format(classification_loss, domain_loss))

            print(f"Finished epoch {epoch_num}", time.ctime())
            print('Mean total loss:{}, lambda: {}'.format(mean_total_loss.result().numpy(), lambda_.numpy()))
            print('classification loss: {}'.format(mean_class_loss.result().numpy()))
            if train_domain_head:
                print('domain_loss: {}'.format(mean_domain_loss.result().numpy()))

            print("Testing on source...")
            tester = Tester()
            tester.test(classification_model, source_generator)

            print("Testing on target...")
            tester = Tester()
            tester.test(
                classification_model,
                self.domain_generator.make_generator(
                    domain=self.config["dataset"]["target"],
                    batch_size=self.config["batch_size"],
                    target_size=self.config["backbone"]["img_size"]
                )
            )



    @staticmethod
    def _get_lambda(p=0):
        """ Original lambda scheduler """
        return 2 / (1 + np.exp(-10 * p)) - 1

    @staticmethod
    def _get_classifier_head(num_classes):
        return keras.models.Sequential([
            keras.layers.Dense(units=1024, activation='relu'),
            keras.layers.Dense(units=1024, activation='relu'),
            keras.layers.Dense(units=1024, activation='relu'),
            keras.layers.Dense(units=1024, activation='relu'),
            keras.layers.Dense(units=1024, activation='relu'),
            keras.layers.Dense(units=num_classes)
        ])