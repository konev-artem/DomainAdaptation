from enum import Enum
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from ..tester import Tester
from ..trainer import Trainer
from ..visualizer import Visualizer
from ..data_provider import DomainGenerator
from ..models import GradientReversal


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
        assert isinstance(self.config['backbone']['num_trainable_layers'] >= -1, int)
        if self.config['backbone']['num_trainable_layers'] != -1:
            num_non_trainable_layers = len(instance.layers) - self.config['backbone']['num_trainable_layers']
            for layer in instance.layers[:num_non_trainable_layers]:
                layer.trainable = False
        
        return instance

    @staticmethod
    def _cross_entropy(model, x_batch, y_batch):
        logits = model(x_batch)
        return tf.nn.softmax_cross_entropy_with_logits(y_batch, logits)

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
        return keras.layers.Dense(units=num_classes)


def print_callback(iteration, loss):
    print(f"iteration: {iteration}, loss: {loss}")

class DANNExperiment(Experiment):
    """
    Domain-Adversarial Training of Neural Networks
    link: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, config):
        super().__init__(config)

    def experiment_no_domain_adaptation(self):
        backbone = self._get_new_backbone_instance()
        
        classifier_head = self._get_classifier_head(num_classes=self.config["dataset"]["classes"])
        
        classification_model = keras.Model(
            inputs=backbone.inputs,
            outputs=classifier_head(backbone.outputs[0]))
        
        
        source_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]
        )
        
        trainer = Trainer(
            model=classification_model,
            grads_update_freq=self.config["grads_update_freq"])
        optimizer = keras.optimizers.Adam()
        
        for i in range(self.config["epochs"]):
            trainer.train(
                compute_loss=self._cross_entropy,
                optimizer=optimizer,
                train_generator=source_generator,
                steps=self.config["steps"])
            print('epoch {} finished'.format(i + 1))
        
        
        tester = Tester()
        tester.test(classification_model, source_generator)
        
        tester = Tester()
        tester.test(classification_model, self.domain_generator.make_generator(
            domain=self.config["dataset"]["target"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]))


    def experiment_domain_adaptation(self):
        ###################### MODEL
        backbone = self._get_new_backbone_instance()
        
        classifier_head = self._get_classifier_head(num_classes=self.config["dataset"]["classes"])
        domain_head     = self._get_classifier_head(num_classes=2)
        
        classification_model = keras.Model(
            inputs=backbone.inputs,
            outputs=classifier_head(backbone.outputs[0]))
        
        lambda_ = tf.Variable(initial_value=1., trainable=False, dtype=tf.float32)
        gradient_reversal_layer = GradientReversal(lambda_)
        
        domain_model = keras.Model(
            inputs=backbone.inputs,
            outputs=domain_head(gradient_reversal_layer(backbone.outputs[0])))
        ######################
        
        source_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"])
        
        domain_0_generator = self._domain_wrapper(self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]), domain=0)
        
        domain_1_generator = self._domain_wrapper(self.domain_generator.make_generator(
            domain=self.config["dataset"]["target"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]), domain=1)
        
        trainer_classification = Trainer(
            model=classification_model,
            grads_update_freq=self.config["grads_update_freq"])
        
        trainer_domain = Trainer(
            model=domain_model,
            grads_update_freq=self.config["grads_update_freq"])
        
        optimizer = keras.optimizers.Adam()
        
        for i in range(self.config["epochs"]):
            if i % 2 == 0:
                trainer_classification.train(
                    compute_loss=self._cross_entropy,
                    optimizer=optimizer,
                    train_generator=source_generator,
                    steps=self.config["steps"])
            else:
                for j in range(self.config["steps"]):
                    trainer_domain.train(
                        compute_loss=self._cross_entropy,
                        optimizer=optimizer,
                        train_generator=domain_0_generator,
                        steps=1)
                    trainer_domain.train(
                        compute_loss=self._cross_entropy,
                        optimizer=optimizer,
                        train_generator=domain_1_generator,
                        steps=1)
            print('epoch {} finished'.format(i + 1))
        
        
        tester = Tester()
        tester.test(classification_model, source_generator)
        
        tester = Tester()
        tester.test(classification_model, self.domain_generator.make_generator(
            domain=self.config["dataset"]["target"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]))
    
    def experiment_domain_adaptation_v2(self, train_domain_head=True):
        ### MODEL #############################################
        backbone = self._get_new_backbone_instance()
        
        classifier_head = self._get_classifier_head(num_classes=self.config["dataset"]["classes"])
        domain_head     = self._get_classifier_head(num_classes=2)
        
        lambda_ = tf.Variable(initial_value=1., trainable=False, dtype=tf.float32)
        gradient_reversal_layer = GradientReversal(lambda_)
        
        dann_model = keras.Model(inputs=backbone.inputs, outputs=[
            classifier_head(backbone.outputs[0]),
            domain_head(gradient_reversal_layer(backbone.outputs[0]))
        ])
        #######################################################
        
        source_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"])

        target_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["target"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"])
        
        domain_0_generator = self._domain_wrapper(self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]), domain=0)
        
        domain_1_generator = self._domain_wrapper(self.domain_generator.make_generator(
            domain=self.config["dataset"]["target"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img-size"]), domain=1)

        classification_model = keras.Model(
            inputs=dann_model.inputs,
            outputs=dann_model.outputs[0])

        tester = Tester()
        
        optimizer = keras.optimizers.Adam(lr=self.config['lr'])
        steps_per_epoch = len(source_generator)
        for epoch_num in range(self.config["epochs"]):
            print(f"Starting epoch {epoch_num}", time.ctime())
            mean_domain_loss = keras.metrics.MeanTensor()
            mean_class_loss = keras.metrics.MeanTensor()
            mean_total_loss = keras.metrics.MeanTensor()
            for step_during_epoch in range(steps_per_epoch):
                with tf.GradientTape() as tape:
                    X, y = next(source_generator)
                    classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, dann_model(X)[0]))
                    mean_class_loss.update_state(classification_loss)
                    if train_domain_head:
                        X, y = next(domain_0_generator)
                        domain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, dann_model(X)[1]))

                        X, y = next(domain_1_generator)
                        domain_loss = domain_loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, dann_model(X)[1]))

                    if train_domain_head:
                        mean_domain_loss.update_state(classification_loss)
                        total_loss = classification_loss + domain_loss
                    else:
                        total_loss = classification_loss
                    mean_total_loss.update_state(total_loss)
                grads = tape.gradient(total_loss, dann_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, dann_model.trainable_variables))
                    
                # p_ = (steps_per_epoch * epoch_num + step_during_epoch) / (steps_per_epoch * self.config["epochs"])
                p_ = 1
                lambda_.assign(DANNExperiment._get_lambda(p=p_))
            print(f"Finished epoch {epoch_num}", time.ctime())
            print('Mean total loss:{}, lambda: {}'.format(mean_total_loss.result().numpy(), lambda_.numpy()))
            if train_domain_head:
                print('classification loss: {}, domain_loss: {}'.format(mean_class_loss.result().numpy(), mean_domain_loss.result().numpy()))
            print("Testing on source...")
            tester.test(classification_model, source_generator)
            print("Testing on target...")
            tester.test(classification_model, target_generator)
                        
        #######################################################


    @staticmethod
    def _get_lambda(p=0):
        """ Original lambda scheduler """
        return 2 / (1 + np.exp(-10 * p)) - 1
