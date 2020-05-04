import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import sys
import os

from domainadaptation.tester import Tester
from domainadaptation.models import GradientReversal
from domainadaptation.experiment import Experiment
from domainadaptation.visualizer import Visualizer

from domainadaptation.utils import SphericalKMeans, make_batch_normalization_layers_domain_specific
from domainadaptation.data_provider import LabeledDataset, MaskedGenerator

from tqdm import trange
import tqdm


class CANExperiment(Experiment):
    '''https://arxiv.org/abs/1901.00976'''

    def __init__(self, config):
        super().__init__(config)

        # Weight of CDD loss in total loss
        self._beta = 0.3

        # Parameters for learning rate scheduling
        self._conv_lr0 = 0.001
        self._dense_lr0 = 0.01
        self._a = 10
        self._b = 0.75

    def __switch_batchnorm_mode(self, mode):
        assert mode in ['source', 'target']
        self.domain_variable.assign(mode == 'source')

    def __call__(self):
        backbone = self._get_new_backbone_instance()

        self.domain_variable = tf.Variable(False, dtype=tf.bool, trainable=False)
        if self.config['use_domain_specific_batchnormalization'] is True:
            backbone = make_batch_normalization_layers_domain_specific(backbone, self.domain_variable)

        fc1 = keras.layers.Dense(1024, activation='relu')(backbone.outputs[0])
        fc2 = keras.layers.Dense(512, activation='relu')(fc1)
        fc3 = keras.layers.Dense(256, activation='relu')(fc2)
        fc4 = keras.layers.Dense(128, activation='relu')(fc3)
        fc5 = keras.layers.Dense(self.config['dataset']['classes'])(fc4)

        model = keras.Model(inputs=backbone.inputs, outputs=backbone.outputs + [fc1, fc2, fc3, fc4, fc5])

        # save backbone and head variables
        backbone_names = [var.name for var in backbone.trainable_variables]

        self.backbone_variables, self.head_variables = [], []

        for var in model.trainable_variables:
            if var.name in backbone_names:
                self.backbone_variables.append(var)
            else:
                self.head_variables.append(var)
                
        test_model = keras.Model(inputs=model.inputs, outputs=model.outputs[-1])
        
        source_labeled_dataset = LabeledDataset(
            root=os.path.join(self.config["dataset"]["path"], self.config["dataset"]["source"]),
            img_size=self.config["backbone"]["img_size"][0],
            store_in_ram=True,
            type_label=0
        )

        source_masked_generator = MaskedGenerator(
            dataset=source_labeled_dataset,
            mask=np.ones(len(source_labeled_dataset)),
            batch_size=self.config["batch_size"] // 2,
            preprocess_input=self._preprocess_input,
        )

        target_labeled_dataset = LabeledDataset(
            root=os.path.join(self.config["dataset"]["path"], self.config["dataset"]["target"]),
            img_size=self.config["backbone"]["img_size"][0],
            store_in_ram=True,
            type_label=0  # 0 means to read dataset as labeled from folder, bet we ignore this labels
        )  # this is done just to have some initialization

        target_masked_generator = MaskedGenerator(
            dataset=target_labeled_dataset,
            mask=np.ones(len(target_labeled_dataset)),
            batch_size=self.config["batch_size"] // 2,
            preprocess_input=self._preprocess_input,
        )

        source_test_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img_size"]
        )

        target_test_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["target"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img_size"]
        )

        tester = Tester()

        optimizer = keras.optimizers.SGD(learning_rate=self.config['learning_rate'], momentum=0.9)
        p = 0.

        optimizers = {
            'head': keras.optimizers.SGD(learning_rate=self._dense_lr0, momentum=0.9),
            'backbone': keras.optimizers.SGD(learning_rate=self._conv_lr0, momentum=0.9)
        }

        for i in range(self.config['CAN_steps']):
            p = i / self.config['CAN_steps']
            self.__perform_can_loop(
                source_masked_generator=source_masked_generator,
                target_labeled_dataset=target_labeled_dataset,
                target_masked_generator=target_masked_generator,
                model=model, K=self.config['K'], optimizer=optimizers, p=p)

            self.__switch_batchnorm_mode('source')
            tester.test(test_model, source_test_generator)
            self.__switch_batchnorm_mode('target')
            tester.test(test_model, target_test_generator)

    def __perform_can_loop(self, source_masked_generator,
                           target_labeled_dataset, target_masked_generator,
                           model, K, optimizer, p):

        # Estimate centers using source dataset
        centers = self.__estimate_centers_init(source_masked_generator=source_masked_generator, model=model)

        # Reset mask to iterate the whole dataset in __cluster_target_samples
        target_masked_generator.set_mask(np.ones(len(target_labeled_dataset)))

        # Cluster target samples
        target_y_kmeans, centers, convergence, good_classes = self.__cluster_target_samples(
            centers_init=centers,
            target_masked_generator=target_masked_generator,
            model=model)

        # Update target dataset labels with labels obtained from KMeans
        target_labeled_dataset.set_labels(target_y_kmeans)

        for _ in range(K):
            classes_to_use_in_batch = np.random.choice(
                good_classes,
                size=min(self.config['MAX_CLASSES_PER_BATCH'], len(good_classes)),
                replace=False)

            X_target, y_target = target_masked_generator.get_batch(classes_to_use_in_batch)
            X_source, y_source = source_masked_generator.get_batch(classes_to_use_in_batch)

            with tf.GradientTape() as tape:
                self.__switch_batchnorm_mode('source')
                model_output_source = model(X_source)

                self.__switch_batchnorm_mode('target')
                model_output_target = model(X_target)

                logits_source = model_output_source[-1]
                crossentropy_loss = self._crossentropy_loss(y_source, logits_source, from_logits=True)

                cdd_loss = 0
                # ...[:-1] --- do not consider logits in CDD loss
                for out_source, out_target in zip(model_output_source[:-1], model_output_target[:-1]):
                    cdd_loss += CANExperiment._cdd_loss(out_source, y_source, out_target, y_target)

                loss = crossentropy_loss + self._beta * cdd_loss

            grads = tape.gradient(loss, model.trainable_variables)
            grad_convs = grads[:len(self.backbone_variables)]
            grad_denses = grads[len(self.backbone_variables):]

            optimizer['backbone'].apply_gradients(zip(grad_convs, self.backbone_variables))
            optimizer['head'].apply_gradients(zip(grad_denses, self.head_variables))
            # optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print('Progress: {}, loss:{}\ncrossentropy_loss: {}, cdd_loss: {}' \
                  .format(p, loss, crossentropy_loss, cdd_loss))

    def __estimate_centers_init(self, source_masked_generator, model, model_layer_ix=0, eps=1e-8):
        features = []
        labels = []

        self.__switch_batchnorm_mode('source')
        for X, y in tqdm.tqdm(source_masked_generator()):
            model_output = model(X)[model_layer_ix].numpy()

            features.append(model_output)
            labels.append(y)

        features = np.concatenate(features, axis=0)
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + eps)

        labels = np.concatenate(labels, axis=0)

        centers = np.empty((self.config['dataset']['classes'], features.shape[1]))
        for class_ix in range(self.config['dataset']['classes']):
            centers[class_ix] = np.sum(features[labels == class_ix], axis=0)

        return centers

    def __cluster_target_samples(self, centers_init, target_masked_generator, model, model_layer_ix=0):
        features = []

        self.__switch_batchnorm_mode('target')
        for X, _ in tqdm.tqdm(target_masked_generator()):
            model_output = model(X)[model_layer_ix].numpy()
            features.append(model_output)

        features = np.concatenate(features, axis=0)

        kmeans = SphericalKMeans()
        y_kmeans, centers, convergence = kmeans.fit_predict(X=features, init=centers_init)

        close_to_center_mask = self.__find_samples_close_to_centers(features, y_kmeans, centers)
        target_masked_generator.set_mask(close_to_center_mask)

        good_classes = self.__find_good_classes(y_kmeans, close_to_center_mask)

        return y_kmeans, centers, convergence, good_classes

    def __find_samples_close_to_centers(self, features, labels, centers):
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers[labels]  # [N, dim]

        features = features / np.linalg.norm(features, axis=1, keepdims=True)  # [N, dim]

        assert features.ndim == centers.ndim == 2 and features.shape == centers.shape

        dist = 0.5 * (1 - np.sum(features * centers, axis=1))  # [N]
        assert dist.ndim == 1 and dist.shape[0] == features.shape[0]

        mask = dist < self.config['D_0']
        sys.stderr.write("{}% samples from target are close to their centers\n".format(np.mean(mask) * 100))
        return mask

    def __find_good_classes(self, labels, mask):
        labels = labels[mask]  # keep only labels where mask == 1

        good_classes = []
        for class_ix in range(self.config['dataset']['classes']):
            if np.sum(labels == class_ix) > self.config['N_0']:
                good_classes.append(class_ix)

        sys.stderr.write("Found {} good classes after filtering far samples\n".format(len(good_classes)))
        return np.asarray(good_classes, dtype=np.int32)

    @staticmethod
    def _kernel(out_1, out_2, fixed_sigma=None):
        l2_distance = tf.reduce_sum((out_1[:, None] - out_2[None]) ** 2, axis=-1)
        if fixed_sigma:
            bandwidth = fixed_sigma
        else:
            bandwidth = tf.reduce_mean(l2_distance)
        kernel_val = tf.math.exp(-l2_distance / bandwidth)
        return kernel_val

    @staticmethod
    def _get_mask(labels_1, labels_2, intra=True):
        cls_num = tf.unique(labels_1)[0].shape[0]
        assert cls_num == tf.unique(labels_2)[0].shape[0], "Different number of classes in source and target domains"

        n = labels_1.shape[0]
        m = labels_2.shape[0]

        cls_mask = tf.tile(tf.unique(labels_1)[0][:, None, None], [1, n, m])
        a = tf.cast(cls_mask == tf.tile(tf.tile(labels_1[:, None], [1, m])[None], [cls_num, 1, 1]), tf.float32)
        b = tf.cast(cls_mask == tf.tile(tf.tile(labels_2[None], [n, 1])[None], [cls_num, 1, 1]), tf.float32)

        if intra:
            return a * b
        else:
            return a[:, None] * b[None]

    @staticmethod
    def _get_class_discrepancy(out_source, labels_source, out_target, labels_target, intra=True):
        cls_num = len(tf.unique(labels_source)[0])

        mask_ss = CANExperiment._get_mask(labels_source, labels_source, intra=True)
        mask_tt = CANExperiment._get_mask(labels_target, labels_target, intra=True)
        mask_st = CANExperiment._get_mask(labels_source, labels_target, intra=intra)

        axs = None
        dim_coefs = [cls_num, 1, 1]

        kernel_ss = tf.tile(CANExperiment._kernel(out_source, out_source)[axs], dim_coefs)
        kernel_tt = tf.tile(CANExperiment._kernel(out_target, out_target)[axs], dim_coefs)

        if not intra:
            axs = (None, None)
            dim_coefs = [cls_num, cls_num, 1, 1]
        kernel_st = tf.tile(CANExperiment._kernel(out_source, out_target)[axs], dim_coefs)

        e1s = tf.reduce_sum(mask_ss * kernel_ss, (-2, -1)) / tf.reduce_sum(mask_ss, (-2, -1))
        e2s = tf.reduce_sum(mask_tt * kernel_tt, (-2, -1)) / tf.reduce_sum(mask_tt, (-2, -1))
        e3s = tf.reduce_sum(mask_st * kernel_st, (-2, -1)) / tf.reduce_sum(mask_st, (-2, -1))

        if intra:
            return tf.reduce_mean(e1s + e2s - 2 * e3s)
        else:
            return (tf.reduce_sum((cls_num - 1) * (e1s + e2s)) \
                    - 2 * tf.reduce_sum(e3s - tf.eye(cls_num) * e3s)) / (cls_num * (cls_num - 1))

    @staticmethod
    def _cdd_loss(out_source, labels_source, out_target, labels_target):
        return CANExperiment._get_class_discrepancy(out_source, labels_source, out_target, labels_target, intra=True) \
               - CANExperiment._get_class_discrepancy(out_source, labels_source, out_target, labels_target, intra=False)

    def _crossentropy_loss(self, labels, logits, from_logits=True):
        cls_num = self.config['dataset']['classes']
        y = tf.one_hot(labels, cls_num, dtype=tf.float32)
        if from_logits:
            log_probs = tf.nn.log_softmax(logits, -1)
        else:
            log_probs = tf.math.log(logits, -1)
        return -tf.reduce_mean(tf.reduce_sum(y * log_probs, -1))