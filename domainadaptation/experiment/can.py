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

from domainadaptation.utils import SphericalKMeans
from domainadaptation.data_provider import LabeledDataset, MaskedGenerator

from tqdm import trange
import tqdm


class CANExperiment(Experiment):
    '''https://arxiv.org/abs/1901.00976'''

    def __init__(self, config):
        super().__init__(config)

    def __call__(self):
        backbone = self._get_new_backbone_instance()
        fc1 = keras.layers.Dense(1024, activation='relu')(backbone.outputs[0])
        fc2 = keras.layers.Dense(512, activation='relu')(fc1)
        fc3 = keras.layers.Dense(256, activation='relu')(fc2)
        fc4 = keras.layers.Dense(128, activation='relu')(fc3)
        fc5 = keras.layers.Dense(self.config['dataset']['classes'])(fc4)

        model = keras.Model(inputs=backbone.inputs, outputs=backbone.outputs + [fc1, fc2, fc3, fc4, fc5])

        # source_generator = self.domain_generator.make_generator(
        #     domain=self.config["dataset"]["source"],
        #     batch_size=self.config["batch_size"] // 2,
        #     target_size=self.config["backbone"]["img_size"]
        # )

        source_labeled_dataset = LabeledDataset(
            root=os.path.join(self.config["dataset"]["path"], self.config["dataset"]["source"]),
            img_size=self.config["backbone"]["img_size"][0],
            store_in_ram=True,
            type_label=0)

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
            type_label=0)   # 0 means to read dataset as labeled from folder, bet we ignore this labels
                            # this is done just to have some initialization

        target_masked_generator = MaskedGenerator(
            dataset=target_labeled_dataset,
            mask=np.ones(len(target_labeled_dataset)),
            batch_size=self.config["batch_size"] // 2,
            preprocess_input=self._preprocess_input,
        )

        for _ in range(self.config['CAN_steps']):
            self.__perform_can_loop(
                source_masked_generator=source_masked_generator,
                target_labeled_dataset=target_labeled_dataset,
                target_masked_generator=target_masked_generator,
                model=model,
                K=self.config['K'])

    def __perform_can_loop(self, source_masked_generator,
                           target_labeled_dataset, target_masked_generator,
                           model, K):
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
                # TODO: Compute loss
                # TODO: Back-prop
                pass

    def __estimate_centers_init(self, source_masked_generator, model, model_layer_ix=0, eps=1e-8):
        features = []
        labels = []

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
        centers = centers[labels]   # [N, dim]

        features = features / np.linalg.norm(features, axis=1, keepdims=True)    # [N, dim]

        assert features.ndim == centers.ndim == 2 and features.shape == centers.shape

        dist = 0.5 * (1 - np.sum(features * centers, axis=1))   # [N]
        assert dist.ndim == 1 and dist.shape[0] == features.shape[0]

        mask = dist < self.config['D_0']
        sys.stderr.write("{}% samples from target are close to their centers\n".format(np.mean(mask) * 100))
        return mask

    def __find_good_classes(self, labels, mask):
        labels = labels[mask]   # keep only labels where mask == 1

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
            return (tf.reduce_sum((cls_num - 1) * (e1s + e2s))\
                    - 2 * tf.reduce_sum(e3s - tf.eye(cls_num) * e3s)) / (cls_num * (cls_num - 1))
        
    @staticmethod
    def _cdd_loss(out_source, labels_source, out_target, labels_target):
        return CANExperiment._get_class_discrepancy(out_source, labels_source, out_target, labels_target, intra=True)\
            - CANExperiment._get_class_discrepancy(out_source, labels_source, out_target, labels_target, intra=False)
    
    @staticmethod
    def _crossentropy_loss(y, logits, from_logits=True):
        if from_logits:
            log_probs = tf.nn.log_softmax(logits, -1)
        else:
            log_probs = tf.math.log(logits, -1)
        return -tf.reduce_mean(tf.reduce_sum(y * log_probs, -1))

    @staticmethod
    def _loss(out_source, y_true_source, logits_source,
              out_target, labels_kmeans_target, beta, from_logits=True):
        labels_true_source = tf.argmax(y_true_source, -1)
        loss = CANExperiment._crossentropy_loss(y_true_source, logits_source, from_logits=from_logits)\
            + beta * CANExperiment._cdd_loss(out_source, labels_true_source, out_target, labels_kmeans_target)
        return loss
