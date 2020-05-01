import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import sys

from domainadaptation.tester import Tester
from domainadaptation.models import GradientReversal
from domainadaptation.experiment import Experiment
from domainadaptation.visualizer import Visualizer
from domainadaptation.utils import SphericalKMeans

from tqdm import trange


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

        source_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["source"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img_size"]
        )

        target_generator = self.domain_generator.make_generator(
            domain=self.config["dataset"]["target"],
            batch_size=self.config["batch_size"],
            target_size=self.config["backbone"]["img_size"]
        )

        self.__perform_can_loop(
            source_generator=source_generator,
            target_generator=target_generator,
            model=model,
            K=10)

    def __perform_can_loop(self, source_generator, target_generator, model, K):
        centers = self.__estimate_centers_init(source_generator=source_generator, model=model)

        target_y_kmeans, centers, convergence = self.__cluster_target_samples(
            centers_init=centers, target_generator=target_generator, model=model)

        # TODO: Filter the ambiguous target samples and classes

        for _ in range(K):
            # TODO: Class-aware sampling
            # TODO: Compute loss
            # TODO: Back-prop
            pass

    def __estimate_centers_init(self, source_generator, model, model_layer_ix=0, eps=1e-8):
        features = []
        labels = []

        for ix in trange(len(source_generator)):
            X, y = source_generator[ix]

            model_output = model(X)[model_layer_ix].numpy()

            features.append(model_output)
            labels.append(y.argmax(axis=-1))

        features = np.concatenate(features, axis=0)
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + eps)

        labels = np.concatenate(labels, axis=0)

        centers = np.empty((self.config['dataset']['classes'], features.shape[1]))
        for class_ix in range(self.config['dataset']['classes']):
            centers[class_ix] = np.sum(features[labels == class_ix], axis=0)

        return centers

    def __cluster_target_samples(self, centers_init, target_generator, model, model_layer_ix=0):
        features = []
        y_gt = []

        for ix in trange(len(target_generator)):
            X, y = target_generator[ix]

            model_output = model(X)[model_layer_ix].numpy()

            features.append(model_output)
            y_gt.append(y.argmax(axis=-1))

        features = np.concatenate(features, axis=0)
        y_gt = np.concatenate(y_gt, axis=0)

        kmeans = SphericalKMeans()
        y_kmeans, centers, convergence = kmeans.fit_predict(X=features, init=centers_init)

        sys.stderr.write("KMeans Accuracy: {}, Convergence: {}\n".format(np.mean(y_kmeans == y_gt), convergence))

        return y_kmeans, centers, convergence

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
