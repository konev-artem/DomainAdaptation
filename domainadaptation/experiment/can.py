import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from domainadaptation.tester import Tester
from domainadaptation.models import GradientReversal
from domainadaptation.experiment import Experiment
from domainadaptation.visualizer import Visualizer

from tqdm import trange

class CANExperiment(Experiment):
    '''https://arxiv.org/abs/1901.00976'''

    def __init__(self, config):
        super().__init__(config)

    def __call__(self):
        pass

    @staticmethod
    def _kernel(out_1, out_2, sigma=1.):
        norm = tf.reduce_sum((out_1[:, None] - out_2[None]) ** 2, -1)
        return tf.exp(-norm / sigma)
    
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
