import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from domainadaptation.tester import Tester
from domainadaptation.models import GradientReversal
from domainadaptation.experiment.experiment import Experiment
from domainadaptation.visualizer import Visualizer

from tqdm import trange

class CANExperiment(Experiment):
    '''https://arxiv.org/abs/1901.00976'''

    def __init__(self, config):
        super().__init__(config)

    def __call__(self):
        
        np.random.seed(2)
        
        n = 23
        m = 20
        n_features = 10
        cls_num = 6
        
        out_s = tf.convert_to_tensor(np.random.rand(n, n_features), tf.float32)
        y_s = np.random.randint(0, cls_num, n)
        out_t = tf.convert_to_tensor(np.random.rand(m, n_features), tf.float32)
        y_t = np.random.randint(0, cls_num, m)
        
        assert tf.unique(y_s)[0].shape[0] == cls_num and tf.unique(y_t)[0].shape[0] == cls_num
        
        k_ss = CANExperiment._kernel(out_s, out_s)
        k_tt = CANExperiment._kernel(out_t, out_t)
        k_st = CANExperiment._kernel(out_s, out_t)
        
        intra_disc = 0
        for c in range(cls_num):
            e1 = 0
            cnt = 0
            for i in range(n):
                for j in range(n):
                    if y_s[i] != c or y_s[j] != c:
                        continue
                    e1 += k_ss[i][j]
                    cnt += 1
            e1 /= cnt
            
            e2 = 0
            cnt = 0
            for i in range(m):
                for j in range(m):
                    if y_t[i] != c or y_t[j] != c:
                        continue
                    e2 += k_tt[i][j]
                    cnt += 1
            e2 /= cnt
                    
            e3 = 0
            cnt = 0
            for i in range(n):
                for j in range(m):
                    if y_s[i] != c or y_t[j] != c:
                        continue
                    e3 += k_st[i][j]
                    cnt += 1
            e3 /= cnt
            
            intra_disc += e1 + e2 - 2 * e3
            
        intra_disc /= cls_num
        intra_disc_pred = CANExperiment._get_class_discrepancy(out_s, tf.one_hot(y_s, cls_num),
                                                               out_t, tf.one_hot(y_t, cls_num), intra=True)
        print(intra_disc, intra_disc_pred)
        assert np.allclose(intra_disc, intra_disc_pred)
        
        inter_disc = 0
        for c1 in range(cls_num):
            for c2 in range(cls_num):
                if c1 == c2:
                    continue
                e1 = 0
                cnt = 0
                for i in range(n):
                    for j in range(n):
                        if y_s[i] != c1 or y_s[j] != c1:
                            continue
                        e1 += k_ss[i][j]
                        cnt += 1
                e1 /= cnt

                e2 = 0
                cnt = 0
                for i in range(m):
                    for j in range(m):
                        if y_t[i] != c2 or y_t[j] != c2:
                            continue
                        e2 += k_tt[i][j]
                        cnt += 1
                e2 /= cnt

                e3 = 0
                cnt = 0
                for i in range(n):
                    for j in range(m):
                        if y_s[i] != c1 or y_t[j] != c2:
                            continue
                        e3 += k_st[i][j]
                        cnt += 1
                e3 /= cnt

                inter_disc += e1 + e2 - 2 * e3
        
        inter_disc /= cls_num * (cls_num - 1)
        inter_disc_pred = CANExperiment._get_class_discrepancy(out_s, tf.one_hot(y_s, cls_num),
                                                               out_t, tf.one_hot(y_t, cls_num), intra=False)
        print(inter_disc, inter_disc_pred)
        assert np.allclose(inter_disc, inter_disc_pred)
        
    @staticmethod
    def _kernel(out_1, out_2):
        return tf.reduce_max(out_1, -1)[:, None] * tf.reduce_max(out_2, -1)[None]
    
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
    def _get_class_discrepancy(out_source, y_source, out_target, y_target, intra=True):
        labels_source = tf.argmax(y_source, -1)
        labels_target = tf.argmax(y_target, -1)
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
    def _cdd_loss(out_source, y_source, out_target, y_target):
        return CANExperiment._get_class_discrepancy(out_source, y_source, out_target, y_target, intra=True)\
            - CANExperiment._get_class_discrepancy(out_source, y_source, out_target, y_target, intra=False)
    
    @staticmethod
    def _crossentropy_loss(y_source, y_target, from_logits=True):
        if from_logits:
            log_probs = tf.math.log(tf.nn.softmax(y_target, -1))
        else:
            log_probs = tf.math.log(y_target, -1)
        return -tf.reduce_mean(tf.reduce_sum(y_source * log_probs, -1))

    @staticmethod
    def _loss(out_source, y_source, kmeans_labels_source,
              out_target, y_target, kmeans_labels_target, beta, from_logits=True):
        cls_num = tf.unique(kmeans_labels_source)[0].shape[0]
        assert cls_num == tf.unique(kmeans_labels_target)[0].shape[0],\
            "Different number of classes in source and target domains"
        
        loss = CANExperiment._crossentropy_loss(y_source, y_target, from_logits=from_logits)\
            + beta * CANExperiment._cdd_loss(out_source, tf.one_hot(kmeans_labels_source, cls_num),
                                             out_target, tf.one_hot(kmeans_labels_target, cls_num))
