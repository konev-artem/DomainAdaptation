import numpy as np
import tensorflow as tf
from domainadaptation.experiment import CANExperiment


def compute_e(kernel, labels_1, labels_2, class_1, class_2):
    e = 0
    cnt = 0
    for i in range(len(labels_1)):
        for j in range(len(labels_2)):
            if labels_1[i] != class_1 or labels_2[j] != class_2:
                continue
            e += kernel[i][j]
            cnt += 1
    e /= cnt
    return e


def naive_intra_discrepancy(out_source, labels_source, out_target, labels_target, cls_num):
    kernel_ss = CANExperiment._kernel(out_source, out_source)
    kernel_tt = CANExperiment._kernel(out_target, out_target)
    kernel_st = CANExperiment._kernel(out_source, out_target)

    intra_discrepancy = 0
    for c in range(cls_num):
        e1 = compute_e(kernel_ss, labels_source, labels_source, c, c)
        e2 = compute_e(kernel_tt, labels_target, labels_target, c, c)
        e3 = compute_e(kernel_st, labels_source, labels_target, c, c)

        intra_discrepancy += e1 + e2 - 2 * e3
        
    intra_discrepancy /= cls_num
    return intra_discrepancy


def naive_inter_discrepancy(out_source, labels_source, out_target, labels_target, cls_num):
    kernel_ss = CANExperiment._kernel(out_source, out_source)
    kernel_tt = CANExperiment._kernel(out_target, out_target)
    kernel_st = CANExperiment._kernel(out_source, out_target)
    
    inter_discrepancy = 0
    for c1 in range(cls_num):
        for c2 in range(cls_num):
            if c1 == c2:
                continue

            e1 = compute_e(kernel_ss, labels_source, labels_source, c1, c1)
            e2 = compute_e(kernel_tt, labels_target, labels_target, c2, c2)
            e3 = compute_e(kernel_st, labels_source, labels_target, c1, c2)

            inter_discrepancy += e1 + e2 - 2 * e3

    inter_discrepancy /= cls_num * (cls_num - 1)
    return inter_discrepancy


def test_discrepancy():
    np.random.seed(0)
    
    tests_num = 0
    intra_tests_passed = 0
    inter_tests_passed = 0
    for n in range(15, 20):
        for m in range(15, 20):
            for n_features in range(7, 10):
                for cls_num in range(4, 7):
                    out_source = tf.convert_to_tensor(np.random.rand(n, n_features), tf.float32)
                    labels_source = np.random.randint(0, cls_num, n)
                    out_target = tf.convert_to_tensor(np.random.rand(m, n_features), tf.float32)
                    labels_target = np.random.randint(0, cls_num, m)
                    
                    if not (tf.unique(labels_source)[0].shape[0] == cls_num
                            and tf.unique(labels_target)[0].shape[0] == cls_num):
                        continue
                        
                    tests_num += 1
                    
                    intra_true = naive_intra_discrepancy(out_source, labels_source, out_target, labels_target, cls_num)
                    intra_pred = CANExperiment._get_class_discrepancy(out_source, labels_source,
                                                                      out_target, labels_target, intra=True)
                    print("TEST {}".format(tests_num))
                    print("intra_true = {}, intra_pred = {}".format(intra_true.numpy(), intra_pred.numpy()))
                    if np.allclose(intra_true, intra_pred):
                        intra_tests_passed += 1
                        
                        
                    inter_true = naive_inter_discrepancy(out_source, labels_source, out_target, labels_target, cls_num)
                    inter_pred = CANExperiment._get_class_discrepancy(out_source, labels_source,
                                                                      out_target, labels_target, intra=False)
                    print("inter_true = {}, inter_pred = {}\n".format(inter_true.numpy(), inter_pred.numpy()))
                    if np.allclose(inter_true, inter_pred):
                        inter_tests_passed += 1
                        
    print("Testing completed.")
    print("{}/{} 'intra' tests passed".format(intra_tests_passed, tests_num))
    print("{}/{} 'inter' tests passed".format(inter_tests_passed, tests_num))
