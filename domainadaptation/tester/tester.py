import os
import tqdm
import datetime

import numpy as np
import tensorflow as tf


class Tester:
    def __init__(self):
        pass

    def test(self, 
             model, 
             generator, 
             model_name='baseline', 
             tensorboard=False, 
             log_dir='../domainadaptation/logs'):
        if tensorboard:
            log_dir = os.path.join(log_dir,
                '{}_{}'.format(model_name, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
            os.makedirs(log_dir, exist_ok=True)
            file_writer = tf.summary.FileWriter(log_dir)

        accuracies = []
        
        tbar = tqdm.trange(generator.__len__())
        for index in tbar:
            x_batch_test, y_batch_test = generator[index]

            logits = model(x_batch_test, training=False)
            try:
                accuracy = np.mean(np.argmax(logits, axis=1) == y_batch_test.argmax(axis=1))
            except IndexError:
                accuracy = np.mean(np.argmax(logits, axis=1) == y_batch_test)
            accuracies.append(accuracy)

            tbar.set_description(
                "Average acc: {:.2f}%".format(np.mean(accuracies) * 100))

            if tensorboard:
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag='accuracy', simple_value=accuracy),
                    tf.Summary.Value(tag='average_accuracy', simple_value=np.mean(accuracies))
                ])
                file_writer.add_summary(summary, global_step=index)
                file_writer.flush()

        return accuracies

    def bootstrap(self, accuracy, sz=1000, seed=42, ci_lvl=0.95, verbose=True):
        np.random.seed(seed)
        bts = np.random.choice(accuracy, size=(sz, len(accuracy)), replace=True)
        bts = np.sort(np.mean(bts, 1))
        quant_left = int((1 - ci_lvl) * sz // 2)
        left_bound = bts[quant_left]
        right_bound = bts[-quant_left]
        if verbose:
            print('metric: accuracy, mean: {:.2f}, std: {:.2f}, 95% conf interval: [{:.2f} ,{:.2f}]'.format(
                np.mean(accuracy), np.std(accuracy), left_bound, right_bound))
        return np.mean(accuracy), np.std(accuracy), left_bound, right_bound
