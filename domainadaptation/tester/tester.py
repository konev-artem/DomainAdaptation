import os
import tqdm
import datetime

import tensorflow as tf


def baseline_domainadaptation_test(model,
                                   generator,
                                   model_name='baseline',
                                   tensorboard=False,
                                   log_dir='../domainadaptation/logs'):

    if tensorboard:
        log_dir = os.path.join(log_dir,
            '{}_{}'.format(model_name, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
        os.makedirs(log_dir, exist_ok=True)
        file_writer = tf.summary.FileWriter(log_dir)

    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    tbar = tqdm.trange(generator.__len__())
    for index in tbar:
        x_batch_test, y_batch_test = generator[index]

        logits = model.predict(x_batch_test)

        test_acc_metric(y_batch_test, logits)

        tbar.set_description(
            "Average acc: {:.2f}%".format(test_acc_metric.result().numpy() * 100))

        if tensorboard:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='average_accuracy',
                                 simple_value=test_acc_metric.result().numpy())
            ])
            file_writer.add_summary(summary, global_step=index)
            file_writer.flush()

    return test_acc_metric.result().numpy()
