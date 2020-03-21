import tensorflow as tf


class Trainer:
    def __init__(self):
        pass

    def train(self, model, compute_loss, optimizer, train_generator, steps, callbacks=None):
        for i in range(steps):
            batch_inp, batch_out = next(train_generator)
            with tf.GradientTape() as tape:
                loss = compute_loss(model, batch_inp, batch_out)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if callbacks is not None:
                for callback in callbacks:
                    kwargs = {'iteration': i, 'loss': loss}
                    callback(**kwargs)
