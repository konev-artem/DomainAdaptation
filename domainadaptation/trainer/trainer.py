import tensorflow as tf
import tqdm


class Trainer:
    def __init__(self, model, grads_update_freq=1):
        assert isinstance(grads_update_freq, int)
        assert grads_update_freq >= 1

        self.model = model
        self.grads_update_freq = grads_update_freq

        if grads_update_freq > 1:
            self._initialize_accumulator()

    def _initialize_accumulator(self):
        self._accumulator = [
            tf.Variable(tf.zeros_like(var), trainable=False)
            for var in self.model.trainable_variables
        ]

        self._steps_made = 0

    def _reset_accumulator(self):
        for i in range(len(self._accumulator)):
            self._accumulator[i].assign(tf.zeros_like(self._accumulator[i]))

        self._steps_made = 0

    def _append_accumulator(self, grads):
        assert len(grads) == len(self._accumulator)

        for grad, acc_grad in zip(grads, self._accumulator):
            acc_grad.assign_add(grad)

        self._steps_made += 1

    def _reduce_mean_accumulator(self):
        for acc_grad in self._accumulator:
            acc_grad.assign(acc_grad / self._steps_made)

    def apply_accumulated_grads(self, optimizer):
        assert self.grads_update_freq > 1

        if self._steps_made > 0:
            self._reduce_mean_accumulator()
            optimizer.apply_gradients(zip(self._accumulator, self.model.trainable_variables))
            self._reset_accumulator()

    def train(self, compute_loss, optimizer, train_generator, steps, callbacks=None):
        for i in tqdm.trange(steps):
            batch_inp, batch_out = next(train_generator)

            with tf.GradientTape() as tape:
                loss = compute_loss(self.model, batch_inp, batch_out)

            grads = tape.gradient(loss, self.model.trainable_variables)
            if self.grads_update_freq == 1:
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            else:
                self._append_accumulator(grads)
                if self._steps_made % self.grads_update_freq == 0:
                    self.apply_accumulated_grads(optimizer=optimizer)

            if callbacks is not None:
                for callback in callbacks:
                    kwargs = {'iteration': i, 'loss': loss}
                    callback(**kwargs)