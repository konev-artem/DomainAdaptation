import tensorflow as tf
from tensorflow.python.keras.layers import Layer


def build_grad_reverse(lambda_):
    @tf.custom_gradient
    def grad_reverse(x):
        y = tf.identity(x)

        def custom_grad(dy):
            return -lambda_ * dy

        return y, custom_grad

    return grad_reverse


class GradientReversal(Layer):
    """ Flip the sign of gradient during training. """

    def __init__(self, alpha, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self._trainable = False
        self.alpha = alpha
        self.grad_reverse = build_grad_reverse(self.alpha)

    def build(self, input_shape):
        pass

    def call(self, x, **kwargs):
        return self.grad_reverse(x)

    def compute_output_shape(self, input_shape):
        return input_shape
