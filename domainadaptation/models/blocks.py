import tensorflow as tf

# swish block from  "Searching for activation functions" P. Ramachandran, B. Zoph, and Q. V. Le.
def swish1(x):
    return x * tf.keras.activations.sigmoid(x)


def create_activation(activation):
    if activation == 'swish1':
        return tf.keras.layers.Lambda(swish1)
    else:
        return tf.keras.layers.Activation(activation)

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, activation='relu', add_maxpool=True, **kwargs):
        self.conv = tf.keras.layers.Conv2D(out_channels, 3, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()

        if activation is not None:
            self.nl = create_activation(activation)
        else:
            self.nl = None

        if add_maxpool:
            self.maxpool = tf.keras.layers.MaxPool2D()
        else:
            self.maxpool = None

        self.out_channels = out_channels
        super(ConvBlock, self).__init__(**kwargs)

    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)

        if self.nl is not None:
            out = self.nl(out)

        if self.maxpool is not None:
            out = self.maxpool(out)

        return out

    def set_trainable(self, trainable):
        self.conv.trainable = trainable
        self.bn.trainable = trainable

    def compute_output_shape(self, input_shape):
        out_shape = self.conv.compute_output_shape(input_shape)
        if self.maxpool is not None:
            out_shape = self.maxpool.compute_output_shape(out_shape)

        return out_shape


# Resnet-12 from "Dense Classsification and Implanting for Few-Shot Learning" Yann Lifchitz...
class ResidualBlock(layers.Layer):
    def __init__(self,  out_channels, activation='swish1', **kwargs):
        self.conv1 = ConvBlock(out_channels, activation, add_maxpool=False)
        self.conv2 = ConvBlock(out_channels, activation, add_maxpool=False)
        self.conv3 = ConvBlock(out_channels, activation=None, add_maxpool=False)

        self.nl = create_activation(activation)
        self.maxpool = layers.MaxPool2D()

        self.conv_res = layers.Conv2D(out_channels, 3, padding='same')
        self.bn_res = layers.BatchNormalization()

        self.out_channels = out_channels
        super(ResidualBlock, self).__init__(**kwargs)

    def call(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        z = self.conv_res(x)
        z = self.bn_res(z)

        out = y + z
        out = self.nl(out)
        out = self.maxpool(out)

        return out

    def set_trainable(self, trainable):
        self.conv1.set_trainable(trainable)
        self.conv2.set_trainable(trainable)
        self.conv3.set_trainable(trainable)

        self.conv_res.trainable = trainable
        self.bn_res.trainable = trainable

    def compute_output_shape(self, input_shape):
        return self.conv_res.compute_output_shape(input_shape)
