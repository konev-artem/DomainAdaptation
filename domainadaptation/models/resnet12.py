from tensorflow.python.keras import layers, models

from blocks import ResidualBlock


class Resnet12:
    def __init__(self, input_size, activation='swish1'):
        self.residual1 = ResidualBlock(64, activation)
        self.residual2 = ResidualBlock(128, activation)
        self.residual3 = ResidualBlock(256, activation)
        self.residual4 = ResidualBlock(512, activation)
        self.inputs, self.outputs = self._build_net(input_size)

    def _build_net(self, input_size):
        input = layers.Input(shape=input_size)
        x = self.residual1(input)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        return [input], [x]

    def build_model(self):
        return models.Model(self.inputs, self.outputs)

    def set_trainable(self, trainable):
        self.residual1.set_trainable(trainable)
        self.residual2.set_trainable(trainable)
        self.residual3.set_trainable(trainable)
        self.residual4.set_trainable(trainable)

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs
