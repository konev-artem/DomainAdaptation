import tensorflow as tf
import tensorflow.keras as keras
import sys


def get_class(kls):
    """
    This function return class by its name.
    :param kls: class name
    :return: class object
    """
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


class DomainSpecificBatchNormalization(keras.layers.Layer):
    """
    This is a Domain-Specific Batch Normalization layer.
    """
    def __init__(self, domain_variable, bn_config):
        """
        :param domain_variable: This tf.Variable(dtype=bool) indicates if the layer is used in source or target mode
        :param bn_config: BatchNormalization layers will be constructed with this parameters
        """
        super(DomainSpecificBatchNormalization, self).__init__()

        self.domain_variable = domain_variable
        self.src_bn = keras.layers.BatchNormalization(**bn_config)
        self.trg_bn = keras.layers.BatchNormalization(**bn_config)

    def call(self, inputs):
        return tf.cond(pred=self.domain_variable,
                       true_fn=lambda: self.src_bn(inputs),
                       false_fn=lambda: self.trg_bn(inputs))


def make_batch_normalization_layers_domain_specific_and_set_regularization(model, domain_variable, copy_bnorm_weights=True,
                                                                           kernel_regularizer=None, bias_regularizer=None):
    """
    This function replaces all BatchNormalization layers with DomainSpecificBatchNormalization layers.
    You have to manually change domain_variable in order to switch between source and target modes.
    This function can be also used to set kernel & bias regularizers.
    The weights are copied from old model into new model.
    :param model: Old model
    :param domain_variable: This tf.Variable(dtype=bool) indicates if the layer is used in source or target mode
    :param copy_bnorm_weights: Whether to copy old BatchNormalization weights into new ones
    :param kernel_regularizer: Instance of tf.keras.regularizers.Regularizer.
    :param bias_regularizer: Instance of tf.keras.regularizers.Regularizer.
    :return: New model
    """
    layers = []
    layer_name_to_ix = dict()

    kernel_regularizer_cnt = 0
    bias_regularizer_cnt = 0

    for ix, (layer_config, old_layer) in enumerate(zip(model.get_config()['layers'], model.layers)):
        layer_name_to_ix[layer_config['name']] = ix
        layer_class = get_class("tensorflow.keras.layers.{}".format(layer_config['class_name']))

        if not layers:
            new_layer = keras.layers.Input(
                shape=layer_config['config']['batch_input_shape'][1:],
                dtype=layer_config['config']['dtype'],
                name=layer_config['config']['name'])

            layers.append(new_layer)
            continue

        if layer_config['class_name'] == 'BatchNormalization':
            bn_config = layer_config['config']
            del bn_config['name']
            new_layer = DomainSpecificBatchNormalization(domain_variable, bn_config)
        else:
            new_layer = layer_class(**layer_config['config'])
            for attr, regularizer in zip(['kernel_regularizer', 'bias_regularizer'],
                                         [kernel_regularizer, bias_regularizer]):
                if hasattr(new_layer, attr) and regularizer is not None:
                    setattr(new_layer, attr, regularizer)
                    if attr == 'kernel_regularizer':
                        kernel_regularizer_cnt += 1
                    if attr == 'bias_regularizer':
                        bias_regularizer_cnt += 1

        inputs_names = layer_config['inbound_nodes'][0]
        inputs_names = list(map(lambda x: x[0], inputs_names))

        inputs = list(map(lambda name: layers[layer_name_to_ix[name]], inputs_names))

        new_layer = new_layer(inputs[0] if len(inputs) == 1 else inputs)

        layers.append(new_layer)

    new_model = keras.Model(inputs=layers[0], outputs=layers[-1])
    assert len(model.layers) == len(new_model.layers)

    for layer_ix in range(1, len(model.layers)):
        weights = model.layers[layer_ix].get_weights()
        try:
            new_model.layers[layer_ix].set_weights(weights)
        except ValueError:
            if copy_bnorm_weights:
                new_model.layers[layer_ix].src_bn.set_weights(weights)
                new_model.layers[layer_ix].trg_bn.set_weights(weights)

        new_model.layers[layer_ix].trainable = model.layers[layer_ix].trainable

    sys.stderr.write("Successfully set {} kernel_regularizers to {}.\n".format(kernel_regularizer_cnt, kernel_regularizer))
    sys.stderr.write("Successfully set {} bias_initializers to {}.\n".format(bias_regularizer_cnt, bias_regularizer))

    return new_model
