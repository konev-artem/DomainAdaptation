import tensorflow as tf
import tensorflow.keras as keras


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


def make_batch_normalization_layers_domain_specific(model, domain_variable, copy_bnorm_weights=True):
    """
    This function replaces all BatchNormalization layers with DomainSpecificBatchNormalization layers.
    You have to manually change domain_variable in order to switch between source and target modes.
    The weights are copied from old model into new model.
    :param model: Old model
    :param domain_variable: This tf.Variable(dtype=bool) indicates if the layer is used in source or target mode
    :param copy_bnorm_weights: Whether to copy old BatchNormalization weights into new ones
    :return: New model
    """
    layers = []
    layer_name_to_ix = dict()

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

    return new_model
