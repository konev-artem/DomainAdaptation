import tensorflow as tf
import tensorflow.keras as keras
import sys
import numpy as np
import os


imagenet_means = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_stds  = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def proprocess_input(X_batch):
    X_batch = X_batch.copy()

    X_batch /= 255.
    X_batch[..., 0], X_batch[..., 2] = X_batch[..., 2], X_batch[..., 0]
    X_batch -= imagenet_means
    return X_batch


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''
    From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group==0
    assert c_o % group==0

    if group == 1:
        conv_layer = keras.layers.Conv2D(
            kernel_size=(k_h, k_w,),
            filters=c_o,
            strides=(s_h, s_w,),
            padding=padding,
            kernel_initializer=keras.initializers.Constant(kernel),
            bias_initializer=keras.initializers.Constant(biases)
        )
        
        conv = conv_layer(input)
    else:
        input_groups =  tf.split(input, group, 3)
        kernel_groups = np.split(kernel, group, 3)
        biases_groups = np.split(biases, group, -1)
        
        conv_layers = []
        for kernel_group, biases_group in zip(kernel_groups, biases_groups):
            conv_layers.append(keras.layers.Conv2D(
                kernel_size=(kernel_group.shape[:2]),
                filters=kernel_group.shape[-1],
                strides=(s_h, s_w,),
                padding=padding,
                kernel_initializer=keras.initializers.Constant(kernel_group),
                bias_initializer=keras.initializers.Constant(biases_group)))
            
        output_groups = [
            conv_layer(i)
            for conv_layer, i in zip(conv_layers, input_groups)
        ]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def AlexNet(input_shape=(227, 227, 3), weights='imagenet', include_top=False, **kwargs):
    assert weights == 'imagenet', 'Only imagenet weights are supported'
    if input_shape != (227, 227, 3):
        sys.stderr.write("AlexNet was pretrained with input_shape=(227, 227, 3), results are not guaranteed otherwise\n")
    for k in kwargs.keys():
        sys.stderr.write("AlexNet will ignore parameter {}\n".format(k))

    os.system('wget https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy -nc -O bvlc_alexnet.npy')
    net_data = np.load("bvlc_alexnet.npy", encoding="latin1", allow_pickle=True).item()

    x = tf.keras.Input(shape=input_shape, dtype=tf.float32)

    #conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = net_data["conv1"][0]
    conv1b = net_data["conv1"][1]
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                            depth_radius=radius,
                                            alpha=alpha,
                                            beta=beta,
                                            bias=bias)

    # #maxpool1
    # #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = net_data["conv2"][0]
    conv2b = net_data["conv2"][1]
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                            depth_radius=radius,
                                            alpha=alpha,
                                            beta=beta,
                                            bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = net_data["conv3"][0]
    conv3b = net_data["conv3"][1]
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = net_data["conv4"][0]
    conv4b = net_data["conv4"][1]
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)


    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = net_data["conv5"][0]
    conv5b = net_data["conv5"][1]
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #fc6
    #fc(4096, name='fc6')
    fc6W = net_data["fc6"][0]
    fc6b = net_data["fc6"][1]
    fc6 = keras.layers.Dense(
        units=4096,
        activation='relu',
        kernel_initializer=keras.initializers.Constant(fc6W),
        bias_initializer=keras.initializers.Constant(fc6b),
    )(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]))

    #fc7
    #fc(4096, name='fc7')
    fc7W = net_data["fc7"][0]
    fc7b = net_data["fc7"][1]
    fc7 = keras.layers.Dense(
        units=4096,
        activation='relu',
        kernel_initializer=keras.initializers.Constant(fc7W),
        bias_initializer=keras.initializers.Constant(fc7b),
    )(fc6)

    #fc8
    #fc(1000, relu=False, name='fc8')
    fc8W = net_data["fc8"][0]
    fc8b = net_data["fc8"][1]
    fc8 = keras.layers.Dense(
        units=1000,
        activation='linear',
        kernel_initializer=keras.initializers.Constant(fc8W),
        bias_initializer=keras.initializers.Constant(fc8b),
    )(fc7)

    #prob
    #softmax(name='prob'))
    prob = tf.nn.softmax(fc8)

    model = tf.keras.Model(
        inputs=x,
        outputs=prob if include_top else fc7
    )

    for layer in model.layers:
        layer.trainable = True

    return model