import os
import tensorflow.keras as keras


def train(backbone,
          classifier_head,
          domain_head,
          optimizer,
          loss,
          metrics,
          train_generator,
          epochs,
          validation_generator=None,
          save_checkpoints=True,
          monitor='loss',
          checkpoint_dir='./',
          model_name='some_model',
          save_best_only=True,
          reduce_lr=False,
          reduce_lr_factor=0.1,
          **kwargs):

    model = keras.Model(inputs=backbone.inputs,
                        outputs=[classifier_head(backbone.outputs),
                                 domain_head(backbone.outputs)])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = []
    if save_checkpoints:
        filepath = os.path.join(checkpoint_dir, model_name + '-{epoch:03d}.hdf5')
        callbacks.append(
            keras.callbacks.ModelCheckpoint(filepath, monitor=monitor, save_best_only=save_best_only)
        )

    if reduce_lr:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=reduce_lr_factor))

    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        **kwargs
    )
    return model


def train_DANN(backbone,
               classifier_head,
               domain_head, train_generator,
               epochs,
               validation_generator=None,
               **kwargs):

    adam = keras.optimizers.Adam()
    return train(backbone=backbone,
                 classifier_head=classifier_head,
                 domain_head=domain_head,
                 optimizer=adam,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'],
                 train_generator=train_generator,
                 epochs=epochs,
                 validation_generator=validation_generator,
                 **kwargs)
