from domainadaptation.experiment import DANNExperiment

exp = DANNExperiment(config={
    'backbone': {
        'type': 'resnet50',
        'num_trainable_layers': 7,
        'img-size': (224, 224),
        'weights': 'imagenet',
        'pooling': 'max'},
    'dataset': {
        'classes': 31,
        'path': './data/office31',
        'augmentations': {},
        'source': "amazon",
        'target': 'dslr'},
    'batch_size': 16,
    'epochs': 10,
    'steps': 4000 // 16,
    'grads_update_freq': 1,
})

exp.experiment_domain_adaptation()
