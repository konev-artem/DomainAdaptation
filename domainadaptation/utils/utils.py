import numpy as np

def get_features_and_labels(model, generator, count):
    features = []
    labels = []
    for i in range(count):
        X, y = next(generator)
        features.append(model(X))
        labels.append(np.argmax(y, axis=-1))

    return np.vstack(features), np.hstack(labels)