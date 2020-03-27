import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tester import Tester


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test.resize((*x_test.shape, 1))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    datagen = ImageDataGenerator()
    test_generator = datagen.flow(x_test, y_test, batch_size=32)

    tester = Tester()
    accuracies = tester.test(model, test_generator)
    tester.bootstrap(accuracies, sz=len(accuracies))
