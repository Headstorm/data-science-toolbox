import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class MyCallback(tf.keras.callbacks.Callback):
  def onEpochEnd(self, epoch, logs={}):
    if(logs.get('accuracy')>0.85):
      print("\nReached 85% accuracy so cancelling training!")
      self.model.stop_training = True

def mnist_func():
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    np.set_printoptions(linewidth=200)

    # show first image
    plt.imshow(training_images[0])
    plt.show()

    print(training_images[0])
    print(training_labels[0])

    # normalize pixel values for model
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # create simple dense model
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    # compile model
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # fit model
    model.fit(training_images, training_labels, epochs=100, callbacks=[MyCallback()])

    # evaluate test data
    model.evaluate(test_images, test_labels)

    # predict on test images
    classifications = model.predict(test_images)

    print(classifications[0])
    print(test_labels[0])