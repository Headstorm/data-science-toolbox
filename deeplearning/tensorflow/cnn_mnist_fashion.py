import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models


def mnist_fashion_cnn():
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images=training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images=test_images/255.0
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(training_images, training_labels, epochs=5)
    test_loss = model.evaluate(test_images, test_labels)


    # It's likely gone up to about 93% on the training data and 91% on the validation data.
    #
    # That's significant, and a step in the right direction!
    #
    # Try running it for more epochs -- say about 20, and explore the results! But while the results might seem really good, the validation results may actually go down, due to something called 'overfitting' which will be discussed later.
    #
    # (In a nutshell, 'overfitting' occurs when the network learns the data from the training set really well, but it's too specialised to only that data, and as a result is less effective at seeing *other* data. For example, if all your life you only saw red shoes, then when you see a red shoe you would be very good at identifying it, but blue suade shoes might confuse you...and you know you should never mess with my blue suede shoes.)
    #
    # Then, look at the code again, and see, step by step how the Convolutions were built:

    # Step 1 is to gather the data. You'll notice that there's a bit of a change here in that the training data needed to be reshaped. That's because the first convolution expects a single tensor containing everything, so instead of 60,000 28x28x1 items in a list, we have a single 4D list that is 60,000x28x28x1, and the same for the test images. If you don't do this, you'll get an error when training as the Convolutions do not recognize the shape.
    #
    #
    #
    # ```
    # import tensorflow as tf
    # mnist = tf.keras.datasets.fashion_mnist
    # (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # training_images=training_images.reshape(60000, 28, 28, 1)
    # training_images=training_images / 255.0
    # test_images = test_images.reshape(10000, 28, 28, 1)
    # test_images=test_images/255.0
    # ```
    #

    # Next is to define your model. Now instead of the input layer at the top, you're going to add a Convolution. The parameters are:
    #
    # 1. The number of convolutions you want to generate. Purely arbitrary, but good to start with something in the order of 32
    # 2. The size of the Convolution, in this case a 3x3 grid
    # 3. The activation function to use -- in this case we'll use relu, which you might recall is the equivalent of returning x when x>0, else returning 0
    # 4. In the first layer, the shape of the input data.
    #
    # You'll follow the Convolution with a MaxPooling layer which is then designed to compress the image, while maintaining the content of the features that were highlighted by the convlution. By specifying (2,2) for the MaxPooling, the effect is to quarter the size of the image. Without going into too much detail here, the idea is that it creates a 2x2 array of pixels, and picks the biggest one, thus turning 4 pixels into 1. It repeats this across the image, and in so doing halves the number of horizontal, and halves the number of vertical pixels, effectively reducing the image by 25%.
    #
    # You can call model.summary() to see the size and shape of the network, and you'll notice that after every MaxPooling layer, the image size is reduced in this way.

    # ```
    # model = tf.keras.models.Sequential([
    #   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    #   tf.keras.layers.MaxPooling2D(2, 2),
    # ```
    #

    # Add another convolution

    # ```
    #   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #   tf.keras.layers.MaxPooling2D(2,2)
    # ```
    #

    # Now flatten the output. After this you'll just have the same DNN structure as the non convolutional version
    #
    # ```
    #   tf.keras.layers.Flatten(),
    # ```
    #

    # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
    #
    #
    #
    # ```
    #   tf.keras.layers.Dense(128, activation='relu'),
    #   tf.keras.layers.Dense(10, activation='softmax')
    # ])
    # ```
    #

    # Now compile the model, call the fit method to do the training, and evaluate the loss and accuracy from the test set.
    #
    #
    #
    # ```
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.fit(training_images, training_labels, epochs=5)
    # test_loss, test_acc = model.evaluate(test_images, test_labels)
    # print(test_acc)
    # ```
    #

    # # Visualizing the Convolutions and Pooling
    #
    # This code will show us the convolutions graphically. The print (test_labels[;100]) shows us the first 100 labels in the test set, and you can see that the ones at index 0, index 23 and index 28 are all the same value (9). They're all shoes. Let's take a look at the result of running the convolution on each, and you'll begin to see common features between them emerge. Now, when the DNN is training on that data, it's working with a lot less, and it's perhaps finding a commonality between shoes based on this convolution/pooling combination.

    print(test_labels[:100])

    f, axarr = plt.subplots(3,4)
    FIRST_IMAGE = 0
    SECOND_IMAGE = 7
    THIRD_IMAGE = 26
    CONVOLUTION_NUMBER = 1
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
    for x in range(0,4):
      f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
      axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
      axarr[0,x].grid(False)
      f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
      axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
      axarr[1,x].grid(False)
      f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
      axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
      axarr[2,x].grid(False)

    plt.show()
