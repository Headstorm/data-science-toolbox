import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import os
import zipfile
import random
import math
from shutil import copyfile
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

# extract all from path to the extraction location
def unzip(path, to):
    zip_ref = zipfile.ZipFile(path, 'r')
    zip_ref.extractall(to)
    zip_ref.close()

# gets a relative pathname from the pwd
def get_relative_path_name(path_extension = ''):
    this_dir = os.path.dirname(__file__)
    return this_dir + path_extension


def split_data(source, training, testing, split_size):
    # split size must be between 0 and 1
    if split_size >= 1 or split_size <= 0:
        return

    images = random.sample(os.listdir(source), len(os.listdir(source)))
    num_images = len(images)

    while split_size < 1:
        split_size *= 10

    split_size = math.floor(split_size) + 1

    idx = 0
    while idx < num_images:
        if os.path.getsize('{}/{}'.format(source, images[idx])):
            if idx % split_size == 0:
                copyfile('{}/{}'.format(source, images[idx]), '{}/{}'.format(testing, images[idx]))
            else:
                copyfile('{}/{}'.format(source, images[idx]), '{}/{}'.format(training, images[idx]))
        idx += 1


def cat_or_dog(zip_path, extraction_dir):
    # if we haven't fetched the source images, do so
    if not os.path.isdir('{}/PetImages'.format(extraction_dir)):
        unzip(zip_path, extraction_dir)

    if not os.path.isdir(get_relative_path_name('/img')):
        os.mkdir(get_relative_path_name('/img'))

    if not os.path.isdir(get_relative_path_name('/img/training')):
        os.mkdir(get_relative_path_name('/img/training'))

    if not os.path.isdir(get_relative_path_name('/img/testing')):
        os.mkdir(get_relative_path_name('/img/testing'))

    animals = ['Cat', 'Dog']
    for animal in animals:
        SOURCE_DIR = extraction_dir + '/PetImages/{}'.format(animal)
        TRAIN_DIR = get_relative_path_name('/img/training/{}'.format(animal.lower()))
        TEST_DIR = get_relative_path_name('/img/testing/{}'.format(animal.lower()))

        if not os.path.isdir(TRAIN_DIR):
            os.mkdir(TRAIN_DIR)
        if not os.path.isdir(TEST_DIR):
            os.mkdir(TEST_DIR)

        split_data(SOURCE_DIR, TRAIN_DIR, TEST_DIR, 0.90)

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cat') and 1 for the other ('dog')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['acc'])

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        get_relative_path_name('/img/training'),  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=10,  # FOR COURSERA we must use batch size 10
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    # Flow training images in batches of 128 using train_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
        get_relative_path_name('/img/testing'),  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    history = model.fit_generator(train_generator,
                                  epochs=2,
                                  verbose=1,
                                  validation_data=validation_generator)

    print(history)

    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.figure()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")

    plt.title('Training and validation loss')

    # Desired output. Charts with training and validation metrics. No crash :)

# global var for username
USER = 'ericbaumann'

# unzip image files to this source folder to be split into test/train
EXTRACTION_PATH = get_relative_path_name()
ZIP_PATH = '/Users/{}/Downloads/cats-and-dogs.zip'.format(USER)

if __name__ == "__main__":
    cat_or_dog(ZIP_PATH, EXTRACTION_PATH)



