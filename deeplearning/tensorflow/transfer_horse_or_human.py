import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# extract all from path to the extraction location
def unzip(path, to):
    zip_ref = zipfile.ZipFile(path, 'r')
    zip_ref.extractall(to)
    zip_ref.close()

# gets a relative pathname from the pwd
def get_relative_path_name(path_extension):
    this_dir = os.path.dirname(__file__)
    return this_dir + path_extension

def horse_or_human(path, validation_path):
    if(not os.path.isdir(get_relative_path_name('/img'))):
        unzip(path, get_relative_path_name('/img'))

    if (not os.path.isdir(get_relative_path_name('/validation-img'))):
        unzip(validation_path, get_relative_path_name('/validation-img'))

    train_horse_dir = os.path.join(get_relative_path_name('/img/horses'))
    train_human_dir = os.path.join(get_relative_path_name('/img/humans'))

    train_horse_names = os.listdir(train_horse_dir)
    train_human_names = os.listdir(train_human_dir)

    validation_horse_dir = os.path.join(get_relative_path_name('/validation-img/horses'))
    validation_human_dir = os.path.join(get_relative_path_name('/validation-img/humans'))

    validation_horse_names = os.listdir(validation_horse_dir)
    validation_human_names = os.listdir(validation_human_dir)

    # get current figure
    fig = plt.gcf()
    fig.set_size_inches(cols * 4, rows * 4)

    # init pic index to a random image and 8 prev images
    pic_index = random.randint(0, min(len(train_horse_names), len(train_human_names)))
    next_horse_pix = [os.path.join(train_horse_dir, fname)
                    for fname in train_horse_names[pic_index-8:pic_index]]
    next_human_pix = [os.path.join(train_human_dir, fname)
                    for fname in train_human_names[pic_index-8:pic_index]]

    for i, img_path in enumerate(next_horse_pix+next_human_pix):
      # Set up subplot; subplot indices start at 1
      sp = plt.subplot(rows, cols, i + 1)
      sp.axis('Off') # Don't show axes (or gridlines)

      img = mpimg.imread(img_path)
      plt.imshow(img)

    plt.show()

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
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fifth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    validation_datagen = ImageDataGenerator(rescale=1 / 255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        get_relative_path_name('/img'),  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    # Flow training images in batches of 128 using train_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
       get_relative_path_name('/validation-img'),  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    history = model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=8)

    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]
    # visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
    # Let's prepare a random input image from the training set.
    horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
    human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
    img_path = random.choice(horse_img_files + human_img_files)

    img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
    x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= 255

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers[1:]]

    # Now let's display our representations
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            n_features = feature_map.shape[-1]  # number of features in feature map
            # The feature map has shape (1, size, size, n_features)
            size = feature_map.shape[1]
            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                # Postprocess the feature to make it visually palatable
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # We'll tile each filter into this big horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x
            # Display the grid
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

# global var for username
USER = 'ericbaumann'

# vars for matplotlib
rows = 4
cols = 4

# output classes
HORSE = 0
HUMAN = 1

if __name__ == "__main__":
    #horse_or_human('/Users/{}/Downloads/horse-or-human.zip'.format(USER), '/Users/{}/Downloads/horse-or-human.zip'.format(USER))
    train_happy_sad()
