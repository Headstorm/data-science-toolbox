from cnn_mnist_fashion import mnist_fashion_cnn
from horse_or_human import horse_or_human
from cat_or_dog import cat_or_dog, get_relative_path_name


# global var for username
USER = 'ericbaumann'

# unzip image files to this source folder to be split into test/train
EXTRACTION_PATH = get_relative_path_name()
ZIP_PATH = '/Users/{}/Downloads/cats-and-dogs.zip'.format(USER)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


if __name__ == '__main__':
    # mnist_fashion_cnn()
    # horse_or_human()
    cat_or_dog(ZIP_PATH, EXTRACTION_PATH)

