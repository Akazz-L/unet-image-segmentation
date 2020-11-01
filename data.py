#import tifffile as tiff
import os
import matplotlib.pyplot as plt
from skimage.io import imsave

from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.utils import Sequence
import numpy as np


DATA_PATH = "data/"
TRAIN_PATH = DATA_PATH + "train/image/"
LABEL_PATH = DATA_PATH + "train/label/"
TEST_PATH = DATA_PATH + "test/"

TRAIN_TIFF = "train-volume.tif"
LABEL_TIFF = "train-labels.tif"


class DataLoader(Sequence):
    """ Helper generator to iterate over the data (as Numpy arrays)"""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, rescale = None):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        if rescale is None:
            self.rescale = 1
        else:
            self.rescale = rescale

    def __len__(self):
        """ Returns the # we can iterate through the dataloader and get data batch """
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """ Returns (inputs,targets) corresponding to batch #idx
        Parameter
        -
        idx : int
            element index or batch index
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        # 4 dimensions array containing a batch of grayscale images
        # Using grayscale color_mode in load_img will load a (width,height) shape
        # it is necessary to add one dimension for the pixel values
        batch_input_img = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.asarray(load_img(path, color_mode="grayscale", target_size=self.img_size))
            batch_input_img[j] = np.expand_dims(img * self.rescale, 2)

        batch_target_img = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = np.asarray(load_img(path, color_mode="grayscale", target_size=self.img_size))
            batch_target_img[j] = np.expand_dims(img * self.rescale, 2)


        return batch_input_img, batch_target_img







def get_train_data_paths(input_dir, target_dir):
    """ Return the list of training images and label masks paths for future dynamic image loading
        Parameter
        -
        input_dir : str
            Training input images directory
        target_dir : str
            Training target masks directory
    """

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".png")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png")
        ]
    )

    return input_img_paths, target_img_paths


def init_dataset(show = False):
    """ Initialize the raw dataset containing tiff files to  train/test folder architecture with .png file
    Parameter
    -
    show : boolean, optional
        Plot the 5 first training images and label masks
    """

    if not os.path.exists(DATA_PATH):
        os.makedirs(TRAIN_PATH)
        os.makedirs(LABEL_PATH)
        os.makedirs(TEST_PATH)


    img_train = tiff.imread(TRAIN_TIFF)
    img_label = tiff.imread(LABEL_TIFF)

    print("Train shape : {}".format(img_train.shape))
    print("Label shape : {}".format(img_label.shape))


    for index, (img, mask) in enumerate(zip(img_train, img_label)):
        img_name = TRAIN_PATH + str(index) + ".png"
        mask_name = LABEL_PATH + str(index) + ".png"
        imsave(img_name, img)
        imsave(mask_name, mask)

        if show and index < 5:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img, cmap='gray')
            ax[1].imshow(mask, cmap='gray')
            plt.show()

def train_generator(batch_size,train_path, image_folder, mask_folder, aug_dict,
                    image_color_mode="grayscale", mask_color_mode="grayscale",
                    image_save_prefix="image", mask_save_prefix="mask",
                    save_to_dir = None, target_size = (512,512), seed = 1):

    """
    Generate images and masks using an augmentation dict including the enabled transformations
    Seed used to apply the same transformation on images and their masks

    :param batch_size: int
    :param train_path: str
    :param image_folder: str
    :param mask_folder: str
    :param aug_dict: dict
    :param image_color_mode: str
    :param mask_color_mode: str
    :param save_to_dir: str
    :param target_size:
    :param seed:
    :return: Generator yielding (x,y) where x is a batch of images and y is the corresponding batch of masks
    """


    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    # Dynamically generate augmented images
    image_generator = image_datagen.flow_from_directory(directory = train_path,
                                                        target_size = target_size,
                                                        color_mode = image_color_mode,
                                                        class_mode = None,
                                                        classes = [image_folder],
                                                        batch_size = batch_size,
                                                        save_to_dir = save_to_dir,
                                                        save_prefix=image_save_prefix,
                                                        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(directory = train_path,
                                                        target_size = target_size,
                                                        color_mode = mask_color_mode,
                                                        class_mode = None,
                                                        classes = [mask_folder],
                                                        batch_size = batch_size,
                                                        save_to_dir = save_to_dir,
                                                        save_prefix = mask_save_prefix,
                                                        seed = seed)

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        #Rescale to [0,1]
        img = img / 255
        mask = mask/ 255
        yield (img, mask)



