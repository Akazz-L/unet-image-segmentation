from model import *
from data import *
import os
import datetime
from keras.callbacks import ModelCheckpoint

input_dir = "/opt/ml/input/data/train/image/"
target_dir = "/opt/ml/input/data/train/label/"
output_dir = "/opt/ml/model/"

if __name__ == '__main__':
    list_files = os.listdir("/opt/ml/input/")
    input_img_paths, target_img_paths = get_train_data_paths(input_dir, target_dir)
    # Load data
    train_datagen = DataLoader(batch_size = 1, 
                            img_size = (512,512), 
                            input_img_paths = input_img_paths,
                            target_img_paths = target_img_paths,
                                rescale = 1 / 255)

    # Load and train model
    model = unet_model()

    callbacks = []
    weights_path = output_dir +'unet_membrane' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.hdf5'
    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=1, save_best_only=True)
    callbacks.append(checkpoint)
    model.fit(train_datagen, steps_per_epoch=30, epochs=10, verbose=1, callbacks = callbacks)
