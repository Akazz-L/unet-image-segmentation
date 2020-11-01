from model import *
from data import *

import tifffile as tiff
import datetime

TEST_TIFF = "test-volume.tif"

if __name__ == "__main__":
    # Load model
    model = unet_model(pretrained_weights='unet_membrane.hdf5')

    # Load test imgs
    imgs = tiff.imread(TEST_TIFF)

    # Rescale img
    imgs = imgs / 255.0

    # Predict mask labels
    predictions = model.predict(imgs, batch_size = 1,verbose=1)

    # Labels post-processing
    predictions = predictions.squeeze()
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    predictions = (predictions * 255).astype(np.uint8)

    # Save into tiff file
    result_name = "test-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".tif"
    tiff.imwrite(result_name, predictions)
