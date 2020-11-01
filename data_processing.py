import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries


def decompose_mask(mask, display = False):
    """ Return binary masks for each class from a single mask wherein class has different values

    Parameters
    -----
    mask : 2D-array of shape (image_width, image_height)
      Segmentation mask where each instance or class has a different value

    display : boolean
        Enable plot of classes masks horizontally

    Returns
    -----
    class_masks : 3D-array of shape (number_of_class, image_width, image_height)
        Array of binary masks
    """

    nClasses = np.unique(mask)
    class_masks = np.zeros((len(nClasses), mask.shape[0], mask.shape[1]))

    for index, class_ in enumerate(nClasses):
        class_mask = (mask == class_).astype(int)
        class_masks[index] = class_mask


    if display:
        fig, ax = plt.subplots(1, len(class_masks))
        fig.suptitle('Classes masks')
        for i, mask in enumerate(class_masks):
            ax[i].imshow(mask, cmap="gray")

        plt.show()

    return class_masks


# https://jaidevd.github.io/posts/weighted-loss-functions-for-instance-segmentation/
# To learn the separation between objects, a weighted map must be introduced
# to penalize the loss near the boundaries of regions


w0 = 10
sigma = 5

def make_weight_map(masks):
    """
    Generate the weight maps as specified in the UNet paper
    for a set of binary masks.

    Parameters
    ----------
    masks: array-like
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice of the matrix along the 0th axis represents one binary mask.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)

    """
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    distMap = np.zeros((nrows * ncols, masks.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = X1.ravel(), Y1.ravel()

    #X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')

        X2, Y2 = np.nonzero(bounds) #  indexes of non zero values (borders)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)

    ix = np.arange(distMap.shape[0])
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    ZZ = xBLoss + loss
    return ZZ