# the script contains utility functions
import copy
import torch
import numpy as np
from PySide6 import QtGui


# some helper functions
def qt_image_to_tensor(img, share_memory=False):
    """
    Creates a PyTorch Tensor from a QImage via converting to Numpy array first.

    If share_memory is True, the numpy array and the QImage is shared.
    Be careful: make sure the numpy array is destroyed before the image,
                otherwise the array will point to unreserved memory!!
    """
    assert isinstance(img, QtGui.QImage), "img must be a QtGui.QImage object"
    if img.format() == QtGui.QImage.Format.Format_Grayscale8:
        num_channel = 1
    elif img.format() == QtGui.QImage.Format.Format_RGB32:
        # RGBA
        num_channel = 4

    img_size = img.size()
    buffer = img.constBits()

    # Sanity check
    n_bits_buffer = len(buffer) * 8
    n_bits_image  = img_size.width() * img_size.height() * img.depth()

    assert n_bits_buffer == n_bits_image, "size mismatch: {} != {}".format(n_bits_buffer, n_bits_image)

    # Note the different width height parameter order!
    arr = np.ndarray(shape  = (img_size.height(), img_size.width(), num_channel),
                     buffer = buffer,
                     dtype  = np.uint8)

    if share_memory:
        return torch.from_numpy(arr)
    else:
        return torch.from_numpy(copy.deepcopy(arr))


def tensor_to_qt_image(img_pt):
    img_np = img_pt.numpy()
    if img_np.shape[-1] == 1:
        # qt image uses width, height
        img_qt = QtGui.QImage(img_np, img_np.shape[1], img_np.shape[0], QtGui.QImage.Format_Grayscale8)
    elif img_np.shape[-1] == 3:
        img_qt = QtGui.QImage(img_np, img_np.shape[1], img_np.shape[0], QtGui.QImage.Format_RGB888)

    return img_qt