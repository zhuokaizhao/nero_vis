import math
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui
from PySide6.QtGui import QPixmap, QFont
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import manifold
from sklearn.manifold import TSNE
import umap

import quaternions
import nero_utilities


# helper function on normalizing low dimension points within [-1, 1] sqaure
def normalize_low_dim_result(low_dim):
    # to preserve shape, largest bound are taken from either axis
    all_min = np.min(low_dim)
    all_max = np.max(low_dim)

    # lerp each point's x and y into [-1, 1]
    new_low_dim = np.zeros(low_dim.shape)
    new_low_dim[:, 0] = nero_utilities.lerp(low_dim[:, 0], all_min, all_max, -1, 1)
    new_low_dim[:, 1] = nero_utilities.lerp(low_dim[:, 1], all_min, all_max, -1, 1)

    return new_low_dim


# function used as model icon
def draw_circle(painter, center_x, center_y, radius, color):
    # set up brush and pen
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setBrush(QtGui.QColor(color))
    painter.setPen(QtGui.QColor(color))
    center = QtCore.QPoint(center_x, center_y)
    # draw the circle
    painter.drawEllipse(center, radius, radius)
    painter.end()


def cart2pol(x, y):
    length = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    return (length, angle)


def pol2cart(length, angle):
    x = int(length * math.cos(math.radians(angle)))
    y = int(length * math.sin(math.radians(angle)))
    return (x, y)


def cart2img(cart, img_size):
    x_img = cart[0] + img_size // 2
    y_img = cart[1] + img_size // 2

    return (x_img, y_img)


def img2cart(x_img, y_img, img_size):
    x_cart = x_img - img_size // 2
    y_cart = y_img - img_size // 2

    return (x_cart, y_cart)


# create raster image that holds custom displacement of results
def process_point_cloud_result(accuracies, plane, all_axis_angles, all_rot_angles, block_size=1):

    assert (
        len(all_axis_angles) == accuracies.shape[0] and len(all_rot_angles) == accuracies.shape[1]
    )

    image_np = np.zeros((len(all_rot_angles) * 2, len(all_rot_angles) * 2))
    # outside place filled with NaN
    quaternion_np = np.full((len(all_rot_angles) * 2, len(all_rot_angles) * 2, 4), np.nan)

    for i, cur_axis_angle in enumerate(all_axis_angles):
        for j, cur_rot_angle in enumerate(all_rot_angles):
            # coordinate in raster image
            cartesian_coordinate = pol2cart(j, cur_axis_angle)
            # cartesian coordinate to image coordinate
            image_coordinate = cart2img(cartesian_coordinate, len(image_np))
            image_np[image_coordinate] = accuracies[i, j]

            # also save quaternion for each position
            # get corresponding axis from plane information and axis angle
            axis_angle_rad = cur_axis_angle / 180 * np.pi
            start_axis = {
                'xy': np.matrix([[1], [0], [0]]),
                'xz': np.matrix([[1], [0], [0]]),
                'yz': np.matrix([[0], [1], [0]]),
            }
            rot_matrix = {
                'xy': np.matrix(
                    [
                        [math.cos(axis_angle_rad), -math.sin(axis_angle_rad), 0],
                        [math.sin(axis_angle_rad), math.cos(axis_angle_rad), 0],
                        [0, 0, 1],
                    ]
                ),
                'xz': np.matrix(
                    [
                        [math.cos(axis_angle_rad), 0, math.sin(axis_angle_rad)],
                        [0, 1, 0],
                        [-math.sin(axis_angle_rad), 0, math.cos(axis_angle_rad)],
                    ]
                ),
                'yz': np.matrix(
                    [
                        [1, 0, 0],
                        [0, math.cos(axis_angle_rad), -math.sin(axis_angle_rad)],
                        [0, math.sin(axis_angle_rad), math.cos(axis_angle_rad)],
                    ]
                ),
            }
            # rotate corresponding start axis by corresponding rotation matrix
            cur_axis = np.squeeze(np.array(rot_matrix[plane] * start_axis[plane]))
            # convert axis-angle rotation to quaternion and save
            quaternion_np[image_coordinate] = quaternions.axis_angle_to_quaternion(
                cur_axis, cur_rot_angle, unit='degree'
            )

    # resize and convert to image
    image_np = np.kron(image_np, np.ones((block_size, block_size)))
    quaternion_np = np.kron(quaternion_np, np.ones((block_size, block_size)))

    return image_np, quaternion_np


def dimension_reduce(method, high_dim, target_dim):

    if method == 'PCA':
        pca = PCA(n_components=target_dim, svd_solver='full')
        low_dim = pca.fit_transform(high_dim)
    elif method == 'ICA':
        ica = FastICA(n_components=target_dim, random_state=12)
        low_dim = ica.fit_transform(high_dim)
    elif method == 'ISOMAP':
        low_dim = manifold.Isomap(n_neighbors=5, n_components=target_dim, n_jobs=-1).fit_transform(
            high_dim
        )
    elif method == 't-SNE':
        low_dim = TSNE(n_components=target_dim, n_iter=250).fit_transform(high_dim)
    elif method == 'UMAP':
        low_dim = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=target_dim).fit_transform(
            high_dim
        )

    return low_dim


# compute intensity values with either average or variance
def compute_intensity(input_data, method):
    if method == 'mean':
        intensity = np.mean(input_data, axis=1)
    elif method == 'variance':
        intensity = np.var(input_data, axis=1)

    return intensity


# helper functions on managing the database
def load_from_cache(name, cache):
    # if it exists
    if name in cache.keys():
        return cache[name], True
    else:
        print(f'No precomputed result named {name}')
        return np.zeros(0), False


def save_to_cache(names, contents, cache, path):
    if type(names) == list:
        assert type(contents) == list
        assert len(names) == len(contents)
        for i in range(len(names)):
            cache[names[i]] = contents[i]
    else:
        # replace if exists
        cache[names] = contents

    np.savez(path, **cache)