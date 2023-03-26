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


# convert from click array position to axis and rotation angles
def click_to_rotation(click_image_x, click_image_y, all_axis_angles, all_rot_angles):
    # transform from image coordinate to cartesian coordinate
    print(f'Position in image coordinate:', click_image_x, click_image_y)
    array_x, array_y = img2array(click_image_x, click_image_y, len(all_rot_angles) * 2 - 1)
    # print('Position in array coordinate:', array_x, array_y)
    # transform from cartesian coordinate to polar coordinate
    length, angle = cart2pol(array_x, array_y)
    # print('Position in polar coordinate:', length, angle)
    # recover rotation axis angle and the actual rotate angle around that axis
    axis_angle_index = np.where(all_axis_angles == angle)[0][0]
    rot_angle_index = int(length)

    return axis_angle_index, rot_angle_index


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
    angle = math.degrees(np.arctan2(y, x))
    return (length, angle)


def pol2cart(length, angle):
    x = length * math.cos(math.radians(angle))
    y = length * math.sin(math.radians(angle))
    return (x, y)


def array2img(array_pos, img_size):
    # in image coordinate, x are columns, y are rows
    # in array coordinate, x are rows, y are columns
    x_img = array_pos[1] + img_size // 2
    y_img = array_pos[0] + img_size // 2

    return (x_img, y_img)


def img2array(x_img, y_img, img_size):
    # in image coordinate, x are columns, y are rows
    # in array coordinate, x are rows, y are columns
    x_array = x_img - (img_size // 2)
    y_array = -y_img + (img_size // 2)   # because plot was reversed in y

    return (x_array, y_array)


# create raster image that holds custom displacement of results
def point_cloud_result_to_polar_image(
    accuracies, plane, all_axis_angles, all_rot_angles, block_size=1, scale=1
):

    assert (
        len(all_axis_angles) == accuracies.shape[0] and len(all_rot_angles) == accuracies.shape[1]
    )
    # outside place filled with NaN
    image_np = np.full(
        (len(all_rot_angles) * 2 * scale - 1, len(all_rot_angles) * 2 * scale - 1), np.nan
    )
    quaternion_np = np.full(
        (len(all_rot_angles) * 2 * scale - 1, len(all_rot_angles) * 2 * scale - 1, 4), np.nan
    )
    # initialize assignment dictionary
    assignment = {}
    # assign the result to polar raster image
    for i, cur_axis_angle in enumerate(all_axis_angles):
        for j, cur_rot_angle in enumerate(all_rot_angles):
            # coordinate in raster image
            array_pos = pol2cart(j, cur_axis_angle)
            array_pos = (round(array_pos[0] * scale), round(array_pos[1] * scale))
            # array coordinate to image coordinate
            image_pos = array2img(array_pos, len(image_np))
            # when current position has been occupied by other, we skip
            if not np.isnan(image_np[image_pos]):
                continue
            # assign value
            image_np[image_pos] = accuracies[i, j]
            # save assignment for easier reverse tracking
            assignment[image_pos] = (i, j)

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
            quaternion_np[image_pos] = quaternions.axis_angle_to_quaternion(
                cur_axis, cur_rot_angle, unit='degree'
            )
    # resize and convert to image
    image_np = np.kron(image_np, np.ones((block_size, block_size)))
    quaternion_np = np.kron(quaternion_np, np.ones((block_size, block_size)))

    return image_np, quaternion_np, assignment


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
