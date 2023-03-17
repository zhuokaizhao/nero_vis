import numpy as np
from PySide6 import QtCore, QtGui


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
