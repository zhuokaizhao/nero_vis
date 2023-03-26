from pyqtgraph.Qt import QtCore, QtGui
from PySide6 import QtWidgets
import pyqtgraph.opengl as gl
import numpy as np
import sys


app = QtWidgets.QApplication([])
point_cloud_widget = gl.GLViewWidget()
point_cloud_widget.opts['distance'] = 10
point_cloud_widget.show()
point_cloud_widget.setBackgroundColor('black')
point_cloud_widget.setWindowTitle('pyqtgraph example: GLScatterPlotItem')

# axis - x is blue, y is yellow, z is green
axis = gl.GLAxisItem()
axis_size = 2
axis.setSize(x=axis_size, y=axis_size, z=axis_size)
point_cloud_widget.addItem(axis)
# labels
x_label = gl.GLTextItem(pos=np.array([axis_size, 0, 0]), text='x', color='blue')
y_label = gl.GLTextItem(pos=np.array([0, axis_size, 0]), text='y', color='yellow')
z_label = gl.GLTextItem(pos=np.array([0, 0, axis_size]), text='z', color='green')
point_cloud_widget.addItem(x_label)
point_cloud_widget.addItem(y_label)
point_cloud_widget.addItem(z_label)


point_cloud_path = (
    './example_data/point_cloud_classification/modelnet_normal_resampled/bathtub/bathtub_0107.txt'
)
point_cloud = np.loadtxt(point_cloud_path, delimiter=',').astype(np.float32)
# random sample 500 points
# random_indices = np.random.randint(0, len(point_cloud), 1000)
point_cloud_pos = point_cloud[:, :3]
# point size
sizes = np.array([0.02] * len(point_cloud_pos))
# assign color
colors = np.atleast_2d([0.5, 0.5, 0.5, 0.5]).repeat(repeats=len(point_cloud_pos), axis=0)
# set data
point_cloud_vis_item = gl.GLScatterPlotItem()
point_cloud_vis_item.setData(pos=point_cloud_pos, color=colors, size=sizes, pxMode=False)
# point_cloud_vis_item.translate(5, 5, 0)
point_cloud_widget.addItem(point_cloud_vis_item)


# run the app
app.exec()

# exit the app
sys.exit()
