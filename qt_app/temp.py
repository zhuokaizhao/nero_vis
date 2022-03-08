# import numpy
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, QtCore

# def gaussian(A, B, x):
#   return A * numpy.exp(-(x/(2. * B))**2.)

# def mouseMoved(evt):
#   mousePoint = p.vb.mapSceneToView(evt[0])
#   label.setText("<span style='font-size: 14pt; color: white'> x = %0.2f, <span style='color: white'> y = %0.2f</span>" % (mousePoint.x(), mousePoint.y()))


# # Initial data frame
# x = numpy.linspace(-5., 5., 10000)
# y = gaussian(5., 0.2, x)


# # Generate layout
# win = pg.GraphicsWindow()
# label = pg.LabelItem(justify = "right")
# win.addItem(label)

# p = win.addPlot()

# plot = p.plot(x, y, pen = "y")

# proxy = pg.SignalProxy(p.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

# # Update layout with new data
# i = 0
# while True:
#   noise = numpy.random.normal(0, .2, len(y))
#   y_new = y + noise

#   plot.setData(x, y_new, pen = "y", clear = True)
#   p.enableAutoRange("xy", False)

#   pg.QtGui.QApplication.processEvents()

#   i += 1

# win.close()


# import torch
# import models

# model = models.Pre_Trained_FastRCNN()
# torch.save(model.state_dict(), '/home/zhuokai/Desktop/UChicago/Research/nero_vis/qt_app/example_models/object_detection/pre_trained/pretrained_fasterrcnn.pth')

import sys

import numpy as np
import pyqtgraph as pg
from colour import Color
from PySide6 import QtWidgets
from pgcolorbar.colorlegend import ColorLegendItem

#Sample array
data = np.random.normal(size=(200, 200))
data[40:80, 40:120] += 4
data = pg.gaussianFilter(data, (15, 15))
data += np.random.normal(size=(200, 200)) * 0.1

app = QtWidgets.QApplication(sys.argv)

window = pg.GraphicsLayoutWidget()

blue, red = Color('blue'), Color('red')
colors = blue.range_to(red, 256)
colors_array = np.array([np.array(color.get_rgb()) * 255 for color in colors])
look_up_table = colors_array.astype(np.uint8)

image = pg.ImageItem()
image.setOpts(axisOrder='row-major')
image.setLookupTable(look_up_table)
image.setImage(data)

view_box = pg.ViewBox()
view_box.setAspectLocked(lock=True)
view_box.addItem(image)

plot = pg.PlotItem(viewBox=view_box)

# color_bar = ColorLegendItem(imageItem=image, showHistogram=True, label='sample')  # 2021/01/20 add label
# color_bar.resetColorLevels()

cm = pg.colormap.get('CET-L9')
bar = pg.ColorBarItem( values= (0, 100), cmap=cm )
bar.setImageItem( image, insert_in=plot )

window.addItem(plot)
# window.addColorBar( image, colorMap='viridis', values=(0, 1) )
# window.addItem(color_bar)

window.show()

sys.exit(app.exec_())