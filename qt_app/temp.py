import pyqtgraph as pg
import numpy as np
from pyqtgraph import QtCore, QtGui


def plot_polar(plot):
    # plot = pg.plot()
    plot.setAspectLocked()

    # Add polar grid lines
    plot.addLine(x=0, pen=0.2)
    plot.addLine(y=0, pen=0.2)
    for r in range(2, 20, 2):
        circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
        circle.setPen(pg.mkPen(0.2))
        plot.addItem(circle)

    # make polar data
    # theta = np.linspace(0, 2 * np.pi, 10)
    # radius = np.random.normal(loc=10, size=10)

    # # Transform to cartesian and plot
    # x = radius * np.cos(theta)
    # y = radius * np.sin(theta)
    # plot.plot(x, y)

    return plot


# from pyqtgraph.Qt import QtGui, QtCore

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
mw.resize(800, 800)
view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
mw.setCentralWidget(view)
mw.show()
mw.setWindowTitle('pyqtgraph example: ScatterPlot')


## create four areas to add plots
w3 = view.addPlot()
w3 = plot_polar(w3)

## Make all plots clickable
lastClicked = []
def clicked(plot, points):
    global lastClicked
    for p in lastClicked:
        p.resetPen()
    print("clicked points", points)
    for p in points:
        p.setPen('b', width=2)
    lastClicked = points


s3 = pg.ScatterPlotItem(pxMode=False)   ## Set pxMode=False to allow spots to transform with the view
spots3 = []
for i in range(10):
    for j in range(10):
        # Transform to cartesian and plot
        x = i * np.cos(j)
        y = i * np.sin(j)
        spots3.append({'pos': (1*x, 1*y), 'size': 1, 'pen': {'color': 'w', 'width': 2}, 'brush':pg.intColor(i*10+j, 100)})
s3.addPoints(spots3)
w3.addItem(s3)
s3.sigClicked.connect(clicked)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()