import sys
import pyqtgraph as pg
import numpy as np
from PySide6 import QtWidgets

class DrawingImage(pg.ImageItem):
    def mouseClickEvent(self, event):
        print("Click", event.pos())

    def mouseDragEvent(self, event):
        if event.isStart():
            print("Start drag", event.pos())
        elif event.isFinish():
            print("Stop drag", event.pos())
        else:
            print("Drag", event.pos())

    def hoverEvent(self, event):
        if not event.isExit():
            # the mouse is hovering over the image; make sure no other items
            # will receive left click/drag events from here.
            event.acceptDrags(pg.QtCore.Qt.LeftButton)
            event.acceptClicks(pg.QtCore.Qt.LeftButton)

#GUI control object
app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget()
img = DrawingImage(np.random.normal(size=(100, 150)), axisOrder='row-major')
# view = win.addPlot()
# view.addItem(img)
view_box = pg.ViewBox()
view_box.addItem(img)
#Plot object creation&View created above_set box
plot = pg.PlotItem(viewBox=view_box)
#Add plot to window
win.addItem(plot)
win.show()
sys.exit(app.exec_())
