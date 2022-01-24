import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import QObject, QRectF, Qt
from PySide6.QtWidgets import QFileDialog, QWidget, QVBoxLayout, QGraphicsScene, QGraphicsView
from PySide6.QtGui import QPixmap

class ImageViewer(QWidget):
    def __init__(self, image_path):
        super().__init__()

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

        self.load_image(image_path)

    def load_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.scene.addPixmap(pixmap)
        self.view.fitInView(QRectF(0, 0, pixmap.width(), pixmap.height()), Qt.KeepAspectRatio)
        self.scene.update()


class UI_MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QtWidgets.QVBoxLayout(self)

        # welcome text
        self.text = QtWidgets.QLabel('Non-Equivariance Revealed on Orbits',
                                     alignment=QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.text)

        # select dataset button
        self.data_button = QtWidgets.QPushButton('Select data path(s)')
        self.layout.addWidget(self.data_button)
        self.data_button.clicked.connect(self.select_images)
        # model path
        self.model_button = QtWidgets.QPushButton('Select model path')
        self.layout.addWidget(self.model_button)
        self.model_button.clicked.connect(self.select_model)

    @QtCore.Slot()
    def select_images(self):
        file_path, _ = QFileDialog.getOpenFileName(self, QObject.tr('Load Image'))
        self.image_viewer = ImageViewer(file_path)
        self.image_viewer.view()

    @QtCore.Slot()
    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, QObject.tr('Load Model'))



if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = UI_MainWindow()
    widget.resize(1024, 768)
    widget.show()

    sys.exit(app.exec())



#










# if __name__ == "__main__":
#     app = QtWidgets.QApplication([])
#     main_window = MainWindow()
#     main_window.resize(1024, 768)
#     main_window.show()
#     sys.exit(app.exec())
