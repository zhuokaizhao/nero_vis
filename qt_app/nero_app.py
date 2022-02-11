import sys
import requests
# import random
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui  import QPixmap
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QFileDialog, QWidget, QLabel


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
        self.image_path, _ = QFileDialog.getOpenFileName(self, QObject.tr('Load Image'))
        self.display_image()

    @QtCore.Slot()
    def select_model(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, QObject.tr('Load Model'))

    def display_image(self):
        # change current window's title to reflect change
        self.setWindowTitle(f'Loaded Image {self.image_path}')

        # add a new label for loaded image
        image_label  = QLabel(self)
        loaded_image = QtGui.QImage(self.image_path)

        # resize the image to a fixed size
        image_pixmap = QPixmap(loaded_image.scaledToWidth(100))
        image_label.setPixmap(image_pixmap)

        # add this image to the display window
        self.layout.addWidget(image_label)





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
