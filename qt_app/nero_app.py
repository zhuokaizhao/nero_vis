import sys
import requests
# import random
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui  import QPixmap, QFont
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QFileDialog, QWidget, QLabel, QRadioButton


class UI_MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # general layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignCenter)
        # left, top, right, and bottom margins
        self.layout.setContentsMargins(100, 100, 100, 100)

        # individual laytout for different widgets
        # title
        self.title_layout = QtWidgets.QVBoxLayout()
        self.title_layout.setContentsMargins(0, 0, 0, 20)
        # mode selection radio buttons
        self.mode_layout = QtWidgets.QGridLayout()
        self.mode_layout.setContentsMargins(0, 0, 0, 50)
        # image and model loading push buttons
        self.load_button_layout = QtWidgets.QGridLayout()
        # future loaded images and model layout
        self.loaded_layout = QtWidgets.QGridLayout()
        self.loaded_layout.setContentsMargins(30, 50, 30, 50)

        # title of the application
        self.title = QLabel('Non-Equivariance Revealed on Orbits',
                            alignment=QtCore.Qt.AlignCenter)
        self.title.setFont(QFont('Helvetica', 20))
        self.title_layout.addWidget(self.title)
        self.title_layout.setContentsMargins(0, 0, 0, 50)

        # radio buttons on mode selection (classification, object detection, PIV)
        # radio_buttons_layout = QtWidgets.QGridLayout(self)
        self.radio_button_1 = QRadioButton('Classification')
        self.radio_button_1.setChecked(True)
        self.radio_button_1.pressed.connect(self.radio_button_1_clicked)
        self.mode_layout.addWidget(self.radio_button_1, 0, 0)

        self.radio_button_2 = QRadioButton('Object Detection')
        self.radio_button_2.setChecked(False)
        self.radio_button_2.pressed.connect(self.radio_button_2_clicked)
        self.mode_layout.addWidget(self.radio_button_2, 0, 1)

        self.radio_button_3 = QRadioButton('Particle Image Velocimetry (PIV)')
        self.radio_button_3.setChecked(False)
        self.radio_button_3.pressed.connect(self.radio_button_3_clicked)
        self.mode_layout.addWidget(self.radio_button_3, 0, 2)

        # load data button
        self.data_button = QtWidgets.QPushButton('Load Test Image')
        # self.data_button.setFixedSize(QtCore.QSize(300, 100))
        self.load_button_layout.addWidget(self.data_button)
        self.data_button.clicked.connect(self.load_image_clicked)
        # load model button
        self.model_button = QtWidgets.QPushButton('Load Model')
        self.load_button_layout.addWidget(self.model_button)
        self.model_button.clicked.connect(self.load_model_clicked)

        # add individual layouts to the display general layout
        self.layout.addLayout(self.title_layout)
        self.layout.addLayout(self.mode_layout)
        self.layout.addLayout(self.load_button_layout)
        self.layout.addLayout(self.loaded_layout)

        # flag to check if an image has been displayed
        self.image_existed = False
        self.model_existed = False
        # default app mode is classification
        self.mode = 'classification'

    @QtCore.Slot()
    def radio_button_1_clicked(self):
        print('Classification button clicked')
        self.mode = 'classification'

    @QtCore.Slot()
    def radio_button_2_clicked(self):
        print('Object detection button clicked')
        self.mode = 'object_detection'

    @QtCore.Slot()
    def radio_button_3_clicked(self):
        print('PIV button clicked')
        self.mode = 'PIV'

    @QtCore.Slot()
    def load_image_clicked(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, QObject.tr('Load Test Image'))
        # in case user did not load any image
        if self.image_path == '':
            return
        print(f'Loaded image {self.image_path}')
        # display the image
        self.display_image(self.image_existed)
        # change the button text
        image_name = self.image_path.split('/')[-1]
        self.data_button.setText(f'Loaded image {image_name}. Click to load new image')

    @QtCore.Slot()
    def load_model_clicked(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, QObject.tr('Load Model'))
        # in case user did not load any image
        if self.model_path == '':
            return
        print(f'Loaded model {self.model_path}')
        # display the model
        self.display_model(self.model_existed)
        # change the button text
        model_name = self.model_path.split('/')[-1]
        self.model_button.setText(f'Loaded model {model_name}. Click to load new model')

    def display_image(self, image_existed):

        loaded_image = QtGui.QImage(self.image_path)

        # add a new label for loaded image if no image has existed
        if not image_existed:
            self.image_label = QLabel(self)

        # resize the image to a fixed size
        image_pixmap = QPixmap(loaded_image.scaledToWidth(100))
        self.image_label.setPixmap(image_pixmap)

        # add this image to the layout
        self.loaded_layout.addWidget(self.image_label, 0, 0)
        self.image_existed = True

    # draw model diagram, return model pixmap
    def draw_model_diagram(self, width, height, boundary_width):
        model_pixmap = QPixmap(width+2*boundary_width, height+2*boundary_width)
        model_pixmap.fill(QtCore.Qt.white)

        # draw model diagram (simple rectangle for now)
        painter = QtGui.QPainter(model_pixmap)
        # set pen (used to draw outlines of shapes) and brush (draw the background of a shape)
        pen = QtGui.QPen()
        pen.setWidth(boundary_width)
        pen.setColor(QtGui.QColor('#EB5160'))
        painter.setPen(pen)
        # painter.setBrush(QtGui.QColor('orange'))

        # design rectangle
        rectangle = QtCore.QRect(boundary_width, boundary_width, width, height)
        painter.drawRect(rectangle)
        painter.end()

        return model_pixmap

    def display_model(self, model_existed):
        # add a new label for loaded image if no image has existed
        if not model_existed:
            self.model_label = QLabel(self)

        model_pixmap = self.draw_model_diagram(100, 100, 3)
        self.model_label.setPixmap(model_pixmap)

        # add this rectangle to the layout
        self.loaded_layout.addWidget(self.model_label, 0, 2)
        self.model_existed = True



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
