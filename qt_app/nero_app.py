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

        self.layout = QtWidgets.QVBoxLayout(self)
        self.title_layout = QtWidgets.QVBoxLayout()
        self.mode_layout = QtWidgets.QGridLayout()
        self.load_button_layout = QtWidgets.QGridLayout()

        # title of the application
        self.title = QLabel('Non-Equivariance Revealed on Orbits',
                            alignment=QtCore.Qt.AlignCenter)
        self.title.setFont(QFont('Helvetica', 20))
        self.title_layout.addWidget(self.title)
        self.title_layout.addStretch(1)

        # default app mode is classification
        self.mode = 'classification'

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
        self.data_button.clicked.connect(self.select_images)
        # load model button
        self.model_button = QtWidgets.QPushButton('Load Model')
        self.load_button_layout.addWidget(self.model_button)
        self.model_button.clicked.connect(self.select_model)

        self.layout.addLayout(self.title_layout)
        self.layout.addLayout(self.mode_layout)
        self.layout.addLayout(self.load_button_layout)

        # flag to check if an image has been displayed
        self.image_existed = False

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
    def select_images(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, QObject.tr('Load Test Image'))
        print(f'Loaded image {self.image_path}')
        # display the image
        self.display_image(self.image_existed)
        # change the button text
        image_name = self.image_path.split('/')[-1]
        self.data_button.setText(f'Loaded Image {image_name}. Click to reload New Image')

    @QtCore.Slot()
    def select_model(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, QObject.tr('Load Model'))

    def display_image(self, image_existed):

        loaded_image = QtGui.QImage(self.image_path)

        # add a new label for loaded image if no image has existed
        if not image_existed:
            self.image_label  = QLabel(self)

        # resize the image to a fixed size
        image_pixmap = QPixmap(loaded_image.scaledToWidth(100))
        self.image_label.setPixmap(image_pixmap)

        # add this image to the display window
        if not image_existed:
            self.layout.addWidget(self.image_label)

        self.image_existed = True





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
