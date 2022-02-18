import sys
import copy
import math
import torch
import requests
import torchvision
import numpy as np
from PIL import Image
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui  import QPixmap, QFont
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QFileDialog, QWidget, QLabel, QRadioButton


# some helper functions
def qt_image_to_tensor(img, share_memory=False):
    """
    Creates a PyTorch Tensor from a QImage via converting to Numpy array first.

    If share_memory is True, the numpy array and the QImage is shared.
    Be careful: make sure the numpy array is destroyed before the image,
                otherwise the array will point to unreserved memory!!
    """
    assert isinstance(img, QtGui.QImage), "img must be a QtGui.QImage object"
    assert img.format() == QtGui.QImage.Format.Format_RGB32, "img format must be QImage.Format.Format_RGB32, got: {}".format(img.format())

    img_size = img.size()
    buffer = img.constBits()

    # Sanity check
    n_bits_buffer = len(buffer) * 8
    n_bits_image  = img_size.width() * img_size.height() * img.depth()
    assert n_bits_buffer == n_bits_image, \
        "size mismatch: {} != {}".format(n_bits_buffer, n_bits_image)

    assert img.depth() == 32, "unexpected image depth: {}".format(img.depth())

    # Note the different width height parameter order!
    arr = np.ndarray(shape  = (img_size.height(), img_size.width(), img.depth()//8),
                     buffer = buffer,
                     dtype  = np.uint8)

    if share_memory:
        return torch.from_numpy(arr)
    else:
        return torch.from_numpy(copy.deepcopy(arr))


def tensor_to_qt_image(img_pt):
    img_np = img_pt.numpy()
    img_qt = QtGui.QImage(img_np.data, img_np.shape[1], img_np.shape[0], QtGui.QImage.Format_BGR888)

    return img_qt


# rotate mnist image
def rotate_mnist_image(image, angle, target_img_size):
    if target_img_size != 29:
        raise Exception('Warning: MNIST model takes image size 29')
    img_size = len(image)
    # transform includes upsample, rotate, downsample and padding (right and bottom) to image_size
    image = torchvision.transforms.Resize(img_size*3)(image)
    image = torchvision.transforms.RandomRotation(degrees=(angle, angle), resample=Image.BILINEAR, expand=False)(image)
    image = torchvision.transforms.Resize(img_size)(image)
    image = torchvision.transforms.Pad(((target_img_size-img_size)//2, (target_img_size-img_size)//2, (target_img_size-img_size)//2+1, (self.target_img_size-self.img_size)//2+1), fill=0, padding_mode='constant')(image)

    return image

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

        # image (input data) modification mode
        self.translation = False
        self.rotation = False

        # total rotate angle
        self.total_rotate_angle = 0

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
        self.image_size = 100
        self.display_image(self.image_existed, self.image_size)
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

        model_name = self.model_path.split('/')[-1]
        # display the model
        self.display_model(self.model_existed, model_name)
        # change the button text
        self.model_button.setText(f'Loaded model {model_name}. Click to load new model')

    def display_image(self, image_existed, image_size):

        # load the image and scale the size
        self.loaded_image = QtGui.QImage(self.image_path)
        # save a numpy copy of the loaded image
        self.loaded_image_pt = qt_image_to_tensor(self.loaded_image, share_memory=False)
        # resize the display QImage
        self.loaded_image = self.loaded_image.scaledToWidth(image_size)

        # prepare a pixmap for the image
        self.image_pixmap = QPixmap(self.loaded_image)

        # add a new label for loaded image if no image has existed
        if not image_existed:
            # set minimum size to prepare for later rotation
            diag = (self.image_pixmap.width()**2 + self.image_pixmap.height()**2)**0.5
            self.image_label = QLabel(self)
            self.image_label.setMinimumSize(diag, diag)
            self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        # put pixmap in the label
        self.image_label.setPixmap(self.image_pixmap)

        # add this image to the layout
        self.loaded_layout.addWidget(self.image_label, 0, 0)
        self.image_existed = True

    # draw model diagram, return model pixmap
    def draw_model_diagram(self, name, width, height, boundary_width):
        model_pixmap = QPixmap(width, height)
        model_pixmap.fill(QtCore.Qt.white)

        # draw model diagram (simple rectangle for now)
        painter = QtGui.QPainter(model_pixmap)
        # set pen (used to draw outlines of shapes) and brush (draw the background of a shape)
        pen = QtGui.QPen()
        pen.setWidth(boundary_width)
        pen.setColor(QtGui.QColor('#EB5160'))
        painter.setPen(pen)
        # painter.setBrush(QtGui.QColor('orange'))

        # draw rectangle
        rectangle = QtCore.QRect(boundary_width, boundary_width, width-2*boundary_width, height-2*boundary_width)
        painter.drawRect(rectangle)

        # draw model name
        painter.drawText(0, height//2-2*boundary_width, width, height, QtGui.Qt.AlignHCenter, name)
        painter.end()

        return model_pixmap

    def display_model(self, model_existed, model_name):
        # add a new label for loaded image if no image has existed
        if not model_existed:
            self.model_label = QLabel(self)

        model_pixmap = self.draw_model_diagram(model_name, 200, 150, 3)
        self.model_label.setPixmap(model_pixmap)

        # add this rectangle to the layout
        self.loaded_layout.addWidget(self.model_label, 0, 2)
        self.model_existed = True

    def mouseMoveEvent(self, event):
        # print("mouseMoveEvent")
        # when in translation mode
        if self.translation:
            print('translating')
        # when in rotation mode
        elif self.rotation:
            self.all_mouse_x.append(event.x())
            self.all_mouse_y.append(event.y())
            if len(self.all_mouse_x) > 2:
                self.all_mouse_x.pop(0)
                self.all_mouse_y.pop(0)
            # angle = math.atan2(event.y() - self.all_mouse_y[-2], event.x() - self.all_mouse_x[-2]) / math.pi * 180
            # naive way to determine rotate angle
            if self.all_mouse_x[-1] > self.all_mouse_x[0] or self.all_mouse_y[-1] > self.all_mouse_y[0]:
                self.total_rotate_angle += 1
            else:
                self.total_rotate_angle -= 1

            # rotate the image
            cur_image_pt = rotate_mnist_image(self.loaded_image_pt, self.total_rotate_angle, 29)
            # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(angle), QtCore.Qt.SmoothTransformation)
            # convert image tensor to qt
            cur_image = tensor_to_qt_image(cur_image_pt)
            # update the pixmap and label
            self.image_pixmap = QPixmap(cur_image)
            self.image_label.setPixmap(self.image_pixmap)

    def mousePressEvent(self, event):
        print("mousePressEvent")
        self.all_mouse_x = [event.x()]
        self.all_mouse_y = [event.y()]

    def mouseReleaseEvent(self, event):
        print("mouseReleaseEvent")

    # called when a key is pressed
    def keyPressEvent(self, event):
        key_pressed = event.text()

        # different key pressed
        if 'h' == key_pressed or '?' == key_pressed:
            self.print_help()
        if 'r' == key_pressed:
            print('Rotation mode ON')
            self.rotation = True
            self.translation = False
        if 't' == key_pressed:
            print('Translation mode ON')
            self.translation = True
            self.rotation = False
        if 'q' == key_pressed:
            app.quit()


    # print help message
    def print_help(self):
        print('Ah Oh, help not available')

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
