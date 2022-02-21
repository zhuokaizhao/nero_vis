import sys
import torch
import torchvision
import numpy as np
from PIL import Image
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui  import QPixmap, QFont
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QFileDialog, QWidget, QLabel, QRadioButton

import nero_transform
import nero_utilities
import nero_run_model


class UI_MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # white background color
        self.setStyleSheet("background-color: white;")
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
        # buttons layout for run model
        self.run_button_layout = QtWidgets.QGridLayout()
        # warning text layout
        self.warning_layout = QtWidgets.QVBoxLayout()

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
        self.layout.addLayout(self.run_button_layout)
        self.layout.addLayout(self.warning_layout)

        # flag to check if an image has been displayed
        self.image_existed = False
        self.model_existed = False
        self.run_buttons_existed = False
        self.result_existed = False
        self.warning_existed = False
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

        # load the image and scale the size
        # self.loaded_image = QtGui.QImage(self.image_path)
        self.loaded_image_pt = torch.from_numpy(np.asarray(Image.open(self.image_path)))[:, :, None]
        self.cur_image_pt = self.loaded_image_pt.clone()
        # QImage for display purpose
        self.cur_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt)
        # resize the display QImage
        self.image_size = 150
        self.cur_image = self.cur_image.scaledToWidth(self.image_size)

        # display the image
        self.display_image()
        # change the button text
        image_name = self.image_path.split('/')[-1]
        self.data_button.setText(f'Loaded image {image_name}. Click to load new image')

        # show the run button when both ready
        if self.model_existed and self.image_existed and not self.run_buttons_existed:
            # run once button
            self.run_once_button = QtWidgets.QPushButton('Run model once')
            self.run_button_layout.addWidget(self.run_once_button)
            self.run_once_button.clicked.connect(self.run_once_button_clicked)
            # load model button
            self.run_all_button = QtWidgets.QPushButton('Run model on all transformations')
            self.run_button_layout.addWidget(self.run_all_button)
            self.run_all_button.clicked.connect(self.run_all_button_clicked)

            self.run_buttons_existed = True

    @QtCore.Slot()
    def load_model_clicked(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, QObject.tr('Load Model'))
        # in case user did not load any image
        if self.model_path == '':
            return
        print(f'Loaded model {self.model_path}')

        model_name = self.model_path.split('/')[-1]
        width = 300
        height = 200
        # display the model
        self.display_model(model_name, width, height, boundary_width=3)
        # change the button text
        self.model_button.setText(f'Loaded model {model_name}. Click to load new model')

        # show the run button when both ready
        if self.model_existed and self.image_existed and not self.run_buttons_existed:
            # run once button
            self.run_once_button = QtWidgets.QPushButton('Run model once')
            self.run_button_layout.addWidget(self.run_once_button)
            self.run_once_button.clicked.connect(self.run_once_button_clicked)
            # load model button
            self.run_all_button = QtWidgets.QPushButton('Run model on all transformations')
            self.run_button_layout.addWidget(self.run_all_button)
            self.run_all_button.clicked.connect(self.run_all_button_clicked)

            self.run_buttons_existed = True

    @QtCore.Slot()
    def run_once_button_clicked(self):
        # check if both input image and model are ready
        if self.image_existed and self.model_existed:
            output, pred = nero_run_model.run_mnist_once('non-eqv', self.model_path, self.cur_image_pt)
            # display the result
            # add a new label for loaded image if no image has existed
            if not self.result_existed:
                self.mnist_label = QLabel(self)
                self.mnist_label.setAlignment(QtCore.Qt.AlignCenter)
                self.mnist_label.setWordWrap(True)
                self.mnist_label.setTextFormat(QtGui.Qt.AutoText)
                self.result_existed = True

            self.display_mnist_result(output, pred, width=300, height=200, boundary_width=3)

        else:
            # display a warning text
            # prepare a pixmap for the image
            warning_pixmap = QPixmap(200, 100)
            warning_pixmap.fill(QtCore.Qt.white)

            if not self.warning_existed:
                self.warning_label = QLabel(self)
                self.warning_label.setAlignment(QtCore.Qt.AlignCenter)
                self.warning_existed = True

            # display warning text
            painter = QtGui.QPainter(warning_pixmap)
            if not self.image_existed:
                warn_text = 'Please load data image(s) first'
            elif not self.model_existed:
                warn_text = 'Please load model first'
            elif not self.image_existed and not self.model_existed:
                warn_text = 'Please load data image(s) and model first'

            painter.drawText(0, 0, 200, 100, QtGui.Qt.AlignHCenter, warn_text)
            painter.end()


    @QtCore.Slot()
    def run_all_button_clicked(self):
        if self.mode == 'classification':
            # run all rotation test with 5 degree increment
            nero_run_model.run_mnist_all()


    def display_image(self):

        # prepare a pixmap for the image
        image_pixmap = QPixmap(self.cur_image)

        # add a new label for loaded image if no image has existed
        if not self.image_existed:
            self.image_label = QLabel(self)
            self.image_label.setAlignment(QtCore.Qt.AlignLeft)
            self.image_existed = True

        # put pixmap in the label
        self.image_label.setPixmap(image_pixmap)

        # add this image to the layout
        self.loaded_layout.addWidget(self.image_label, 0, 0)

    # draw arrow
    def draw_arrow(self, painter, pen, width, height, boundary_width):
        # draw arrow to indicate feeding
        pen.setWidth(boundary_width)
        pen.setColor(QtGui.QColor('black'))
        painter.setPen(pen)
        # horizontal line
        painter.drawLine(0, height//2, width, height//2)
        # upper arrow
        painter.drawLine(int(0.6*width), int(0.4*height), width, height//2)
        # bottom arrow
        painter.drawLine(int(0.6*width), int(0.6*height), width, height//2)

    # draw model diagram, return model pixmap
    def draw_model_diagram(self, painter, pen, name, width, height, boundary_width):

        # draw rectangle to represent model
        pen.setWidth(boundary_width)
        pen.setColor(QtGui.QColor('red'))
        painter.setPen(pen)
        rectangle = QtCore.QRect(int(width//3)+boundary_width, boundary_width, width//3*2-2*boundary_width, height-2*boundary_width)
        painter.drawRect(rectangle)

        # draw model name
        if len(name) > 20:
            name = name[:20] + '\n' + name[20:]
            painter.drawText(int(width//3)+boundary_width, height//2-6*boundary_width, width//3*2, height, QtGui.Qt.AlignHCenter, name)
        else:
            painter.drawText(int(width//3)+boundary_width, height//2-2*boundary_width, width//3*2, height, QtGui.Qt.AlignHCenter, name)

    def display_model(self, model_name, width, height, boundary_width):
        # add a new label for loaded image if no image has existed
        if not self.model_existed:
            self.model_label = QLabel(self)
            self.model_label.setWordWrap(True)
            self.model_label.setTextFormat(QtGui.Qt.AutoText)
            self.image_label.setAlignment(QtCore.Qt.AlignLeft)
            self.model_existed = True

        # total model pixmap size
        model_pixmap = QPixmap(width, height)
        model_pixmap.fill(QtCore.Qt.white)

        # define painter that is working on the pixmap
        painter = QtGui.QPainter(model_pixmap)
        # set pen (used to draw outlines of shapes) and brush (draw the background of a shape)
        pen = QtGui.QPen()

        # draw standard arrow
        self.draw_arrow(painter, pen, 80, 150, boundary_width)
        # draw the model diagram
        self.draw_model_diagram(painter, pen, model_name, width, 150, boundary_width)
        painter.end()

        # add to the label and layout
        self.model_label.setPixmap(model_pixmap)
        self.loaded_layout.addWidget(self.model_label, 0, 2)


    def display_mnist_result(self, output, pred, width, height, boundary_width):
        # use the loaded_layout
        mnist_pixmap = QPixmap(width, height)
        mnist_pixmap.fill(QtCore.Qt.white)
        # draw arrow
        painter = QtGui.QPainter(mnist_pixmap)
        # set pen (used to draw outlines of shapes) and brush (draw the background of a shape)
        pen = QtGui.QPen()
        # draw arrow to indicate feeding
        self.draw_arrow(painter, pen, 80, 150, boundary_width)
        # draw result
        result_text = ''
        for i in range(len(output)):
            prob = '{:,.2}'.format(output[i]) if output[i] >= 0.01 else '{:,.2e}'.format(output[i])
            result_text += f'Class {i}: {prob}\n'

        painter.drawText(int(width//3)+boundary_width, 0, width//3*2, height, QtGui.Qt.AlignLeft, result_text)
        painter.end()

        self.mnist_label.setPixmap(mnist_pixmap)
        # add to the layout
        self.loaded_layout.addWidget(self.mnist_label, 0, 4)

        return mnist_pixmap


    def mouseMoveEvent(self, event):
        # print("mouseMoveEvent")
        # when in translation mode
        if self.translation:
            print('translating')
        # when in rotation mode
        elif self.rotation:
            self.all_mouse_x.append(event.position().x())
            self.all_mouse_y.append(event.position().y())
            if len(self.all_mouse_x) > 2:
                self.all_mouse_x.pop(0)
                self.all_mouse_y.pop(0)
            # angle = math.atan2(event.y() - self.all_mouse_y[-2], event.x() - self.all_mouse_x[-2]) / math.pi * 180
            # naive way to determine rotate angle
            if self.all_mouse_x[-1] < self.all_mouse_x[0] or self.all_mouse_y[-1] < self.all_mouse_y[0]:
                self.total_rotate_angle += 1
            else:
                self.total_rotate_angle -= 1

            # rotate the image tensor
            self.cur_image_pt = nero_transform.rotate_mnist_image(self.loaded_image_pt, self.total_rotate_angle)
            # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(angle), QtCore.Qt.SmoothTransformation)
            # convert image tensor to qt image
            self.cur_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt)
            # resize the at image
            self.cur_image = self.cur_image.scaledToWidth(self.image_size)
            # update the pixmap and label
            self.image_pixmap = QPixmap(self.cur_image)
            self.image_label.setPixmap(self.image_pixmap)

    def mousePressEvent(self, event):
        print("mousePressEvent")
        self.all_mouse_x = [event.position().x()]
        self.all_mouse_y = [event.position().y()]

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
