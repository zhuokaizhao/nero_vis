import os
import sys
import glob
import torch
import torchvision
import numpy as np
from PIL import Image
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui  import QPixmap, QFont
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QFileDialog, QWidget, QLabel, QRadioButton
from PySide6.QtCharts import QBarSet, QBarSeries, QChart, QBarCategoryAxis, QValueAxis, QChartView

import nero_transform
import nero_utilities
import nero_run_model


class UI_MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # set window title
        self.setWindowTitle('Non-Equivariance Revealed on Orbits')
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
        self.mode_layout.setContentsMargins(50, 0, 0, 50)
        # image and model loading buttons and drop down menus
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

        # radio buttons on mode selection (digit_recognition, object detection, PIV)
        # pixmap on text telling what this is
        # use the loaded_layout
        mode_pixmap = QPixmap(100, 30)
        mode_pixmap.fill(QtCore.Qt.white)
        # draw text
        painter = QtGui.QPainter(mode_pixmap)
        # set pen (used to draw outlines of shapes) and brush (draw the background of a shape)
        pen = QtGui.QPen()
        painter.setFont(QFont('Helvetica', 14))
        painter.drawText(0, 0, 100, 30, QtGui.Qt.AlignLeft, 'Model type: ')
        painter.end()
        # create label to contain the texts
        self.mode_label = QLabel(self)
        self.mode_label.setAlignment(QtCore.Qt.AlignLeft)
        self.mode_label.setWordWrap(True)
        self.mode_label.setTextFormat(QtGui.Qt.AutoText)
        self.mode_label.setPixmap(mode_pixmap)
        # add to the layout
        self.mode_layout.addWidget(self.mode_label, 0, 0)
        # radio_buttons_layout = QtWidgets.QGridLayout(self)
        self.radio_button_1 = QRadioButton('Digit recognition')
        self.radio_button_1.setChecked(True)
        self.radio_button_1.setStyleSheet('QRadioButton{font: 14pt Helvetica;} QRadioButton::indicator { width: 15px; height: 15px;};')
        self.radio_button_1.pressed.connect(self.digit_reconition_button_clicked)
        self.mode_layout.addWidget(self.radio_button_1, 0, 2)

        self.radio_button_2 = QRadioButton('Object detection')
        self.radio_button_2.setChecked(False)
        self.radio_button_2.setStyleSheet('QRadioButton{font: 14pt Helvetica;} QRadioButton::indicator { width: 15px; height: 15px;};')
        self.radio_button_2.pressed.connect(self.object_detection_button_clicked)
        self.mode_layout.addWidget(self.radio_button_2, 0, 3)

        self.radio_button_3 = QRadioButton('Particle Image Velocimetry (PIV)')
        self.radio_button_3.setChecked(False)
        self.radio_button_3.setStyleSheet('QRadioButton{font: 14pt Helvetica;} QRadioButton::indicator { width: 15px; height: 15px;};')
        self.radio_button_3.pressed.connect(self.piv_button_clicked)
        self.mode_layout.addWidget(self.radio_button_3, 0, 4)

        # default app mode is digit_recognition
        self.mode = 'digit_recognition'
        self.display_image_size = 150

        # load data button
        self.data_button = QtWidgets.QPushButton('Load Test Image')
        self.load_button_layout.addWidget(self.data_button, 0, 0)
        self.data_button.clicked.connect(self.load_image_clicked)
        # load models choices
        model_1_menu = QtWidgets.QComboBox()
        model_2_menu = QtWidgets.QComboBox()
        if self.mode == 'digit_recognition':
            model_1_menu.addItem('Simple model')
            model_1_menu.addItem('E2CNN model')
            model_1_menu.addItem('Data augmentation model')
            model_1_menu.setCurrentText('Simple model')
            model_2_menu.addItem('Simple model')
            model_2_menu.addItem('E2CNN model')
            model_2_menu.addItem('Data augmentation model')
            model_2_menu.setCurrentText('E2CNN model')

        # connect the drop down menu with actions
        model_1_menu.currentTextChanged.connect(self.model_1_text_changed)
        model_2_menu.currentTextChanged.connect(self.model_2_text_changed)
        self.load_button_layout.addWidget(model_1_menu, 0, 1)
        self.load_button_layout.addWidget(model_2_menu, 0, 2)

        # add individual layouts to the display general layout
        self.layout.addLayout(self.title_layout)
        self.layout.addLayout(self.mode_layout)
        self.layout.addLayout(self.load_button_layout)
        self.layout.addLayout(self.loaded_layout)
        self.layout.addLayout(self.run_button_layout)
        self.layout.addLayout(self.warning_layout)

        # set the model selection to be the simple model
        self.model_1_name = 'Simple model'
        self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
        self.model_2_name = 'E2CNN model'
        self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
        # preload model
        self.model_1 = nero_run_model.load_model('non-eqv', self.model_1_path)
        self.model_2 = nero_run_model.load_model('rot-eqv', self.model_2_path)

        # flag to check if an image has been displayed
        self.image_existed = False
        self.result_existed = False
        self.warning_existed = False

        # image (input data) modification mode
        self.translation = False
        self.rotation = False

        # total rotate angle
        self.cur_rotation_angle = 0
        self.prev_rotation_angle = 0

    # three radio buttons that define the mode
    @QtCore.Slot()
    def digit_reconition_button_clicked(self):
        print('Digit recognition button clicked')
        self.mode = 'digit_recognition'
    @QtCore.Slot()
    def object_detection_button_clicked(self):
        print('Object detection button clicked')
        self.mode = 'object_detection'
    @QtCore.Slot()
    def piv_button_clicked(self):
        print('PIV button clicked')
        self.mode = 'PIV'

    # push button that loads data
    @QtCore.Slot()
    def load_image_clicked(self):
        self.image_paths, _ = QFileDialog.getOpenFileNames(self, QObject.tr('Load Test Image'))
        # in case user did not load any image
        if self.image_paths == []:
            return
        print(f'Loaded image(s) {self.image_paths}')

        # load the image and scale the size
        self.loaded_images_pt = []
        self.cur_images_pt = []
        self.display_images = []
        self.loaded_image_names = []
        # get the label of the image(s)
        self.loaded_image_labels = []
        for i in range(len(self.image_paths)):
            self.loaded_images_pt.append(torch.from_numpy(np.asarray(Image.open(self.image_paths[i])))[:, :, None])
            self.loaded_image_names.append(self.image_paths[i].split('/')[-1])
            self.loaded_image_labels.append(int(self.image_paths[i].split('/')[-1].split('_')[1]))

            # keep a copy to represent the current (rotated) version of the original images
            self.cur_images_pt.append(self.loaded_images_pt[-1].clone())
            # convert to QImage for display purpose
            self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_images_pt[-1])
            # resize the display QImage
            self.display_images.append(self.cur_display_image.scaledToWidth(self.display_image_size))

        # display the image
        self.display_image()
        self.data_button.setText(f'Click to load new image')
        self.image_existed = True

        # show the run button when data is loaded
        # run once button (single sample)
        self.run_once_button = QtWidgets.QPushButton('Run model once')
        self.run_button_layout.addWidget(self.run_once_button)
        self.run_once_button.clicked.connect(self.run_once_button_clicked)
        # run all button (all transformations on single sample)
        self.run_all_button = QtWidgets.QPushButton('Run model on all transformations')
        self.run_button_layout.addWidget(self.run_all_button)
        self.run_all_button.clicked.connect(self.run_all_button_clicked)

    # two drop down menus that let user choose models
    @QtCore.Slot()
    def model_1_text_changed(self, text):
        print('Model 1:', text)
        self.model_1_name = text
        # load the mode
        if text == 'Simple model':
            self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
            # reload model
            self.model_1 = nero_run_model.load_model('non-eqv', self.model_1_path)
        elif text == 'E2CNN model':
            self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
            # reload model
            self.model_1 = nero_run_model.load_model('rot-eqv', self.model_1_path)
        elif text == 'Data augmentation model':
            self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
            # reload model
            self.model_1 = nero_run_model.load_model('aug-eqv', self.model_1_path)

        print('Model 1 path:', self.model_1_path)


    @QtCore.Slot()
    def model_2_text_changed(self, text):
        print('Model 2:', text)
        self.model_2_name = text
        # load the mode
        if text == 'Simple model':
            self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
            # reload model
            self.model_2 = nero_run_model.load_model('non-eqv', self.model_2_path)
        elif text == 'E2CNN model':
            self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
            # reload model
            self.model_2 = nero_run_model.load_model('rot-eqv', self.model_2_path)
        elif text == 'Data augmentation model':
            self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
            # reload model
            self.model_2 = nero_run_model.load_model('aug-eqv', self.model_2_path)

        print('Model 2 path:', self.model_2_path)


    @QtCore.Slot()
    def run_once_button_clicked(self):
        self.run_model_once()


    @QtCore.Slot()
    def run_all_button_clicked(self):
        self.run_model_all()

    # run model on a single test sample
    def run_model_once(self):
        if self.mode == 'digit_recognition':
            output_1 = nero_run_model.run_mnist_once(self.model_1, self.cur_images_pt[0])
            output_2 = nero_run_model.run_mnist_once(self.model_2, self.cur_images_pt[0])
            # display the result
            # add a new label for result if no result has existed
            if not self.result_existed:
                self.mnist_label = QLabel(self)
                self.mnist_label.setAlignment(QtCore.Qt.AlignCenter)
                self.mnist_label.setWordWrap(True)
                self.mnist_label.setTextFormat(QtGui.Qt.AutoText)
                self.result_existed = True
                self.repaint = False
            else:
                self.repaint = True

            self.display_mnist_result(self.model_1_name, output_1,
                                        self.model_2_name, output_2, boundary_width=3, repaint=self.repaint)

    # run model on all the available transformations on a single sample
    def run_model_all(self):
        if self.mode == 'digit_recognition':
            # run all rotation test with 5 degree increment
            output_1 = nero_run_model.run_mnist_all_rotations(self.model_1, self.cur_image_pt)
            output_2 = nero_run_model.run_mnist_all_rotations(self.model_2, self.cur_image_pt)

    def display_image(self):

        # single image case
        if len(self.display_images) == 1:
            # prepare a pixmap for the image
            image_pixmap = QPixmap(self.display_images[0])

            # add a new label for loaded image if no imager has existed
            if not self.image_existed:
                self.image_label = QLabel(self)
                self.image_label.setAlignment(QtCore.Qt.AlignCenter)
                self.image_existed = True

            # put pixmap in the label
            self.image_label.setPixmap(image_pixmap)

            # name of the image
            name_label = QLabel(self.loaded_image_names[0])
            name_label.setAlignment(QtCore.Qt.AlignCenter)

            # add this image to the layout
            self.loaded_layout.addWidget(self.image_label, 0, 0)
            self.loaded_layout.addWidget(name_label, 1, 0)

        # when loaded multiple images

    # draw arrow
    def draw_arrow(self, painter, pen, width, height, boundary_width):
        # draw arrow to indicate feeding
        pen.setWidth(boundary_width)
        pen.setColor(QtGui.QColor('black'))
        painter.setPen(pen)
        # horizontal line
        painter.drawLine(0, height//2, width, height//2)
        # upper arrow
        painter.drawLine(int(0.6*width), int(0.45*height), width, height//2)
        # bottom arrow
        painter.drawLine(int(0.6*width), int(0.55*height), width, height//2)


    def display_mnist_result(self, name_1, output_1, name_2, output_2, boundary_width, repaint=False):

        # use the loaded_layout
        mnist_pixmap = QPixmap(80, 300)
        mnist_pixmap.fill(QtCore.Qt.white)
        # draw arrow
        painter = QtGui.QPainter(mnist_pixmap)
        # set pen (used to draw outlines of shapes) and brush (draw the background of a shape)
        pen = QtGui.QPen()
        # draw arrow to indicate feeding
        self.draw_arrow(painter, pen, 80, 300, boundary_width)

        self.mnist_label.setPixmap(mnist_pixmap)
        # add to the layout
        self.loaded_layout.addWidget(self.mnist_label, 0, 1)

        # draw result using bar plot
        # all the different classes
        categories = []
        for i in range(len(output_1)):
            categories.append(str(i))

        # all results
        model_1_set = QBarSet(name_1)
        model_1_set.append(list(output_1))
        model_2_set = QBarSet(name_2)
        model_2_set.append(list(output_2))

        # bar series
        bar_series = QBarSeries()
        bar_series.append(model_1_set)
        bar_series.append(model_2_set)

        chart = QChart()
        chart.addSeries(bar_series)
        # animation when plotting
        if repaint:
            chart.setAnimationOptions(QChart.NoAnimation)
        else:
            chart.setAnimationOptions(QChart.SeriesAnimations)

        # x and y axis
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.setAxisX(axis_x, bar_series)
        axis_y = QValueAxis()
        chart.setAxisY(axis_y, bar_series)
        axis_y.setRange(0, 1)

        # legend
        chart.legend().setVisible(True)
        chart.legend().setAlignment(QtGui.Qt.AlignBottom)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QtGui.QPainter.Antialiasing)
        # add to the layout
        self.loaded_layout.addWidget(chart_view, 0, 2)

        painter.end()


        return mnist_pixmap

    def mouseMoveEvent(self, event):
        # print("mouseMoveEvent")
        # when in translation mode
        if self.translation:
            print('translating')
        # when in rotation mode
        elif self.rotation:
            cur_mouse_pos = [event.position().x(), event.position().y()]

            angle_change = -((self.prev_mouse_pos[0]*cur_mouse_pos[1] - self.prev_mouse_pos[1]*cur_mouse_pos[0])
                            / (self.prev_mouse_pos[0]*self.prev_mouse_pos[0] + self.prev_mouse_pos[1]*self.prev_mouse_pos[1]))*180

            self.cur_rotation_angle += angle_change
            # rotate the image tensor
            self.cur_images_pt[0] = nero_transform.rotate_mnist_image(self.loaded_images_pt[0], self.cur_rotation_angle)
            # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(angle), QtCore.Qt.SmoothTransformation)
            # convert image tensor to qt image and resize for display
            self.display_images[0] = nero_utilities.tensor_to_qt_image(self.cur_images_pt[0]).scaledToWidth(self.display_image_size)
            # update the pixmap and label
            self.image_pixmap = QPixmap(self.display_images[0])
            self.image_label.setPixmap(self.image_pixmap)

            # update the model output
            if self.result_existed:
                self.run_model_once()

            self.prev_mouse_pos = cur_mouse_pos

    def mousePressEvent(self, event):
        print("mousePressEvent")
        self.prev_mouse_pos = [event.position().x(), event.position().y()]

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













# @QtCore.Slot()
# def load_model_clicked(self):
#     self.model_path, _ = QFileDialog.getOpenFileName(self, QObject.tr('Load Model'))
#     # in case user did not load any image
#     if self.model_path == '':
#         return
#     print(f'Loaded model {self.model_path}')

#     model_name = self.model_path.split('/')[-1]
#     width = 300
#     height = 300
#     # display the model
#     self.display_model(model_name, width, height, boundary_width=3)
#     # change the button text
#     self.model_button.setText(f'Loaded model {model_name}. Click to load new model')

#     # show the run button when both ready
#     if self.model_existed and self.image_existed and not self.run_buttons_existed:
#         # run once button
#         self.run_once_button = QtWidgets.QPushButton('Run model once')
#         self.run_button_layout.addWidget(self.run_once_button)
#         self.run_once_button.clicked.connect(self.run_once_button_clicked)
#         # load model button
#         self.run_all_button = QtWidgets.QPushButton('Run model on all transformations')
#         self.run_button_layout.addWidget(self.run_all_button)
#         self.run_all_button.clicked.connect(self.run_all_button_clicked)

#         self.run_buttons_existed = True

# draw model diagram, return model pixmap
# def draw_model_diagram(self, painter, pen, name, font_size, width, height, boundary_width):

#     # draw rectangle to represent model
#     pen.setWidth(boundary_width)
#     pen.setColor(QtGui.QColor('red'))
#     painter.setPen(pen)
#     rectangle = QtCore.QRect(int(width//3)+boundary_width, boundary_width, width//3*2-2*boundary_width, height-2*boundary_width)
#     painter.drawRect(rectangle)

#     # draw model name
#     painter.setFont(QFont('Helvetica', font_size))
#     if len(name) > 20:
#         name = name[:20] + '\n' + name[20:]
#         painter.drawText(int(width//3)+boundary_width, height//2-6*boundary_width, width//3*2, height, QtGui.Qt.AlignHCenter, name)
#     else:
#         painter.drawText(int(width//3)+boundary_width, height//2-2*boundary_width, width//3*2, height, QtGui.Qt.AlignHCenter, name)

# might be useful later
# def display_model(self, model_name, width, height, boundary_width):
#     # add a new label for loaded image if no image has existed
#     if not self.model_existed:
#         self.model_label = QLabel(self)
#         self.model_label.setWordWrap(True)
#         self.model_label.setTextFormat(QtGui.Qt.AutoText)
#         self.model_label.setAlignment(QtCore.Qt.AlignLeft)
#         self.model_existed = True

#     # total model pixmap size
#     model_pixmap = QPixmap(width, height)
#     model_pixmap.fill(QtCore.Qt.white)

#     # define painter that is working on the pixmap
#     painter = QtGui.QPainter(model_pixmap)
#     # set pen (used to draw outlines of shapes) and brush (draw the background of a shape)
#     pen = QtGui.QPen()

#     # draw standard arrow
#     self.draw_arrow(painter, pen, 80, 150, boundary_width)
#     # draw the model diagram
#     self.draw_model_diagram(painter, pen, model_name, 12, width, 150, boundary_width)
#     painter.end()

#     # add to the label and layout
#     self.model_label.setPixmap(model_pixmap)
#     self.loaded_layout.addWidget(self.model_label, 0, 2)