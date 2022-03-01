from gc import callbacks
import os
import sys
import glob
import time
import torch
import numpy as np
from PIL import Image
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui  import QPixmap, QFont
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QWidget, QLabel, QRadioButton

import nero_transform
import nero_utilities
import nero_run_model

# globa configurations
pg.setConfigOptions(antialias=True, background='w')
# use pyside gpu acceleration if gpu detected
if torch.cuda.is_available():
    pg.setConfigOption('useCupy', True)
    os.environ['CUDA_VISIBLE_DEVICES']='1'
else:
    pg.setConfigOption('useCupy', False)


class UI_MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # set window title
        self.setWindowTitle('Non-Equivariance Revealed on Orbits')
        # white background color
        self.setStyleSheet('background-color: white;')
        # general layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignCenter)
        # left, top, right, and bottom margins
        self.layout.setContentsMargins(50, 50, 50, 50)

        # individual laytout for different widgets
        # title (no title)
        # self.init_title_layout()
        # mode selections
        self.mode = 'digit_recognition'
        # save the previous mode selection for layout swap
        self.previous_mode = None
        # if any mode selection has been made
        self.selection_existed = False
        self.init_control_layout()
        self.run_button_existed = False

        print(f'\nFinished rendering main layout')


    # helper function that recursively clears a layout
    def clear_layout(self, layout):
        if layout:
            while layout.count():
                # remove the item at index 0 from the layout, and return the item
                item = layout.takeAt(0)
                # delete if it is a widget
                if item.widget():
                    item.widget().deleteLater()
                # delete if it is a layout
                else:
                    self.clear_layout(item.layout())

            layout.deleteLater()


    def init_title_layout(self):
        # title
        self.title_layout = QtWidgets.QVBoxLayout()
        self.title_layout.setContentsMargins(0, 0, 0, 20)
        # title of the application
        self.title = QLabel('Non-Equivariance Revealed on Orbits',
                            alignment=QtCore.Qt.AlignCenter)
        self.title.setFont(QFont('Helvetica', 24))
        self.title_layout.addWidget(self.title)
        self.title_layout.setContentsMargins(0, 0, 0, 50)

        # add to general layout
        self.layout.addLayout(self.title_layout)


    def init_control_layout(self):
        # three radio buttons that define the mode
        @QtCore.Slot()
        def digit_recognition_button_clicked():
            print('Digit recognition button clicked')
            self.mode = 'digit_recognition'
            self.radio_button_1.setChecked(True)

            if self.previous_mode != self.mode:
                # clear the previous method's layout
                if self.selection_existed:
                    self.clear_layout(self.control_layout)
                    self.clear_layout(self.result_layout)

                self.init_load_layout()
                self.init_result_layout()

                # display mnist image size
                self.display_image_size = 150
                # image (input data) modification mode
                self.rotation = True
                self.translation = False
                # rotation angles
                self.cur_rotation_angle = 0

                # predefined model paths
                self.model_1_name = 'Simple model'
                self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
                self.model_2_name = 'Data augmentation model'
                self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
                # preload model
                self.model_1 = nero_run_model.load_mnist_model('non-eqv', self.model_1_path)
                self.model_2 = nero_run_model.load_mnist_model('aug-eqv', self.model_2_path)

                # unique quantity of the result of current data
                self.all_quantities_1 = []
                self.all_quantities_2 = []

                # when doing highlighting
                # last_clicked is not None when it is either clicked or during manual rotation
                self.last_clicked = None
                self.cur_line = None

                self.previous_mode = self.mode


        @QtCore.Slot()
        def object_detection_button_clicked():
            print('Object detection button clicked')
            self.mode = 'object_detection'
            self.radio_button_2.setChecked(True)

            # below layouts depend on mode selection
            if self.previous_mode != self.mode:
                if self.selection_existed:
                    self.clear_layout(self.control_layout)
                    self.clear_layout(self.result_layout)

                self.init_load_layout()
                self.init_result_layout()

                # display mnist image size
                self.display_image_size = 150
                # image (input data) modification mode
                self.rotation = False
                self.translation = False
                # rotation angles
                self.cur_rotation_angle = 0

                # predefined model paths
                # model_1_name = 'Simple model'
                self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
                # model_2_name = 'Data augmentation model'
                self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
                # preload model
                self.model_1 = nero_run_model.load_mnist_model('non-eqv', self.model_1_path)
                self.model_2 = nero_run_model.load_mnist_model('aug-eqv', self.model_2_path)

                # unique quantity of the result of current data
                self.all_quantities_1 = []
                self.all_quantities_2 = []

                # when doing highlighting
                # last_clicked is not None when it is either clicked or during manual rotation
                self.last_clicked = None
                self.cur_line = None

                self.previous_mode = self.mode

        @QtCore.Slot()
        def piv_button_clicked():
            print('PIV button clicked')
            self.mode = 'PIV'
            self.radio_button_3.setChecked(True)

            # below layouts depend on mode selection
            if self.selection_existed:
                self.clear_layout(self.control_layout)
                self.clear_layout(self.result_layout)

            self.init_load_layout()
            self.init_result_layout()

            # display mnist image size
            self.display_image_size = 150
            # image (input data) modification mode
            self.rotation = False
            # rotation angles
            self.cur_rotation_angle = 0

            # predefined model paths
            self.model_1_name = 'Simple model'
            self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
            self.model_2_name = 'Data augmentation model'
            self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
            # preload model
            self.model_1 = nero_run_model.load_mnist_model('non-eqv', self.model_1_path)
            self.model_2 = nero_run_model.load_mnist_model('aug-eqv', self.model_2_path)

            # unique quantity of the result of current data
            self.all_quantities_1 = []
            self.all_quantities_2 = []

            # when doing highlighting
            # last_clicked is not None when it is either clicked or during manual rotation
            self.last_clicked = None
            self.cur_line = None


        # mode selection radio buttons
        self.control_layout = QtWidgets.QGridLayout()
        self.control_layout.setContentsMargins(50, 0, 0, 50)
        # radio buttons on mode selection (digit_recognition, object detection, PIV)
        # title
        mode_pixmap = QPixmap(150, 30)
        mode_pixmap.fill(QtCore.Qt.white)
        # draw text
        painter = QtGui.QPainter(mode_pixmap)
        painter.setFont(QFont('Helvetica', 18))
        painter.drawText(0, 0, 150, 30, QtGui.Qt.AlignLeft, 'Model type: ')
        painter.end()

        # create label to contain the texts
        self.mode_label = QLabel(self)
        self.mode_label.setAlignment(QtCore.Qt.AlignLeft)
        self.mode_label.setWordWrap(True)
        self.mode_label.setTextFormat(QtGui.Qt.AutoText)
        self.mode_label.setPixmap(mode_pixmap)
        # add to the layout
        self.control_layout.addWidget(self.mode_label, 0, 0)

        # radio_buttons_layout = QtWidgets.QGridLayout(self)
        self.radio_button_1 = QRadioButton('Digit recognition')
        self.radio_button_1.setStyleSheet('QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};')
        self.radio_button_1.pressed.connect(digit_recognition_button_clicked)
        self.control_layout.addWidget(self.radio_button_1, 0, 1)
        # spacer item
        # self.control_layout.addSpacing(30)

        self.radio_button_2 = QRadioButton('Object detection')
        self.radio_button_2.setStyleSheet('QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};')
        self.radio_button_2.pressed.connect(object_detection_button_clicked)
        self.control_layout.addWidget(self.radio_button_2, 1, 1)
        # spacer item
        # self.control_layout.addSpacing(30)

        self.radio_button_3 = QRadioButton('Particle Image Velocimetry (PIV)')
        self.radio_button_3.setStyleSheet('QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};')
        self.radio_button_3.pressed.connect(piv_button_clicked)
        self.control_layout.addWidget(self.radio_button_3, 2, 1)

        # add to general layout
        self.layout.addLayout(self.control_layout)

        # used for default state
        if self.mode == 'digit_recognition':
            self.radio_button_1.setChecked(True)
            digit_recognition_button_clicked()
        elif self.mode == 'object_detection':
            self.radio_button_2.setChecked(True)
            object_detection_button_clicked()
        elif self.mode == 'piv':
            self.radio_button_3.setChecked(True)
            piv_button_clicked()

        self.selection_existed = True


    def init_load_layout(self):

        # push button that loads data
        @QtCore.Slot()
        def image_selection_changed(text):
            print('Loaded image:', text)
            self.image_index = int(text.split(' ')[-1])
            self.image_path = self.mnist_images_paths[self.image_index]
            # load the image and scale the size
            self.loaded_image_pt = torch.from_numpy(np.asarray(Image.open(self.image_path)))[:, :, None]
            self.loaded_image_name = self.image_path.split('/')[-1]
            self.loaded_image_label = int(self.image_path.split('/')[-1].split('_')[1])

            # keep a copy to represent the current (rotated) version of the original images
            self.cur_image_pt = self.loaded_image_pt.clone()
            # convert to QImage for display purpose
            self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt)
            # resize the display QImage
            self.cur_display_image = self.cur_display_image.scaledToWidth(self.display_image_size)

            # display the image
            self.display_image()
            self.image_existed = True

            # show the run button when data is loaded
            if not self.run_button_existed:
                # run button
                # buttons layout for run model
                self.run_button_layout = QtWidgets.QGridLayout()
                self.layout.addLayout(self.run_button_layout)

                self.run_button = QtWidgets.QPushButton('Analyze model')
                self.run_button.setStyleSheet('font-size: 18px')
                run_button_size = QtCore.QSize(500, 50)
                self.run_button.setMinimumSize(run_button_size)
                self.run_button_layout.addWidget(self.run_button)
                self.run_button.clicked.connect(self.run_button_clicked)

                self.run_button_existed = True

        # two drop down menus that let user choose models
        @QtCore.Slot()
        def model_1_selection_changed(text):
            print('Model 1:', text)
            self.model_1_name = text
            if self.mode == 'digit_recognition':
                # load the mode
                if text == 'Simple model':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_mnist_model('non-eqv', self.model_1_path)
                elif text == 'E2CNN model':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_mnist_model('rot-eqv', self.model_1_path)
                elif text == 'Data augmentation model':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_mnist_model('aug-eqv', self.model_1_path)
            elif self.mode == 'object_detection':
                if text == 'Simple model':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_coco_model('non-eqv', self.model_1_path)
                elif text == 'Shift-Invariant model':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_coco_model('rot-eqv', self.model_1_path)

            print('Model 1 path:', self.model_1_path)

        @QtCore.Slot()
        def model_2_selection_changed(text):
            print('Model 2:', text)
            self.model_2_name = text
            if self.mode == 'digit_recognition':
                # load the mode
                if text == 'Simple model':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_mnist_model('non-eqv', self.model_2_path)
                elif text == 'E2CNN model':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_mnist_model('rot-eqv', self.model_2_path)
                elif text == 'Data augmentation model':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_mnist_model('aug-eqv', self.model_2_path)
            elif self.mode == 'object_detection':
                if text == 'Simple model':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_coco_model('non-eqv', self.model_2_path)
                elif text == 'Shift-Invariant model':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_coco_model('rot-eqv', self.model_2_path)

            print('Model 2 path:', self.model_2_path)

        # function used as model icon
        def draw_circle(painter, center_x, center_y, radius, color):
            # optional
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            # make a white drawing background
            painter.setBrush(QtGui.QColor(color))
            # draw red circles
            painter.setPen(QtGui.QColor(color))
            center = QtCore.QPoint(center_x, center_y)
            # optionally fill each circle yellow
            painter.setBrush(QtGui.QColor(color))
            painter.drawEllipse(center, radius, radius)
            painter.end()

        # draw text
        model_pixmap = QPixmap(300, 30)
        model_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(model_pixmap)
        painter.setFont(QFont('Helvetica', 18))
        painter.drawText(0, 0, 300, 30, QtGui.Qt.AlignLeft, 'Data/Model Selection: ')
        painter.end()

        # create label to contain the texts
        self.model_label = QLabel(self)
        self.model_label.setAlignment(QtCore.Qt.AlignLeft)
        self.model_label.setWordWrap(True)
        self.model_label.setTextFormat(QtGui.Qt.AutoText)
        self.model_label.setPixmap(model_pixmap)
        # add to the layout
        self.control_layout.addWidget(self.model_label, 0, 2)

        # images loading drop down menus
        self.image_menu = QtWidgets.QComboBox()
        self.image_menu.setMinimumSize(QtCore.QSize(250, 50))
        self.image_menu.setStyleSheet('font-size: 18px')
        self.image_menu.addItem('Please select input image')
        if self.mode == 'digit_recognition':
            self.mnist_images_paths = []
            # add a image of each class
            for i in range(10):
                cur_image_path = glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, f'label_{i}*.png'))[0]
                self.mnist_images_paths.append(cur_image_path)
                self.image_menu.addItem(QtGui.QIcon(cur_image_path), f'Image {i}')

            self.image_menu.setCurrentIndex(0)

        elif self.mode == 'object_detection':
            self.coco_images_paths = []
            # add a image of each class
            for i in range(10):
                cur_image_path = glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, f'label_{i}*.png'))[0]
                self.coco_images_paths.append(cur_image_path)
                self.image_menu.addItem(QtGui.QIcon(cur_image_path), f'Image {i}')

            self.image_menu.setCurrentIndex(0)

        # connect the drop down menu with actions
        self.image_menu.currentTextChanged.connect(image_selection_changed)
        self.image_menu.setEditable(True)
        self.image_menu.lineEdit().setReadOnly(True)
        self.image_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.control_layout.addWidget(self.image_menu, 0, 3)

        # init flag to inidicate if an image has ever been loaded
        self.image_existed = False

        # load models choices
        # model 1
        # graphic representation
        self.model_1_label = QLabel(self)
        self.model_1_label.setAlignment(QtCore.Qt.AlignCenter)
        model_1_icon = QPixmap(25, 25)
        model_1_icon.fill(QtCore.Qt.white)
        # draw model representation
        painter = QtGui.QPainter(model_1_icon)
        draw_circle(painter, 12, 12, 10, 'blue')

        # spacer item
        # self.control_layout.addSpacing(30)

        model_1_menu = QtWidgets.QComboBox()
        model_1_menu.setMinimumSize(QtCore.QSize(250, 50))
        model_1_menu.setStyleSheet('font-size: 18px')
        if self.mode == 'digit_recognition':
            model_1_menu.addItem(model_1_icon, 'Simple model')
            model_1_menu.addItem(model_1_icon, 'E2CNN model')
            model_1_menu.addItem(model_1_icon, 'Data augmentation model')
            model_1_menu.setCurrentText('Simple model')
        elif self.mode == 'object_detection':
            model_1_menu.addItem(model_1_icon, 'Simple model')
            model_1_menu.addItem(model_1_icon, 'Shift-Invariant model')
            model_1_menu.setCurrentText('Simple model')

        # connect the drop down menu with actions
        model_1_menu.currentTextChanged.connect(model_1_selection_changed)
        model_1_menu.setEditable(True)
        model_1_menu.lineEdit().setReadOnly(True)
        model_1_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.control_layout.addWidget(model_1_menu, 1, 3)

        # model 2
        # graphic representation
        self.model_2_label = QLabel(self)
        self.model_2_label.setAlignment(QtCore.Qt.AlignCenter)
        model_2_icon = QPixmap(25, 25)
        model_2_icon.fill(QtCore.Qt.white)
        # draw model representation
        painter = QtGui.QPainter(model_2_icon)
        draw_circle(painter, 12, 12, 10, 'Green')

        # spacer item
        # self.control_layout.addSpacing(30)

        model_2_menu = QtWidgets.QComboBox()
        model_2_menu.setMinimumSize(QtCore.QSize(250, 50))
        model_2_menu.setStyleSheet('font-size: 18px')
        model_2_menu.setEditable(True)
        model_2_menu.lineEdit().setReadOnly(True)
        model_2_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        if self.mode == 'digit_recognition':
            model_2_menu.addItem(model_2_icon, 'Simple model')
            model_2_menu.addItem(model_2_icon, 'E2CNN model')
            model_2_menu.addItem(model_2_icon, 'Data augmentation model')
            model_2_menu.setCurrentText('Data augmentation model')
        elif self.mode == 'object_detection':
            model_2_menu.addItem(model_1_icon, 'Simple model')
            model_2_menu.addItem(model_1_icon, 'Shift-Invariant model')
            model_2_menu.setCurrentText('Simple model')

        # connect the drop down menu with actions
        model_2_menu.currentTextChanged.connect(model_2_selection_changed)
        self.control_layout.addWidget(model_2_menu, 2, 3)


    def init_result_layout(self):
        # loaded images and model layout
        self.result_layout = QtWidgets.QGridLayout()
        self.result_layout.setContentsMargins(30, 50, 30, 50)

        # add to general layout
        self.layout.addLayout(self.result_layout)

        # if model result ever existed
        self.result_existed = False


    # run button execution that could be used by all modes
    @QtCore.Slot()
    def run_button_clicked(self):
        # run model once and display results
        self.run_model_once()

        # run model all and display results
        self.run_model_all()


    # run model on a single test sample
    def run_model_once(self):
        if self.mode == 'digit_recognition':
            self.output_1 = nero_run_model.run_mnist_once(self.model_1, self.cur_image_pt)
            self.output_2 = nero_run_model.run_mnist_once(self.model_2, self.cur_image_pt)

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

            # display result
            self.display_mnist_result(mode='bar', boundary_width=3)
        elif self.mode == 'object_detection':
            print('Not yet implemented')


    # run model on all the available transformations on a single sample
    def run_model_all(self):
        if self.mode == 'digit_recognition':
            self.all_angles = []
            self.all_quantities_1 = []
            self.all_quantities_2 = []
            # run all rotation test with 5 degree increment
            for self.cur_rotation_angle in range(0, 365, 5):
                # print(f'\nRotated {self.cur_rotation_angle} degrees')
                self.all_angles.append(self.cur_rotation_angle)
                # rotate the image tensor
                self.cur_image_pt = nero_transform.rotate_mnist_image(self.loaded_image_pt, self.cur_rotation_angle)
                # convert image tensor to qt image and resize for display
                self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt).scaledToWidth(self.display_image_size)
                # update the pixmap and label
                image_pixmap = QPixmap(self.cur_display_image)
                self.image_label.setPixmap(image_pixmap)
                # force repaint
                self.image_label.repaint()

                # update the model output
                self.output_1 = nero_run_model.run_mnist_once(self.model_1, self.cur_image_pt)
                self.output_2 = nero_run_model.run_mnist_once(self.model_2, self.cur_image_pt)

                # plotting the quantity regarding the correct label
                quantity_1 = self.output_1[self.loaded_image_label]
                quantity_2 = self.output_2[self.loaded_image_label]

                self.all_quantities_1.append(quantity_1)
                self.all_quantities_2.append(quantity_2)

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

            # display result
            self.display_mnist_result(mode='polar', boundary_width=3)

        elif self.mode == 'object_detection':
            print('Not yet implemented')


    def display_image(self):

        # single image case
        # prepare a pixmap for the image
        self.image_pixmap = QPixmap(self.cur_display_image)

        # add a new label for loaded image if no imager has existed
        if not self.image_existed:
            self.image_label = QLabel(self)
            self.image_label.setAlignment(QtCore.Qt.AlignCenter)
            self.image_existed = True

        # put pixmap in the label
        self.image_label.setPixmap(self.image_pixmap)
        self.image_label.setContentsMargins(0, 0, 0, 0)

        # name of the image
        name_label = QLabel(self.loaded_image_name)
        name_label.setAlignment(QtCore.Qt.AlignCenter)

        # add this image to the layout
        self.result_layout.addWidget(self.image_label, 0, 0)
        self.result_layout.addWidget(name_label, 1, 0)

        # when loaded multiple images


    # draw an arrow, used between input image(s) and model outputs
    def draw_arrow(self, painter, pen, width, height, boundary_width):
        # draw arrow to indicate feeding
        pen.setWidth(boundary_width)
        pen.setColor(QtGui.QColor('black'))
        painter.setPen(pen)
        # horizontal line
        painter.drawLine(0, height//2, width, height//2)
        # upper arrow
        painter.drawLine(int(0.6*width), int(0.25*height), width, height//2)
        # bottom arrow
        painter.drawLine(int(0.6*width), int(0.75*height), width, height//2)



    def display_mnist_result(self, mode, boundary_width):

        # use the result_layout
        mnist_pixmap = QPixmap(100, 50)
        mnist_pixmap.fill(QtCore.Qt.white)
        # draw arrow
        painter = QtGui.QPainter(mnist_pixmap)
        # set pen (used to draw outlines of shapes) and brush (draw the background of a shape)
        pen = QtGui.QPen()
        # draw arrow to indicate feeding
        self.draw_arrow(painter, pen, 100, 50, boundary_width)

        # add to the label and layout
        self.mnist_label.setPixmap(mnist_pixmap)
        self.result_layout.addWidget(self.mnist_label, 0, 1)

        # draw result using bar plot
        if mode == 'bar':
            # create individual bar (item) for individual hover/click control
            class InteractiveBarItem(pg.BarGraphItem):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    if self.opts['brush'] == 'blue':
                        cur_class = int(self.opts.get('x0')[0] + 0.2)
                        cur_value = self.opts.get('height')
                    elif self.opts['brush'] == 'green':
                        cur_class = int(self.opts.get('x0')[0] - 0.2)
                        cur_value = self.opts.get('height')

                    model_name = self.name()
                    self.hover_text = f'{model_name}(x = {cur_class}) = {cur_value}'
                    self.setToolTip(self.hover_text)

                    # required in order to receive hoverEnter/Move/Leave events
                    self.setAcceptHoverEvents(True)

                def hoverEnterEvent(self, event):
                    print('hover!')

                def mousePressEvent(self, event):
                    print('click!')

            self.bar_plot = pg.plot()
            # constrain plot showing limit by setting view box
            self.bar_plot.plotItem.vb.setLimits(xMin=-0.5, xMax=9.5, yMin=0, yMax=1.2)
            self.bar_plot.setBackground('w')
            self.bar_plot.setFixedSize(700, 600)
            # for i in range(10):
            #     cur_graph_1 = InteractiveBarItem(name=f'{self.model_1_name}',
            #                                      x0=[i-0.2],
            #                                      height=self.output_1[i],
            #                                      width=0.4,
            #                                      brush='blue')

            #     cur_graph_2 = InteractiveBarItem(name=f'{self.model_2_name}',
            #                                      x0=[i+0.2],
            #                                      height=self.output_2[i],
            #                                      width=0.4,
            #                                      brush='green')

            #     self.bar_plot.addItem(cur_graph_1)
            #     self.bar_plot.addItem(cur_graph_2)

            graph_1 = pg.BarGraphItem(x=np.arange(len(self.output_1))-0.2, height = list(self.output_1), width = 0.4, brush ='blue')
            graph_2 = pg.BarGraphItem(x=np.arange(len(self.output_1))+0.2, height = list(self.output_2), width = 0.4, brush ='green')
            self.bar_plot.addItem(graph_1)
            self.bar_plot.addItem(graph_2)
            # disable moving around
            self.bar_plot.setMouseEnabled(x=False, y=False)

            self.result_layout.addWidget(self.bar_plot, 0, 2)

        elif mode == 'polar':
            # draw a polar plot
            def draw_polar(plot):
                plot.setXRange(-1, 1)
                plot.setYRange(-1, 1)
                plot.setAspectLocked()

                # Add polar grid lines
                plot.addLine(x=0, pen=pg.mkPen('black', width=2))
                plot.addLine(y=0, pen=pg.mkPen('black', width=2))
                for r in np.arange(0, 1.2, 0.2):
                    circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
                    circle.setPen(pg.mkPen('black', width=2))
                    plot.addItem(circle)

                return plot

            # helper function for clicking inside polar plot
            def clicked(plot, points):
                # clear manual mode line
                if self.cur_line:
                    self.polar_plot.removeItem(self.cur_line)

                # clear previously selected point's visual cue
                if self.last_clicked:
                    self.last_clicked.resetPen()
                    self.last_clicked.setBrush(self.old_brush)

                # clicked point's position
                x_pos = points[0].pos().x()
                y_pos = points[0].pos().y()

                # convert back to polar coordinate
                radius = np.sqrt(x_pos**2 + y_pos**2)
                self.cur_rotation_angle = np.arctan2(y_pos, x_pos) / np.pi * 180

                # update the current image's angle and rotate the display image
                # rotate the image tensor
                self.cur_image_pt = nero_transform.rotate_mnist_image(self.loaded_image_pt, self.cur_rotation_angle)
                # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(angle), QtCore.Qt.SmoothTransformation)
                # convert image tensor to qt image and resize for display
                self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt).scaledToWidth(self.display_image_size)
                # update the pixmap and label
                self.image_pixmap = QPixmap(self.cur_display_image)
                self.image_label.setPixmap(self.image_pixmap)

                # update the model output
                if self.result_existed:
                    self.run_model_once()

                # only allow clicking one point at a time
                # save the old brush
                if points[0].brush() == pg.mkBrush(0, 0, 255, 150):
                    self.old_brush = pg.mkBrush(0, 0, 255, 150)

                elif points[0].brush() == pg.mkBrush(0, 255, 0, 150):
                    self.old_brush = pg.mkBrush(0, 255, 0, 150)

                # create new brush
                new_brush = pg.mkBrush(255, 0, 0, 255)
                points[0].setBrush(new_brush)
                points[0].setPen(5)

                self.last_clicked = points[0]

            # initialize view and plot
            polar_view = pg.GraphicsLayoutWidget()
            polar_view.setBackground('white')
            polar_view.setFixedSize(600, 600)
            self.polar_plot = polar_view.addPlot()
            self.polar_plot = draw_polar(self.polar_plot)

            # Set pxMode=False to allow spots to transform with the view
            # all the points to be plotted
            self.scatter_items = pg.ScatterPlotItem(pxMode=False)
            all_points_1 = []
            all_points_2 = []
            all_x_1 = []
            all_y_1 = []
            all_x_2 = []
            all_y_2 = []
            for i in range(len(self.all_angles)):
                radian = self.all_angles[i] / 180 * np.pi
                # model 1 quantity
                cur_quantity_1 = self.all_quantities_1[i]
                # Transform to cartesian and plot
                x_1 = cur_quantity_1 * np.cos(radian)
                y_1 = cur_quantity_1 * np.sin(radian)
                all_x_1.append(x_1)
                all_y_1.append(y_1)
                all_points_1.append({'pos': (x_1, y_1),
                                    'size': 0.05,
                                    'pen': {'color': 'w', 'width': 0.1},
                                    'brush': (0, 0, 255, 150)})

                # model 2 quantity
                cur_quantity_2 = self.all_quantities_2[i]
                # Transform to cartesian and plot
                x_2 = cur_quantity_2 * np.cos(radian)
                y_2 = cur_quantity_2 * np.sin(radian)
                all_x_2.append(x_2)
                all_y_2.append(y_2)
                all_points_2.append({'pos': (x_2, y_2),
                                    'size': 0.05,
                                    'pen': {'color': 'w', 'width': 0.1},
                                    'brush': (0, 255, 0, 150)})

            # draw lines to better show shape
            line_1 = self.polar_plot.plot(all_x_1, all_y_1, pen = QtGui.QPen(QtGui.Qt.blue, 0.03))
            line_2 = self.polar_plot.plot(all_x_2, all_y_2, pen = QtGui.QPen(QtGui.Qt.green, 0.03))

            # add points to the item
            self.scatter_items.addPoints(all_points_1)
            self.scatter_items.addPoints(all_points_2)

            # add points to the plot
            self.polar_plot.addItem(self.scatter_items)
            # connect click events on scatter items
            self.scatter_items.sigClicked.connect(clicked)

            # used for clicking on the polar plot
            def polar_mouse_clicked(event):
                self.polar_clicked = not self.polar_clicked
                self.polar_plot.scene().items(event.scenePos())
                # check if the click is within the polar plot
                if self.polar_plot.sceneBoundingRect().contains(event._scenePos):
                    self.mouse_pos_on_polar = self.polar_plot.vb.mapSceneToView(event._scenePos)
                    x_pos = self.mouse_pos_on_polar.x()
                    y_pos = self.mouse_pos_on_polar.y()

                    # convert mouse click position to polar coordinate (we care about angle only)
                    self.cur_rotation_angle = np.arctan2(y_pos, x_pos) / np.pi * 180

                    # update the current image's angle and rotate the display image
                    # rotate the image tensor
                    self.cur_image_pt = nero_transform.rotate_mnist_image(self.loaded_image_pt, self.cur_rotation_angle)
                    # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(angle), QtCore.Qt.SmoothTransformation)
                    # convert image tensor to qt image and resize for display
                    self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt).scaledToWidth(self.display_image_size)
                    # update the pixmap and label
                    self.image_pixmap = QPixmap(self.cur_display_image)
                    self.image_label.setPixmap(self.image_pixmap)

                    # update the model output
                    if self.result_existed:
                        self.run_model_once()

                    # remove old line
                    if self.cur_line:
                        self.polar_plot.removeItem(self.cur_line)

                    # draw a line that represents current angle of rotation
                    cur_x = 1 * np.cos(self.cur_rotation_angle/180*np.pi)
                    cur_y = 1 * np.sin(self.cur_rotation_angle/180*np.pi)
                    line_x = [0, cur_x]
                    line_y = [0, cur_y]
                    self.cur_line = self.polar_plot.plot(line_x, line_y, pen = QtGui.QPen(QtGui.Qt.red, 0.03))

            def polar_mouse_moved(event):
                # check if the click is within the polar plot
                if self.polar_clicked and self.polar_plot.sceneBoundingRect().contains(event):
                    self.mouse_pos_on_polar = self.polar_plot.vb.mapSceneToView(event)
                    x_pos = self.mouse_pos_on_polar.x()
                    y_pos = self.mouse_pos_on_polar.y()

                    # convert mouse click position to polar coordinate (we care about angle only)
                    self.cur_rotation_angle = np.arctan2(y_pos, x_pos) / np.pi * 180

                    # update the current image's angle and rotate the display image
                    # rotate the image tensor
                    self.cur_image_pt = nero_transform.rotate_mnist_image(self.loaded_image_pt, self.cur_rotation_angle)
                    # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(angle), QtCore.Qt.SmoothTransformation)
                    # convert image tensor to qt image and resize for display
                    self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt).scaledToWidth(self.display_image_size)
                    # update the pixmap and label
                    self.image_pixmap = QPixmap(self.cur_display_image)
                    self.image_label.setPixmap(self.image_pixmap)

                    # update the model output
                    if self.result_existed:
                        self.run_model_once()

                    # remove old line
                    if self.cur_line:
                        self.polar_plot.removeItem(self.cur_line)

                    # draw a line that represents current angle of rotation
                    cur_x = 1 * np.cos(self.cur_rotation_angle/180*np.pi)
                    cur_y = 1 * np.sin(self.cur_rotation_angle/180*np.pi)
                    line_x = [0, cur_x]
                    line_y = [0, cur_y]
                    self.cur_line = self.polar_plot.plot(line_x, line_y, pen = QtGui.QPen(QtGui.Qt.red, 0.03))

            self.polar_clicked = False
            self.polar_plot.scene().sigMouseClicked.connect(polar_mouse_clicked)
            self.polar_plot.scene().sigMouseMoved.connect(polar_mouse_moved)

            # fix zoom level
            # self.polar_plot.vb.scaleBy((0.5, 0.5))
            self.polar_plot.setMouseEnabled(x=False, y=False)

            # add the plot view to the layout
            self.result_layout.addWidget(polar_view, 0, 3)

        else:
            raise Exception('Unsupported display mode')

        painter.end()


    def mouseMoveEvent(self, event):
        # print("mouseMoveEvent")
        # when in translation mode
        if self.translation:
            print('translating')
        # when in rotation mode
        elif self.rotation and self.image_existed:
            cur_mouse_pos = [event.position().x()-self.image_center_x, event.position().y()-self.image_center_y]

            angle_change = -((self.prev_mouse_pos[0]*cur_mouse_pos[1] - self.prev_mouse_pos[1]*cur_mouse_pos[0])
                            / (self.prev_mouse_pos[0]*self.prev_mouse_pos[0] + self.prev_mouse_pos[1]*self.prev_mouse_pos[1]))*180

            self.cur_rotation_angle += angle_change
            # print(f'\nRotated {self.cur_rotation_angle} degrees')
            # rotate the image tensor
            self.cur_image_pt = nero_transform.rotate_mnist_image(self.loaded_image_pt, self.cur_rotation_angle)
            # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(angle), QtCore.Qt.SmoothTransformation)
            # convert image tensor to qt image and resize for display
            self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt).scaledToWidth(self.display_image_size)
            # update the pixmap and label
            self.image_pixmap = QPixmap(self.cur_display_image)
            self.image_label.setPixmap(self.image_pixmap)

            # update the model output
            if self.result_existed:
                self.run_model_once()

                # remove old line
                if self.cur_line:
                    self.polar_plot.removeItem(self.cur_line)

                # draw a line that represents current angle of rotation
                cur_x = 1 * np.cos(self.cur_rotation_angle/180*np.pi)
                cur_y = 1 * np.sin(self.cur_rotation_angle/180*np.pi)
                line_x = [0, cur_x]
                line_y = [0, cur_y]
                self.cur_line = self.polar_plot.plot(line_x, line_y, pen = QtGui.QPen(QtGui.Qt.red, 0.03))

            self.prev_mouse_pos = cur_mouse_pos


    def mousePressEvent(self, event):
        print('\nmousePressEvent')
        # used for rotating the input image
        if self.image_existed:
            self.image_center_x = self.image_label.x() + self.image_label.width()/2
            self.image_center_y = self.image_label.y() + self.image_label.height()/2
            self.prev_mouse_pos = [event.position().x()-self.image_center_x, event.position().y()-self.image_center_y]

    # def mouseReleaseEvent(self, event):
    #     print("mouseReleaseEvent")


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
    widget.resize(1920, 1080)
    widget.show()

    sys.exit(app.exec())










# # load data button
# self.data_button = QtWidgets.QPushButton('Load Test Image')
# self.data_button.setStyleSheet('font-size: 18px')
# data_button_size = QtCore.QSize(500, 50)
# self.data_button.setMinimumSize(data_button_size)
# self.control_layout.addWidget(self.data_button)
# self.data_button.clicked.connect(load_image_clicked)

# @QtCore.Slot()
# def image_text_clicked():
#     self.image_paths, _ = QFileDialog.getOpenFileNames(self, QObject.tr('Load Test Image'))
#     # in case user did not load any image
#     if self.image_paths == []:
#         return
#     print(f'Loaded image(s) {self.image_paths}')

#     # load the image and scale the size
#     self.loaded_images_pt = []
#     self.cur_images_pt = []
#     self.display_images = []
#     self.loaded_image_names = []
#     # get the label of the image(s)
#     self.loaded_image_labels = []
#     for i in range(len(self.image_paths)):
#         self.loaded_images_pt.append(torch.from_numpy(np.asarray(Image.open(self.image_paths[i])))[:, :, None])
#         self.loaded_image_names.append(self.image_paths[i].split('/')[-1])
#         self.loaded_image_labels.append(int(self.image_paths[i].split('/')[-1].split('_')[1]))

#         # keep a copy to represent the current (rotated) version of the original images
#         self.cur_images_pt.append(self.loaded_images_pt[-1].clone())
#         # convert to QImage for display purpose
#         self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_images_pt[-1])
#         # resize the display QImage
#         self.display_images.append(self.cur_display_image.scaledToWidth(self.display_image_size))

#     # display the image
#     self.display_image()
#     self.data_button.setText(f'Click to load new image')
#     self.image_existed = True

#     # show the run button when data is loaded
#     if not self.run_button_existed:
#         # run button
#         self.run_button = QtWidgets.QPushButton('Analyze model')
#         self.run_button.setStyleSheet('font-size: 18px')
#         run_button_size = QtCore.QSize(500, 50)
#         self.run_button.setMinimumSize(run_button_size)
#         self.run_button_layout.addWidget(self.run_button)
#         self.run_button.clicked.connect(self.run_button_clicked)

#         self.run_button_existed = True


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
#     self.result_layout.addWidget(self.model_label, 0, 2)