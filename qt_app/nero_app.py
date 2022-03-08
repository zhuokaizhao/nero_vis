import enum
from gc import callbacks
import os
import sys
import glob
import time
import torch
import numpy as np
from PIL import Image, ImageDraw
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui  import QPixmap, QFont
# from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QWidget, QLabel, QRadioButton

import nero_transform
import nero_utilities
import nero_run_model

# globa configurations
pg.setConfigOptions(antialias=True, background='w')
# use pyside gpu acceleration if gpu detected
if torch.cuda.is_available():
    pg.setConfigOption('useCupy', True)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
else:
    pg.setConfigOption('useCupy', False)


class UI_MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # window size
        # self.setFixedSize(1920, 1080)
        self.resize(1920, 1080)
        # set window title
        self.setWindowTitle('Non-Equivariance Revealed on Orbits')
        # white background color
        self.setStyleSheet('background-color: white;')
        # general layout
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignCenter)
        # left, top, right, and bottom margins
        self.layout.setContentsMargins(50, 50, 50, 50)

        # individual laytout for different widgets
        # mode selections
        # self.mode = 'digit_recognition'
        self.mode = None
        # input data determines data mode
        self.data_mode = None
        # save the previous mode selection for layout swap
        self.previous_mode = None

        # initialize control panel on mode selection
        self.init_mode_control_layout()
        self.image_existed = False
        self.data_existed = False
        self.run_button_existed = False
        self.aggregate_result_existed = False
        self.single_result_existed = False

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


    # helper function on cleaning up things when switching modes
    def switch_mode_cleanup(self):
        if self.previous_mode:
            print(f'Cleaned {self.previous_mode} control layout')
            self.clear_layout(self.load_menu_layout)
        if self.run_button_existed:
            print(f'Cleaned previous run button')
            self.clear_layout(self.run_button_layout)
            self.run_button_existed = False
        if self.data_existed:
            print(f'Cleaned {self.previous_mode} aggregate_result_layout')
            self.clear_layout(self.aggregate_result_layout)
            self.data_existed = False
        if self.image_existed:
            print(f'Cleaned {self.previous_mode} single_result_layout')
            self.clear_layout(self.single_result_layout)
            self.image_existed = False


    def init_mode_control_layout(self):
        # three radio buttons that define the mode
        @QtCore.Slot()
        def digit_recognition_button_clicked():
            print('Digit recognition button clicked')
            self.mode = 'digit_recognition'
            self.radio_button_1.setChecked(True)

            if self.previous_mode != self.mode or not self.previous_mode:
                # clear previous mode's layout
                self.switch_mode_cleanup()

                self.init_load_layout()

                # analyze and display mnist image size
                self.image_size = 29
                self.display_image_size = 150
                # image (input data) modification mode
                self.rotation = True
                self.translation = False
                # rotation angles
                self.cur_rotation_angle = 0

                # preload model 1
                self.model_1_name = 'Simple model'
                self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
                self.model_1 = nero_run_model.load_model(self.mode, 'non_eqv', self.model_1_path)

                # preload model 2
                self.model_2_name = 'Simple model with DA'
                self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
                self.model_2 = nero_run_model.load_model(self.mode, 'aug_eqv', self.model_2_path)
                # self.model_2_name = 'Simple model with E2CNN'
                # self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
                # self.model_2 = nero_run_model.load_model('rot-eqv', self.model_2_path)

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
            if self.previous_mode != self.mode or not self.previous_mode:
                # clear previous mode's layout
                self.switch_mode_cleanup()

                self.init_load_layout()

                # analyze and display COCO image size
                self.image_size = 128
                self.display_image_size = 512
                # image (input data) modification mode
                self.rotation = False
                self.translation = False

                # predefined model paths
                self.model_1_name = 'Custom-trained FasterRCNN'
                self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_{self.jittering_level}-jittered', '*.pth'))[0]
                # pre-trained model does not need model path
                self.model_2_name = 'Pre-trained FasterRCNN'
                self.model_2_path = None
                # preload model
                self.model_1 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_1_path)
                self.model_2 = nero_run_model.load_model(self.mode, 'pre_trained', self.model_2_path)

                # different class names (original COCO classes, custom 5-class and the one that pretrained PyTorch model uses)
                self.original_coco_names_path = os.path.join(os.getcwd(), 'example_data', self.mode, 'coco.names')
                self.custom_coco_names_path = os.path.join(os.getcwd(), 'example_data', self.mode, 'custom.names')
                self.pytorch_coco_names_path = os.path.join(os.getcwd(), 'example_data', self.mode, 'pytorch_coco.names')

                # load these name files
                self.original_coco_names = nero_utilities.load_coco_classes_file(self.original_coco_names_path)
                self.custom_coco_names = nero_utilities.load_coco_classes_file(self.custom_coco_names_path)
                self.pytorch_coco_names = nero_utilities.load_coco_classes_file(self.pytorch_coco_names_path)

                print(f'Custom 5 classes: {self.custom_coco_names}')

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
            if self.previous_mode != self.mode or not self.previous_mode:
                # clear the previous method's layout
                if self.previous_mode:
                    self.clear_layout(self.load_menu_layout)
                if self.aggregate_result_existed:
                    self.clear_layout(self.aggregate_result_layout)
                if self.image_existed:
                    self.clear_layout(self.single_result_layout)

            self.init_load_layout()

            # display mnist image size
            self.display_image_size = 150
            # image (input data) modification mode
            self.rotation = False
            # rotation angles
            self.cur_rotation_angle = 0

            # predefined model paths
            self.model_1_name = 'Simple model'
            self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
            self.model_2_name = 'Simple model with DA'
            self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
            # preload model
            self.model_1 = nero_run_model.load_model(self.mode, 'non-eqv', self.model_1_path)
            self.model_2 = nero_run_model.load_model(self.mode, 'aug-eqv', self.model_2_path)

            # unique quantity of the result of current data
            self.all_quantities_1 = []
            self.all_quantities_2 = []

            # when doing highlighting
            # last_clicked is not None when it is either clicked or during manual rotation
            self.last_clicked = None
            self.cur_line = None

            self.previous_mode = self.mode


        # mode selection radio buttons
        self.mode_control_layout = QtWidgets.QGridLayout()
        self.mode_control_layout.setContentsMargins(50, 0, 0, 50)
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
        self.mode_label.setFixedSize(QtCore.QSize(150, 30))
        self.mode_label.setAlignment(QtCore.Qt.AlignLeft)
        self.mode_label.setWordWrap(True)
        self.mode_label.setTextFormat(QtGui.Qt.AutoText)
        self.mode_label.setPixmap(mode_pixmap)
        # add to the layout
        self.mode_control_layout.addWidget(self.mode_label, 0, 0)

        # radio_buttons_layout = QtWidgets.QGridLayout(self)
        self.radio_button_1 = QRadioButton('Digit recognition')
        self.radio_button_1.setFixedSize(QtCore.QSize(400, 50))
        self.radio_button_1.setStyleSheet('QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};')
        self.radio_button_1.pressed.connect(digit_recognition_button_clicked)
        self.mode_control_layout.addWidget(self.radio_button_1, 0, 1)
        # spacer item
        # self.mode_control_layout.addSpacing(30)

        self.radio_button_2 = QRadioButton('Object detection')
        self.radio_button_2.setFixedSize(QtCore.QSize(400, 50))
        self.radio_button_2.setStyleSheet('QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};')
        self.radio_button_2.pressed.connect(object_detection_button_clicked)
        self.mode_control_layout.addWidget(self.radio_button_2, 1, 1)
        # spacer item
        # self.mode_control_layout.addSpacing(30)

        self.radio_button_3 = QRadioButton('Particle Image Velocimetry (PIV)')
        self.radio_button_3.setFixedSize(QtCore.QSize(400, 50))
        self.radio_button_3.setStyleSheet('QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};')
        self.radio_button_3.pressed.connect(piv_button_clicked)
        self.mode_control_layout.addWidget(self.radio_button_3, 2, 1)

        # add to general layout
        self.layout.addLayout(self.mode_control_layout, 0, 0)

        # used for default state, if applicable
        if self.mode == 'digit_recognition':
            self.radio_button_1.setChecked(True)
            digit_recognition_button_clicked()
        elif self.mode == 'object_detection':
            self.radio_button_2.setChecked(True)
            object_detection_button_clicked()
        elif self.mode == 'piv':
            self.radio_button_3.setChecked(True)
            piv_button_clicked()


    def init_load_layout(self):

        # load aggregate dataset drop-down menu
        @QtCore.Slot()
        def aggregate_dataset_selection_changed(text):
            # filter out 0 selection signal
            if text == 'Input dataset':
                return

            # clear the single image selection
            self.image_menu.setCurrentIndex(0)
            # clear previous result layout
            if self.aggregate_result_existed:
                self.clear_layout(self.aggregate_result_layout)
                self.aggregate_result_existed = False
                self.data_existed = False
            if self.single_result_existed:
                self.clear_layout(self.single_result_layout)
                self.single_result_existed = False
                self.image_existed = False

            self.init_aggregate_result_layout()

            print('Loaded dataset:', text)
            self.data_mode = 'aggregate'
            self.dataset_index = int(text.split(' ')[-1])
            self.dataset_dir = self.aggregate_data_dirs[self.dataset_index]
            # load the image and scale the size
            # get all the image paths from the directory
            self.all_images_paths = glob.glob(os.path.join(self.dataset_dir, '*.png'))
            self.loaded_images_pt = []
            self.loaded_images_names = []
            self.loaded_images_labels = torch.zeros(len(self.all_images_paths), dtype=torch.int64)
            self.cur_images_pt = torch.zeros((len(self.all_images_paths), 29, 29, 1))

            for i, cur_image_path in enumerate(self.all_images_paths):
                self.loaded_images_pt.append(torch.from_numpy(np.asarray(Image.open(cur_image_path)))[:, :, None])
                self.loaded_images_names.append(cur_image_path.split('/')[-1])
                self.loaded_images_labels[i] = int(cur_image_path.split('/')[-1].split('_')[1])

                # keep a copy to represent the current (rotated) version of the original images
                # prepare image tensor for model purpose
                self.cur_images_pt[i] = nero_transform.prepare_mnist_image(self.loaded_images_pt[-1].clone())

            # self.cur_images_pt = torch.from_numpy(np.asarray(self.cur_images_pt))
            # check the data to be ready
            self.data_existed = True

            # show the run button when data is loaded
            if not self.run_button_existed:
                # run button
                # buttons layout for run model
                self.run_button_layout = QtWidgets.QGridLayout()
                self.layout.addLayout(self.run_button_layout, 2, 0, 1, 2)

                self.run_button = QtWidgets.QPushButton('Analyze model with aggregated dataset')
                self.run_button.setStyleSheet('font-size: 18px')
                run_button_size = QtCore.QSize(500, 50)
                self.run_button.setMinimumSize(run_button_size)
                self.run_button_layout.addWidget(self.run_button)
                self.run_button.clicked.connect(self.run_button_clicked)

                self.run_button_existed = True
            else:
                self.run_button.setText('Analyze model with aggregated dataset')

        # load single image drop-down menu
        @QtCore.Slot()
        def single_image_selection_changed(text):
            # filter out 0 selection signal
            if text == 'Input image':
                return

            # clear the aggregate dataset selection
            self.aggregate_image_menu.setCurrentIndex(0)
            # clear previous result layout
            if self.data_existed:
                self.clear_layout(self.aggregate_result_layout)
                self.aggregate_result_existed = False
                self.data_existed = False
            if self.image_existed:
                self.clear_layout(self.single_result_layout)
                self.single_result_existed = False
                self.image_existed = False

            self.data_mode = 'single'
            self.init_single_result_layout()

            print('Loaded image:', text)
            if self.mode == 'digit_recognition':
                self.image_index = int(text.split(' ')[-1])
                self.image_path = self.single_images_paths[self.image_index]
                self.loaded_image_label = int(self.image_path.split('/')[-1].split('_')[1])
                # load the image
                self.loaded_image_pt = torch.from_numpy(np.asarray(Image.open(self.image_path)))[:, :, None]
                self.loaded_image_name = self.image_path.split('/')[-1]
                # keep a copy to represent the current (rotated) version of the original images
                self.cur_image_pt = self.loaded_image_pt.clone()

            elif self.mode == 'object_detection':
                self.image_index = self.coco_classes.index(text.split(' ')[0])
                self.image_path = self.single_images_paths[self.image_index]
                self.label_path = self.image_path.replace('png', 'npy')
                self.loaded_image_label = np.load(self.label_path)
                # loaded image label is in original coco classes defined by original_coco_names
                # convert to custom names
                for i in range(len(self.loaded_image_label)):
                    self.loaded_image_label[i, -1] = self.custom_coco_names.index(self.original_coco_names[int(self.loaded_image_label[i, -1])])

                # the center of the bounding box is the center of cropped image
                # we know that we only have one object, x is column, y is row
                self.center_x = int((self.loaded_image_label[0, 0] + self.loaded_image_label[0, 2]) // 2)
                self.center_y = int((self.loaded_image_label[0, 1] + self.loaded_image_label[0, 3]) // 2)

                # load the image
                self.loaded_image_pt = torch.from_numpy(np.array(Image.open(self.image_path).convert('RGB'), dtype=np.uint8))
                self.loaded_image_name = self.image_path.split('/')[-1]

                # take the cropped part of the entire input image
                self.cur_image_pt = self.loaded_image_pt[self.center_y-128:self.center_y+128, self.center_x-128:self.center_x+128, :]

            # convert to QImage for display purpose
            self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt)
            # resize the display QImage
            self.cur_display_image = self.cur_display_image.scaledToWidth(self.display_image_size)

            # prepare MNIST image tensor for model purpose
            if self.mode == 'digit_recognition':
                self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)

            # display the image
            self.display_image()
            self.image_existed = True

            # show the run button when data is loaded
            if not self.run_button_existed:
                # run button
                # buttons layout for run model
                self.run_button_layout = QtWidgets.QGridLayout()
                self.layout.addLayout(self.run_button_layout, 3, 0, 1, 2)

                self.run_button = QtWidgets.QPushButton('Analyze model with single image')
                self.run_button.setStyleSheet('font-size: 18px')
                run_button_size = QtCore.QSize(500, 50)
                self.run_button.setMinimumSize(run_button_size)
                self.run_button_layout.addWidget(self.run_button)
                self.run_button.clicked.connect(self.run_button_clicked)

                self.run_button_existed = True
            else:
                self.run_button.setText('Analyze model with single image')

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
                    self.model_1 = nero_run_model.load_model(self.mode, 'non-eqv', self.model_1_path)
                elif text == 'Simple model with E2CNN':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'rot-eqv', self.model_1_path)
                elif text == 'Simple model with DA':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'aug-eqv', self.model_1_path)

            elif self.mode == 'object_detection':
                if text == 'Custom-trained FasterRCNN':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'non-eqv', self.model_1_path)
                elif text == 'Pre-trained FasterRCNN':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'rot-eqv', self.model_1_path)

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
                    self.model_2 = nero_run_model.load_model('non_eqv', self.model_2_path)
                elif text == 'Simple model with E2CNN':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model('rot_eqv', self.model_2_path)
                elif text == 'Simple model with DA':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model('aug_eqv', self.model_2_path)

            elif self.mode == 'object_detection':
                if text == 'Custom-trained FasterRCNN':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_{self.jittering_level}-jittered', '*.pth'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_2_path)
                    print('Model 2 path:', self.model_2_path)
                elif text == 'Pre-trained FasterRCNN':
                    # self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'pre_trained', '*.pt'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(self.mode, 'pre_trained', None)
                    print('Model 2 path: Downloaded from PyTorch')



        @QtCore.Slot()
        def jittering_menu_selection_changed(text):
            print('Jittering level:', text)
            if self.jittering_level != int(text.split('%')[0]):
                self.jittering_level = int(text.split('%')[0])
                # reload non-pretrained model
                if self.model_1_name == 'Custom-trained FasterRCNN':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_{self.jittering_level}-jittered', '*.pth'))[0]
                    self.model_1 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_1_path)
                    print('Model 1 reloaded')

                if self.model_2_name == 'Custom-trained FasterRCNN':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_{self.jittering_level}-jittered', '*.pth'))[0]
                    self.model_2 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_2_path)
                    print('Model 2 reloaded')

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

        # initialize layout for loading menus
        self.load_menu_layout = QtWidgets.QGridLayout()
        self.load_menu_layout.setContentsMargins(50, 0, 0, 50)

        # draw text
        model_pixmap = QPixmap(300, 30)
        model_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(model_pixmap)
        painter.setFont(QFont('Helvetica', 18))
        painter.drawText(0, 0, 300, 30, QtGui.Qt.AlignLeft, 'Data/Model Selection: ')
        painter.end()

        # create label to contain the texts
        self.model_label = QLabel(self)
        self.model_label.setFixedSize(QtCore.QSize(300, 30))
        self.model_label.setAlignment(QtCore.Qt.AlignLeft)
        self.model_label.setWordWrap(True)
        self.model_label.setTextFormat(QtGui.Qt.AutoText)
        self.model_label.setPixmap(model_pixmap)
        # add to the layout
        self.load_menu_layout.addWidget(self.model_label, 0, 2)

        # aggregate images loading drop down menu
        self.aggregate_image_menu = QtWidgets.QComboBox()
        self.aggregate_image_menu.setFixedSize(QtCore.QSize(300, 50))
        self.aggregate_image_menu.setStyleSheet('font-size: 18px')
        self.aggregate_image_menu.addItem('Input dataset')

        if self.mode == 'digit_recognition':
            self.aggregate_data_dirs = glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, f'aggregate*'))
            # load all images in the folder
            for i in range(len(self.aggregate_data_dirs)):
                # TODO: icons for aggregated datasets
                self.aggregate_image_menu.addItem(f'Dataset {i}')

            # set default to the prompt/description
            self.aggregate_image_menu.setCurrentIndex(0)

        elif self.mode == 'object_detection':
            self.aggregate_data_dirs = glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, f'aggregate*'))
            # load all images in the folder
            for i in range(len(self.aggregate_data_dirs)):
                # TODO: icons for aggregated datasets
                self.aggregate_image_menu.addItem(f'Dataset {i}')

            # set default to the prompt/description
            self.aggregate_image_menu.setCurrentIndex(0)

        # connect the drop down menu with actions
        self.aggregate_image_menu.currentTextChanged.connect(aggregate_dataset_selection_changed)
        self.aggregate_image_menu.setEditable(True)
        self.aggregate_image_menu.lineEdit().setReadOnly(True)
        self.aggregate_image_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.load_menu_layout.addWidget(self.aggregate_image_menu, 0, 3)

        # single image loading drop down menu
        self.image_menu = QtWidgets.QComboBox()
        self.image_menu.setFixedSize(QtCore.QSize(300, 50))
        self.image_menu.setStyleSheet('font-size: 18px')
        self.image_menu.addItem('Input image')

        if self.mode == 'digit_recognition':
            self.single_images_paths = []
            # add a image of each class
            for i in range(10):
                cur_image_path = glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, 'single', f'label_{i}*.png'))[0]
                self.single_images_paths.append(cur_image_path)
                self.image_menu.addItem(QtGui.QIcon(cur_image_path), f'Image {i}')

            self.image_menu.setCurrentIndex(0)

        elif self.mode == 'object_detection':
            self.single_images_paths = []
            self.data_config_paths = []
            self.coco_classes = ['car', 'bottle', 'cup', 'chair', 'book']
            # add a image of each class
            for i, cur_class in enumerate(self.coco_classes):
                cur_image_path = glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, 'single', f'{cur_class}*.png'))[0]
                self.single_images_paths.append(cur_image_path)
                self.image_menu.addItem(QtGui.QIcon(cur_image_path), f'{cur_class} image')

            self.image_menu.setCurrentIndex(0)

        # connect the drop down menu with actions
        self.image_menu.currentTextChanged.connect(single_image_selection_changed)
        self.image_menu.setEditable(True)
        self.image_menu.lineEdit().setReadOnly(True)
        self.image_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.load_menu_layout.addWidget(self.image_menu, 1, 3)

        # init flag to inidicate if an image has ever been loaded
        self.data_existed = False
        self.image_existed = False

        # add jittering level selection for the object detection mode
        if self.mode == 'object_detection':
            jittering_menu = QtWidgets.QComboBox()
            jittering_menu.setMinimumSize(QtCore.QSize(250, 50))
            jittering_menu.setStyleSheet('font-size: 18px')
            jittering_menu.setEditable(True)
            jittering_menu.lineEdit().setReadOnly(True)
            jittering_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)

            # add items
            # jittering_menu.addItem('Jittering level')
            for i in [0, 20, 40, 60, 80, 100]:
                jittering_menu.addItem(f'{i}%')

            # default selection is 0 percent jittering
            self.jittering_level = 0
            self.image_menu.setCurrentIndex(0)
            # connect the drop down menu with actions
            jittering_menu.currentTextChanged.connect(jittering_menu_selection_changed)

            self.load_menu_layout.addWidget(jittering_menu, 2, 3)

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
        # self.mode_control_layout.addSpacing(30)

        model_1_menu = QtWidgets.QComboBox()
        model_1_menu.setMinimumSize(QtCore.QSize(250, 50))
        model_1_menu.setStyleSheet('font-size: 18px')
        if self.mode == 'digit_recognition':
            model_1_menu.addItem(model_1_icon, 'Simple model')
            model_1_menu.addItem(model_1_icon, 'Simple model with E2CNN')
            model_1_menu.addItem(model_1_icon, 'Simple model with DA')
            model_1_menu.setCurrentText('Simple model')
        elif self.mode == 'object_detection':
            model_1_menu.addItem(model_1_icon, 'Custom-trained FasterRCNN')
            model_1_menu.addItem(model_1_icon, 'Pre-trained FasterRCNN')
            model_1_menu.setCurrentText('Custom-trained FasterRCNN')

        # connect the drop down menu with actions
        model_1_menu.currentTextChanged.connect(model_1_selection_changed)
        model_1_menu.setEditable(True)
        model_1_menu.lineEdit().setReadOnly(True)
        model_1_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        if self.mode == 'digit_recognition':
            self.load_menu_layout.addWidget(model_1_menu, 2, 3)
        elif self.mode == 'object_detection':
            self.load_menu_layout.addWidget(model_1_menu, 3, 3)

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
        # self.mode_control_layout.addSpacing(30)

        model_2_menu = QtWidgets.QComboBox()
        model_2_menu.setMinimumSize(QtCore.QSize(250, 50))
        model_2_menu.setStyleSheet('font-size: 18px')
        model_2_menu.setEditable(True)
        model_2_menu.lineEdit().setReadOnly(True)
        model_2_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        if self.mode == 'digit_recognition':
            model_2_menu.addItem(model_2_icon, 'Simple model')
            model_2_menu.addItem(model_2_icon, 'Simple model with E2CNN')
            model_2_menu.addItem(model_2_icon, 'Simple model with DA')
            # model_2_menu.setCurrentText('Simple model with E2CNN')
            model_2_menu.setCurrentText('Simple model with DA')
        elif self.mode == 'object_detection':
            model_2_menu.addItem(model_2_icon, 'Custom-trained FasterRCNN')
            model_2_menu.addItem(model_2_icon, 'Pre-trained FasterRCNN')
            model_2_menu.setCurrentText('Pre-trained FasterRCNN')

        # connect the drop down menu with actions
        model_2_menu.currentTextChanged.connect(model_2_selection_changed)
        if self.mode == 'digit_recognition':
            self.load_menu_layout.addWidget(model_2_menu, 3, 3)
        elif self.mode == 'object_detection':
            self.load_menu_layout.addWidget(model_2_menu, 4, 3)


        # add this layout to the general layout
        self.layout.addLayout(self.load_menu_layout, 0, 1)


    def init_aggregate_result_layout(self):
        # loaded images and model result layout
        self.aggregate_result_layout = QtWidgets.QGridLayout()
        # self.aggregate_result_layout.setContentsMargins(30, 50, 30, 50)

        # add to general layout
        self.layout.addLayout(self.aggregate_result_layout, 0, 2)

        # if model result ever existed
        self.aggregate_result_existed = False

        # batch size when running in aggregate mode
        if self.mode == 'digit_recognition':
            self.batch_size = 100
        elif self.mode == 'object_detection':
            self.batch_size = 64
        elif self.mode == 'piv':
            self.batch_size = 16


    def init_single_result_layout(self):
        # loaded images and model result layout
        self.single_result_layout = QtWidgets.QGridLayout()
        self.single_result_layout.setContentsMargins(30, 50, 30, 50)

        # add to general layout
        if self.data_mode == 'single':
            # take up two columns in UI layout
            self.layout.addLayout(self.single_result_layout, 1, 0, 1, 2)
        elif self.data_mode == 'aggregate':
            self.layout.addLayout(self.single_result_layout, 1, 2)

        # if model result ever existed
        self.single_result_existed = False


    # run button execution that could be used by all modes
    @QtCore.Slot()
    def run_button_clicked(self):
        if self.data_mode == 'aggregate':
            self.run_model_aggregated()
            self.aggregate_result_existed = True

        elif self.data_mode == 'single':
            if self.mode == 'digit_recognition':
                # run model once and display results
                self.run_model_once()
                # run model all and display results (Individual NERO plot)
                self.run_model_all()

            elif self.mode == 'object_detection':
                # run model all and display results (Individual NERO plot)
                self.run_model_all()

            self.single_result_existed = True


    # initialize digit selection control drop down menu
    def init_aggregate_polar_control(self):
        # aggregate digit selection drop-down menu
        @QtCore.Slot()
        def aggregate_digit_selection_changed(text):
            # update the current digit selection
            # if average or a specific digit
            if text.split(' ')[0] == 'Averaged':
                self.digit_selection = -1
            elif text.split(' ')[0] == 'Digit':
                self.digit_selection = int(text.split(' ')[-1])

            # display the plot
            self.display_mnist_aggregate_result()

        # run PCA on demand
        @QtCore.Slot()
        def run_dimension_reduction():
            # helper function on computing dimension reduction via PCA
            def run_pca(high_dim, target_dim):
                # get covariance matrix
                cov_matrix = np.cov(high_dim.T)
                # eigendecomposition
                values, vectors = np.linalg.eig(cov_matrix)
                values = np.real(values)
                vectors = np.real(vectors)

                # project onto principle components
                low_dim = np.zeros((len(high_dim), target_dim))
                for i in range(target_dim):
                    low_dim[:, i] = high_dim.dot(vectors.T[i])

                return low_dim

            # helper function for clicking inside the scatter plot
            def clicked(item, points):

                # clear previous visualization
                if self.last_clicked:
                    # previously selected point's visual cue
                    self.last_clicked.resetPen()
                    self.last_clicked.setBrush(self.old_brush)
                    # previous single image vis
                    # self.clear_layout(self.single_result_layout)
                    # self.single_result_existed = False
                    # self.image_existed = False

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

                # get the clicked scatter item's information
                self.image_index = int(item.opts['name'])
                # start single result view from here
                if not self.image_existed:
                    self.init_single_result_layout()
                self.image_path = self.all_images_paths[self.image_index]

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
                # prepare image tensor for model purpose
                self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)

                # display the image
                self.display_image()

                # run model once and display results (Detailed bar plot)
                self.run_model_once()

                # run model all and display results (Individual NERO plot)
                self.run_model_all()

                self.single_result_existed = True


            # run pca of all images on the selected digit
            # each image has tensor with length being the number of rotations
            cur_digit_indices = []
            if self.digit_selection == -1:
                cur_digit_indices = list(range(len(self.loaded_images_labels)))
            else:
                for i in range(len(self.loaded_images_labels)):
                    if self.digit_selection == self.loaded_images_labels[i]:
                        cur_digit_indices.append(i)

            all_high_dim_points_1 = np.zeros((len(cur_digit_indices), len(self.all_aggregate_angles)))
            all_high_dim_points_2 = np.zeros((len(cur_digit_indices), len(self.all_aggregate_angles)))
            for i, index in enumerate(cur_digit_indices):
                for j in range(len(self.all_aggregate_angles)):
                    # all_outputs has shape (num_rotations, num_samples, 10)
                    # all_high_dim_points_1[i, j] = self.all_outputs_1[j][index][self.loaded_images_labels[index]]
                    # all_high_dim_points_2[i, j] = self.all_outputs_2[j][index][self.loaded_images_labels[index]]
                    # all_avg_accuracy_per_digit has shape (num_rotations, 10)
                    all_high_dim_points_1[i, j] = int(self.all_outputs_1[j][index].argmax() == self.loaded_images_labels[index])
                    all_high_dim_points_2[i, j] = int(self.all_outputs_2[j][index].argmax() == self.loaded_images_labels[index])

            # run dimension reduction algorithm
            low_dim_1 = run_pca(all_high_dim_points_1, target_dim=2)
            low_dim_2 = run_pca(all_high_dim_points_2, target_dim=2)

            # scatter plot on low-dim points
            low_dim_scatter_view = pg.GraphicsLayoutWidget()
            low_dim_scatter_view.setBackground('white')
            low_dim_scatter_view.setFixedSize(500, 500)
            self.low_dim_scatter_plot = low_dim_scatter_view.addPlot()

            # Set pxMode=False to allow spots to transform with the view

            for i, index in enumerate(cur_digit_indices):
                # all the points to be plotted
                # add individual items for getting the item's name later when clicking
                self.low_dim_scatter_item = pg.ScatterPlotItem(pxMode=False)
                low_dim_point_1 = [{'pos': (low_dim_1[i, 0], low_dim_1[i, 1]),
                                    'size': 0.05,
                                    'pen': {'color': 'w', 'width': 0.1},
                                    'brush': (0, 0, 255, 150)}]

                low_dim_point_2 = [{'pos': (low_dim_2[i, 0], low_dim_2[i, 1]),
                                    'size': 0.05,
                                    'pen': {'color': 'w', 'width': 0.1},
                                    'brush': (0, 255, 0, 150)}]

                # add points to the item
                self.low_dim_scatter_item.addPoints(low_dim_point_1, name=str(index))
                self.low_dim_scatter_item.addPoints(low_dim_point_2, name=str(index))

                # add points to the plot
                self.low_dim_scatter_plot.addItem(self.low_dim_scatter_item)

                # connect click events on scatter items
                self.low_dim_scatter_item.sigClicked.connect(clicked)

            self.aggregate_result_layout.addWidget(low_dim_scatter_view, 0, 0)



        # drop down menu on choosing the digit
        # self.digit_selection_layout = QtWidgets.QVBoxLayout()
        self.digit_selection_menu = QtWidgets.QComboBox()
        self.digit_selection_menu.setMinimumSize(QtCore.QSize(250, 50))
        self.digit_selection_menu.setStyleSheet('font-size: 18px')
        self.digit_selection_menu.addItem(f'Averaged over all digits')
        # add all digits as items
        for i in range(10):
            self.digit_selection_menu.addItem(f'Digit {i}')

        # set default to digit -1, which means the average one
        self.digit_selection = -1
        self.digit_selection_menu.setCurrentIndex(0)

        # connect the drop down menu with actions
        self.digit_selection_menu.currentTextChanged.connect(aggregate_digit_selection_changed)
        self.digit_selection_menu.setEditable(True)
        self.digit_selection_menu.lineEdit().setReadOnly(True)
        self.digit_selection_menu.lineEdit().setAlignment(QtCore.Qt.AlignRight)

        self.aggregate_result_layout.addWidget(self.digit_selection_menu, 0, 2)

        # push button on running PCA
        self.pca_button = QtWidgets.QPushButton('See Overview')
        self.pca_button.setStyleSheet('font-size: 18px')
        self.pca_button.setMinimumSize(QtCore.QSize(250, 50))
        self.pca_button.clicked.connect(run_dimension_reduction)
        self.aggregate_result_layout.addWidget(self.pca_button, 1, 2)


    # run model on the aggregate dataset
    def run_model_aggregated(self):
        if self.mode == 'digit_recognition':
            # all the rotation angles applied to the aggregated dataset
            self.all_aggregate_angles = list(range(0, 365, 90))
            # average accuracies over all digits under all rotations, has shape (num_rotations, 1)
            self.all_avg_accuracy_1 = np.zeros(len(self.all_aggregate_angles))
            self.all_avg_accuracy_2 = np.zeros(len(self.all_aggregate_angles))
            # average accuracies of each digit under all rotations, has shape (num_rotations, 10)
            self.all_avg_accuracy_per_digit_1 = np.zeros((len(self.all_aggregate_angles), 10))
            self.all_avg_accuracy_per_digit_2 = np.zeros((len(self.all_aggregate_angles), 10))
            # output of each class's probablity of all samples, has shape (num_rotations, num_samples, 10)
            self.all_outputs_1 = np.zeros((len(self.all_aggregate_angles), len(self.cur_images_pt), 10))
            self.all_outputs_2 = np.zeros((len(self.all_aggregate_angles), len(self.cur_images_pt), 10))

            # for all the loaded images
            # for i, self.cur_rotation_angle in enumerate(range(0, 365, 5)):
            for i, self.cur_rotation_angle in enumerate(self.all_aggregate_angles):
                print(f'\nAggregate mode: Rotated {self.cur_rotation_angle} degrees')
                # self.all_angles.append(self.cur_rotation_angle)

                avg_accuracy_1, avg_accuracy_per_digit_1, output_1 = nero_run_model.run_mnist_once(self.model_1,
                                                                                        self.cur_images_pt,
                                                                                        self.loaded_images_labels,
                                                                                        batch_size=self.batch_size,
                                                                                        rotate_angle=self.cur_rotation_angle)

                avg_accuracy_2, avg_accuracy_per_digit_2, output_2 = nero_run_model.run_mnist_once(self.model_2,
                                                                                        self.cur_images_pt,
                                                                                        self.loaded_images_labels,
                                                                                        batch_size=self.batch_size,
                                                                                        rotate_angle=self.cur_rotation_angle)

                # append to results
                self.all_avg_accuracy_1[i] = avg_accuracy_1
                self.all_avg_accuracy_2[i] = avg_accuracy_2
                self.all_avg_accuracy_per_digit_1[i] = avg_accuracy_per_digit_1
                self.all_avg_accuracy_per_digit_2[i] = avg_accuracy_per_digit_2
                self.all_outputs_1[i] = output_1
                self.all_outputs_2[i] = output_2

            # initialize digit selection control
            self.init_aggregate_polar_control()

            # display the result
            self.display_mnist_aggregate_result()


    # run model on a single test sample with no transfomations
    def run_model_once(self):
        if self.mode == 'digit_recognition':

            self.output_1 = nero_run_model.run_mnist_once(self.model_1, self.cur_image_pt)
            self.output_2 = nero_run_model.run_mnist_once(self.model_2, self.cur_image_pt)

            # display result
            self.display_mnist_single_result(type='bar', boundary_width=3)

        # elif self.mode == 'object_detection':

        #     self.output_1 = nero_run_model.run_coco_once(self.model_2_name,
        #                                                     self.model_2,
        #                                                     self.cropped_image_pt,
        #                                                     self.original_coco_names,
        #                                                     self.custom_coco_names,
        #                                                     self.pytorch_coco_names,
        #                                                     test_label=self.cur_image_label)
            # self.output_2 = nero_run_model.run_coco_once(self.model_2, self.cur_image_pt)

            # display result
            # self.display_mnist_single_result(mode=self.data_mode, type='bar', boundary_width=3)




    # run model on all the available transformations on a single sample
    def run_model_all(self):

        # quantity that is displayed in the individual NERO plot
        self.all_quantities_1 = []
        self.all_quantities_2 = []

        if self.mode == 'digit_recognition':
            self.all_angles = []
            # run all rotation test with 5 degree increment
            for self.cur_rotation_angle in range(0, 365, 5):
                # print(f'\nRotated {self.cur_rotation_angle} degrees')
                self.all_angles.append(self.cur_rotation_angle)
                # rotate the image tensor
                self.cur_image_pt = nero_transform.rotate_mnist_image(self.loaded_image_pt, self.cur_rotation_angle)
                # convert image tensor to qt image and resize for display
                self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt).scaledToWidth(self.display_image_size)
                # prepare image tensor for model purpose
                self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)
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

            # display result
            self.display_mnist_single_result(type='polar', boundary_width=3)

        elif self.mode == 'object_detection':

            # all the x and y translations
            # x translates on columns, y translates on rows
            x_translation = list(range(-self.image_size//2, self.image_size//2, 8))
            y_translation = list(range(-self.image_size//2, self.image_size//2, 8))
            self.all_translations = []

            for y_tran in y_translation:
                for x_tran in x_translation:
                    self.all_translations.append((x_tran, y_tran))
                    # x_tran = -64
                    # y_tran = -64
                    # modify the underlying image tensor accordingly
                    # take the cropped part of the entire input image
                    cur_center_x = self.center_x - x_tran
                    cur_center_y = self.center_y - y_tran
                    x_min = cur_center_x - 64
                    x_max = cur_center_x + 64
                    y_min = cur_center_y - 64
                    y_max = cur_center_y + 64
                    # model takes image between [0, 1]
                    self.cropped_image_pt = self.loaded_image_pt[y_min:y_max, x_min:x_max, :] / 255

                    # modify the label accordingly
                    # helper function that computes labels for cut out images
                    def compute_label(cur_bounding_box, x_min, y_min, image_size):
                        # convert key object bounding box to be based on extracted image
                        x_min_center_bb = cur_bounding_box[0] - x_min
                        y_min_center_bb = cur_bounding_box[1] - y_min
                        x_max_center_bb = cur_bounding_box[2] - x_min
                        y_max_center_bb = cur_bounding_box[3] - y_min

                        # compute the center of the object in the extracted image
                        object_center_x = (x_min_center_bb + x_max_center_bb) / 2
                        object_center_y = (y_min_center_bb + y_max_center_bb) / 2

                        # compute the width and height of the real bounding box of this object
                        original_bb_width = cur_bounding_box[2] - cur_bounding_box[0]
                        original_bb_height = cur_bounding_box[3] - cur_bounding_box[1]

                        # compute the range of the bounding box, do the clamping if go out of extracted image
                        bb_min_x = max(0, object_center_x - original_bb_width/2)
                        bb_max_x = min(image_size[1]-1, object_center_x + original_bb_width/2)
                        bb_min_y = max(0, object_center_y - original_bb_height/2)
                        bb_max_y = min(image_size[0]-1, object_center_y + original_bb_height/2)

                        return bb_min_x, bb_min_y, bb_max_x, bb_max_y

                    self.cur_image_label = np.zeros((len(self.loaded_image_label), 6))
                    for i in range(len(self.cur_image_label)):
                        self.cur_image_label[i, 0] = i
                        self.cur_image_label[i, 1] = self.loaded_image_label[i, 4]
                        self.cur_image_label[i, 2:] = compute_label(self.loaded_image_label[i, :4], x_min, y_min, (128, 128))

                    # sanity check on if image/label are correct
                    sanity_check = False
                    if sanity_check:
                        sanity_path = f'/home/zhuokai/Desktop/UChicago/Research/nero_vis/qt_app/example_data/object_detection/sanity_{x_tran}_{y_tran}.png'
                        sanity_image = Image.fromarray(self.cropped_image_pt.numpy())
                        temp = ImageDraw.Draw(sanity_image)
                        temp.rectangle([(self.cur_image_label[0, 2], self.cur_image_label[0, 3]), (self.cur_image_label[0, 4], self.cur_image_label[0, 5])], outline='yellow')
                        sanity_image.save(sanity_path)

                    # re-display image for each rectangle drawn every 8 steps
                    if (x_tran)%2 == 0 and (y_tran)%2 == 0:
                        self.display_image()

                        display_rect_width = self.display_image_size//2
                        display_rect_height = self.display_image_size//2
                        # displayed image has center at the center of the display
                        # since displayed image enlarges the actual image by 2
                        # since the translation measures on the movement of object instead of the point of view, the sign is reversed
                        rect_center_x = self.display_image_size//2 - x_tran*2
                        rect_center_y = self.display_image_size//2 - y_tran*2

                        # draw rectangle on the displayed image to indicate scanning process
                        painter = QtGui.QPainter(self.image_pixmap)
                        # draw the rectangle
                        self.draw_rectangle(painter, rect_center_x, rect_center_y, display_rect_width, display_rect_height, 'red')
                        painter.end()

                        # draw ground truth label on the display image
                        # draw rectangle on the displayed image to indicate scanning process
                        painter = QtGui.QPainter(self.image_pixmap)
                        # draw the ground truth label
                        gt_display_center_x = (self.cur_image_label[0, 2] + self.cur_image_label[0, 4]) // 2 * 2 + (rect_center_x - display_rect_width//2)
                        gt_display_center_y = (self.cur_image_label[0, 3] + self.cur_image_label[0, 5]) // 2 * 2 + (rect_center_y - display_rect_height//2)
                        gt_display_rect_width = (self.cur_image_label[0, 4] - self.cur_image_label[0, 2]) * 2
                        gt_display_rect_height = (self.cur_image_label[0, 5] - self.cur_image_label[0, 3]) * 2
                        self.draw_rectangle(painter, gt_display_center_x, gt_display_center_y, gt_display_rect_width, gt_display_rect_height, 'yellow')
                        painter.end()

                        # update pixmap with the label
                        self.image_label.setPixmap(self.image_pixmap)

                        # force repaint
                        self.image_label.repaint()

                    # run the model
                    # update the model output
                    self.output_1 = nero_run_model.run_coco_once(self.model_1_name,
                                                                    self.model_1,
                                                                    self.cropped_image_pt,
                                                                    self.custom_coco_names,
                                                                    self.pytorch_coco_names,
                                                                    test_label=self.cur_image_label)

                    self.output_2 = nero_run_model.run_coco_once(self.model_2_name,
                                                                    self.model_2,
                                                                    self.cropped_image_pt,
                                                                    self.custom_coco_names,
                                                                    self.pytorch_coco_names,
                                                                    test_label=self.cur_image_label)

                    # plotting the quantity regarding the correct label
                    if self.output_1[0][0]:
                        quantity_1 = self.output_1[0][0][0]
                    else:
                        quantity_1 = np.zeros(7)

                    if self.output_2[0][0]:
                        quantity_2 = self.output_2[0][0][0]
                    else:
                        quantity_2 = np.zeros(7)

                    self.all_quantities_1.append(quantity_1)
                    self.all_quantities_2.append(quantity_2)


            # display the individual NERO plot
            num_x_translations = len(x_translation)
            num_y_translations = len(y_translation)
            self.all_quantities_1 = np.array(self.all_quantities_1).reshape((num_y_translations, num_x_translations, 7))
            self.all_quantities_2 = np.array(self.all_quantities_2).reshape((num_y_translations, num_x_translations, 7))

            # display the individual NERO plot
            self.display_coco_single_result(type='heatmap', boundary_width=3)


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

        # pixel mouse over for object detection mode
        if self.mode == 'object_detection':
            def getPixel(event):
                x = event.pos().x()
                y = event.pos().y()
                # c = self.image_label.pixel(x, y)  # color code (integer): 3235912
                # # depending on what kind of value you like (arbitary examples)
                # c_qobj = QtGui.QColor(c)  # color object
                # c_rgb = QtGui.QColor(c).getRgb()  # 8bit RGBA: (255, 23, 0, 255)
                # c_rgbf = QtGui.QColor(c).getRgbf()  # RGBA float: (1.0, 0.3123, 0.0, 1.0)

                print(x, y)
                return x, y

            self.image_label.mousePressEvent = getPixel

        # name of the image
        self.name_label = QLabel(self.loaded_image_name)
        self.name_label.setAlignment(QtCore.Qt.AlignCenter)

        # add this image to the layout
        if self.data_mode == 'single':
            self.single_result_layout.addWidget(self.image_label, 0, 0)
            self.single_result_layout.addWidget(self.name_label, 1, 0)
        elif self.data_mode == 'aggregate':
            self.single_result_layout.addWidget(self.image_label, 0, 2)
            self.single_result_layout.addWidget(self.name_label, 1, 2)


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


    # draw a rectangle
    def draw_rectangle(self, painter, center_x, center_y, width, height, color, boundary_width=3):
        pen = QtGui.QPen()
        pen.setWidth(boundary_width)
        pen.setColor(QtGui.QColor(color))
        painter.setPen(pen)
        # left, top, width, height for QRect
        rectangle = QtCore.QRect(center_x-width//2, center_y-height//2, width, height)
        painter.drawRect(rectangle)


    # draw a polar plot
    def draw_polar(self, plot):
        plot.setXRange(-1, 1)
        plot.setYRange(-1, 1)
        plot.setAspectLocked()

        plot.hideAxis('bottom')
        plot.hideAxis('left')

        # Add polar grid lines
        plot.addLine(x=0, pen=pg.mkPen('black', width=2))
        plot.addLine(y=0, pen=pg.mkPen('black', width=2))
        for r in np.arange(0, 1.2, 0.2):
            circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
            circle.setPen(pg.mkPen('black', width=2))
            plot.addItem(circle)

        return plot


    # helper function on drawing the little circle on polar plot
    def draw_circle_on_polar(self):
        r = 0.06
        # transform to x and y coordinate
        cur_quantity_1_x = self.output_1[self.loaded_image_label] * np.cos(self.cur_rotation_angle/180*np.pi)
        cur_quantity_1_y = self.output_1[self.loaded_image_label] * np.sin(self.cur_rotation_angle/180*np.pi)
        # plot a circle item
        self.circle_1 = pg.QtGui.QGraphicsEllipseItem(cur_quantity_1_x-r/2, cur_quantity_1_y-r/2, r, r)
        self.circle_1.setPen(pg.mkPen('blue', width=10))
        self.polar_plot.addItem(self.circle_1)

        # transform to x and y coordinate
        cur_quantity_2_x = self.output_2[self.loaded_image_label] * np.cos(self.cur_rotation_angle/180*np.pi)
        cur_quantity_2_y = self.output_2[self.loaded_image_label] * np.sin(self.cur_rotation_angle/180*np.pi)
        # plot a circle item
        self.circle_2 = pg.QtGui.QGraphicsEllipseItem(cur_quantity_2_x-r/2, cur_quantity_2_y-r/2, r, r)
        self.circle_2.setPen(pg.mkPen('green', width=10))
        self.polar_plot.addItem(self.circle_2)


    # draw heatmap
    def draw_heatmaps(self):
        # helper function on drawing individual heatmap
        def draw_individual_heatmap(data, range=(0, 1)):
            # actuall heatmap
            heatmap = pg.ImageItem()
            heatmap.setOpts(axisOrder='row-major')
            heatmap.setImage(data)

            # create view box to contain the heatmap
            view_box = pg.ViewBox()
            view_box.setAspectLocked(lock=True)
            view_box.addItem(heatmap)

            plot = pg.PlotItem(viewBox=view_box)
            # plot.vb.setLimits(xMin=-64, xMax=63, yMin=-64, yMax=63)
            plot.getAxis('bottom').setLabel('Translation in x')
            plot.getAxis('left').setLabel('Translation in y')

            # create colorbar
            color_map = pg.colormap.get('CET-L9')
            color_bar = pg.ColorBarItem(values= range, colorMap=color_map)
            color_bar.setImageItem(heatmap, insert_in=plot)

            return plot

        # check if the data is in shape (128, 128)
        if self.cur_plot_quantity_1.shape != (128, 128):
            # repeat in row
            temp = np.repeat(self.cur_plot_quantity_1, 128//self.cur_plot_quantity_1.shape[1], axis=0)
            # repeat in column
            data_1 = np.repeat(temp, 128//self.cur_plot_quantity_1.shape[0], axis=1)
        else:
            data_1 = self.cur_plot_quantity_1

        if self.cur_plot_quantity_2.shape != (128, 128):
            # repeat in row
            temp = np.repeat(self.cur_plot_quantity_2, 128//self.cur_plot_quantity_2.shape[1], axis=0)
            # repeat in column
            data_2 = np.repeat(temp, 128//self.cur_plot_quantity_2.shape[0], axis=1)
        else:
            data_2 = self.cur_plot_quantity_2

        # heatmap view
        self.heatmap_view_1 = pg.GraphicsLayoutWidget(title='Model 1')
        self.heatmap_view_1.setFixedSize(600, 600)
        self.heatmap_view_2 = pg.GraphicsLayoutWidget(title='Model 2')
        self.heatmap_view_2.setFixedSize(600, 600)
        heatmap_plot_1 = draw_individual_heatmap(data_1)
        heatmap_plot_2 = draw_individual_heatmap(data_2)
        # disable moving around
        heatmap_plot_1.setMouseEnabled(x=False, y=False)
        heatmap_plot_2.setMouseEnabled(x=False, y=False)

        # add to view
        self.heatmap_view_1.addItem(heatmap_plot_1)
        self.heatmap_view_2.addItem(heatmap_plot_2)

        # add to general layout
        self.single_result_layout.addWidget(self.heatmap_view_1, 0, 3)
        self.single_result_layout.addWidget(self.heatmap_view_2, 0, 4)


    # display MNIST aggregated results
    def display_mnist_aggregate_result(self):

        # initialize view and plot
        polar_view = pg.GraphicsLayoutWidget()
        polar_view.setBackground('white')
        polar_view.setFixedSize(500, 500)
        self.aggregate_polar_plot = polar_view.addPlot()
        self.aggregate_polar_plot = self.draw_polar(self.aggregate_polar_plot)

        # Set pxMode=False to allow spots to transform with the view
        # all the points to be plotted
        self.aggregate_scatter_items = pg.ScatterPlotItem(pxMode=False)
        all_points_1 = []
        all_points_2 = []
        all_x_1 = []
        all_y_1 = []
        all_x_2 = []
        all_y_2 = []
        # plot selected digit's average accuracy across all rotations
        for i in range(len(self.all_aggregate_angles)):
            radian = self.all_aggregate_angles[i] / 180 * np.pi
            # model 1 accuracy
            if self.digit_selection == -1:
                cur_quantity_1 = self.all_avg_accuracy_1[i]
            else:
                cur_quantity_1 = self.all_avg_accuracy_per_digit_1[i][self.digit_selection]
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
            if self.digit_selection == -1:
                cur_quantity_2 = self.all_avg_accuracy_2[i]
            else:
                cur_quantity_2 = self.all_avg_accuracy_per_digit_2[i][self.digit_selection]
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
        line_1 = self.aggregate_polar_plot.plot(all_x_1, all_y_1, pen = QtGui.QPen(QtGui.Qt.blue, 0.03))
        line_2 = self.aggregate_polar_plot.plot(all_x_2, all_y_2, pen = QtGui.QPen(QtGui.Qt.green, 0.03))

        # add points to the item
        self.aggregate_scatter_items.addPoints(all_points_1)
        self.aggregate_scatter_items.addPoints(all_points_2)

        # add points to the plot
        self.aggregate_polar_plot.addItem(self.aggregate_scatter_items)

        # fix zoom level
        # self.polar_plot.vb.scaleBy((0.5, 0.5))
        self.aggregate_polar_plot.setMouseEnabled(x=False, y=False)

        # add the plot view to the layout
        self.aggregate_result_layout.addWidget(polar_view, 0, 1)


    # display MNIST single results
    def display_mnist_single_result(self, type, boundary_width):

        # aggregate mode does not draw arrow
        if self.data_mode == 'single':
            # draw arrow
            # add a new label for result if no result has existed
            if not self.single_result_existed:
                self.arrow_label = QLabel(self)
                self.arrow_label.setAlignment(QtCore.Qt.AlignCenter)
                self.arrow_label.setWordWrap(True)
                self.arrow_label.setTextFormat(QtGui.Qt.AutoText)
                self.single_result_existed = True

            arrow_pixmap = QPixmap(100, 50)
            arrow_pixmap.fill(QtCore.Qt.white)
            painter = QtGui.QPainter(arrow_pixmap)
            # set pen (used to draw outlines of shapes) and brush (draw the background of a shape)
            pen = QtGui.QPen()
            # draw arrow to indicate feeding
            self.draw_arrow(painter, pen, 100, 50, boundary_width)

            # add to the label and layout
            self.arrow_label.setPixmap(arrow_pixmap)
            self.single_result_layout.addWidget(self.arrow_label, 0, 1)
            painter.end()

        # draw result using bar plot
        if type == 'bar':
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
            self.bar_plot.setFixedSize(600, 500)
            self.bar_plot.getAxis('bottom').setLabel('Digit')
            self.bar_plot.getAxis('left').setLabel('Confidence')
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
            if self.data_mode == 'single':
                self.single_result_layout.addWidget(self.bar_plot, 0, 2)
            elif self.data_mode == 'aggregate':
                self.single_result_layout.addWidget(self.bar_plot, 0, 0)

        elif type == 'polar':

            # helper function for clicking inside polar plot
            def clicked(item, points):
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
                # prepare image tensor for model purpose
                self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)
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
            polar_view.setFixedSize(500, 500)
            self.polar_plot = polar_view.addPlot()
            self.polar_plot = self.draw_polar(self.polar_plot)

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
                    # prepare image tensor for model purpose
                    self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)
                    # update the pixmap and label
                    self.image_pixmap = QPixmap(self.cur_display_image)
                    self.image_label.setPixmap(self.image_pixmap)

                    # update the model output
                    if self.result_existed:
                        self.run_model_once()

                    # remove old line
                    if self.cur_line:
                        self.polar_plot.removeItem(self.cur_line)
                        self.polar_plot.removeItem(self.circle_1)
                        self.polar_plot.removeItem(self.circle_2)

                    # draw a line that represents current angle of rotation
                    cur_x = 1 * np.cos(self.cur_rotation_angle/180*np.pi)
                    cur_y = 1 * np.sin(self.cur_rotation_angle/180*np.pi)
                    line_x = [0, cur_x]
                    line_y = [0, cur_y]
                    self.cur_line = self.polar_plot.plot(line_x, line_y, pen = QtGui.QPen(QtGui.Qt.red, 0.01))

                    # display current results on the line
                    self.draw_circle_on_polar()

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
                    # prepare image tensor for model purpose
                    self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)
                    # update the pixmap and label
                    self.image_pixmap = QPixmap(self.cur_display_image)
                    self.image_label.setPixmap(self.image_pixmap)

                    # update the model output
                    if self.result_existed:
                        self.run_model_once()

                    # remove old line and circle
                    if self.cur_line:
                        self.polar_plot.removeItem(self.cur_line)
                        self.polar_plot.removeItem(self.circle_1)
                        self.polar_plot.removeItem(self.circle_2)

                    # draw a line that represents current angle of rotation
                    cur_x = 1 * np.cos(self.cur_rotation_angle/180*np.pi)
                    cur_y = 1 * np.sin(self.cur_rotation_angle/180*np.pi)
                    line_x = [0, cur_x]
                    line_y = [0, cur_y]
                    self.cur_line = self.polar_plot.plot(line_x, line_y, pen = QtGui.QPen(QtGui.Qt.red, 0.01))

                    # display current results on the line
                    self.draw_circle_on_polar()


            self.polar_clicked = False
            self.polar_plot.scene().sigMouseClicked.connect(polar_mouse_clicked)
            self.polar_plot.scene().sigMouseMoved.connect(polar_mouse_moved)

            # fix zoom level
            # self.polar_plot.vb.scaleBy((0.5, 0.5))
            self.polar_plot.setMouseEnabled(x=False, y=False)

            # add the plot view to the layout
            self.single_result_layout.addWidget(polar_view, 0, 3)

        else:
            raise Exception('Unsupported display mode')


    # display COCO single results
    def display_coco_single_result(self, type, boundary_width):
        # aggregate mode does not draw arrow
        if self.data_mode == 'single':
            # draw arrow
            # add a new label for result if no result has existed
            if not self.single_result_existed:
                self.arrow_label = QLabel(self)
                self.arrow_label.setAlignment(QtCore.Qt.AlignCenter)
                self.arrow_label.setWordWrap(True)
                self.arrow_label.setTextFormat(QtGui.Qt.AutoText)
                self.single_result_existed = True

            arrow_pixmap = QPixmap(100, 50)
            arrow_pixmap.fill(QtCore.Qt.white)
            painter = QtGui.QPainter(arrow_pixmap)
            # set pen (used to draw outlines of shapes) and brush (draw the background of a shape)
            pen = QtGui.QPen()
            # draw arrow to indicate feeding
            self.draw_arrow(painter, pen, 100, 50, boundary_width)

            # add to the label and layout
            self.arrow_label.setPixmap(arrow_pixmap)
            self.single_result_layout.addWidget(self.arrow_label, 0, 1)
            painter.end()

        # draw result using heatmaps
        if type == 'heatmap':
            @QtCore.Slot()
            def heatmap_quantity_changed(text):
                print('Plotting:', text, 'on heatmap')
                self.quantity_name = text

                if text == 'Confidence*IOU':
                    self.cur_plot_quantity_1 = self.all_quantities_1[:, :, 4] * self.all_quantities_1[:, :, 6]
                    self.cur_plot_quantity_2 = self.all_quantities_2[:, :, 4] * self.all_quantities_2[:, :, 6]
                elif text == 'Confidence':
                    self.cur_plot_quantity_1 = self.all_quantities_1[:, :, 4]
                    self.cur_plot_quantity_2 = self.all_quantities_2[:, :, 4]
                elif text == 'IOU':
                    self.cur_plot_quantity_1 = self.all_quantities_1[:, :, 6]
                    self.cur_plot_quantity_2 = self.all_quantities_2[:, :, 6]

                # re-display the heatmap
                self.draw_heatmaps()

            # drop down menu on selection which quantity to plot
            quantity_menu = QtWidgets.QComboBox()
            quantity_menu.setMinimumSize(QtCore.QSize(250, 50))
            quantity_menu.setStyleSheet('font-size: 18px')
            quantity_menu.setEditable(True)
            quantity_menu.lineEdit().setReadOnly(True)
            quantity_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)

            quantity_menu.addItem('Confidence*IOU')
            quantity_menu.addItem('Confidence')
            quantity_menu.addItem('IOU')
            # self.quantity_menu.setCurrentIndex(0)
            quantity_menu.setCurrentText('Confidence*IOU')

            # connect the drop down menu with actions
            quantity_menu.currentTextChanged.connect(heatmap_quantity_changed)
            self.single_result_layout.addWidget(quantity_menu, 0, 2)

            # define default plotting quantity
            self.cur_plot_quantity_1 = self.all_quantities_1[:, :, 4] * self.all_quantities_1[:, :, 6]
            self.cur_plot_quantity_2 = self.all_quantities_2[:, :, 4] * self.all_quantities_2[:, :, 6]

            # draw the heatmap
            self.draw_heatmaps()


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
            # prepare image tensor for model purpose
            self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)
            # update the pixmap and label
            self.image_pixmap = QPixmap(self.cur_display_image)
            self.image_label.setPixmap(self.image_pixmap)

            # update the model output
            if self.single_result_existed:

                self.run_model_once()

                # remove old line
                if self.cur_line:
                    self.polar_plot.removeItem(self.cur_line)
                    self.polar_plot.removeItem(self.circle_1)
                    self.polar_plot.removeItem(self.circle_2)

                # draw a line that represents current angle of rotation
                cur_x = 1 * np.cos(self.cur_rotation_angle/180*np.pi)
                cur_y = 1 * np.sin(self.cur_rotation_angle/180*np.pi)
                line_x = [0, cur_x]
                line_y = [0, cur_y]
                self.cur_line = self.polar_plot.plot(line_x, line_y, pen = QtGui.QPen(QtGui.Qt.red, 0.01))

                # display current results on the line
                self.draw_circle_on_polar()

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
    # widget.resize(1920, 1080)
    widget.show()

    sys.exit(app.exec())








# def init_title_layout(self):
#     # title
#     self.title_layout = QtWidgets.QVBoxLayout()
#     self.title_layout.setContentsMargins(0, 0, 0, 20)
#     # title of the application
#     self.title = QLabel('Non-Equivariance Revealed on Orbits',
#                         alignment=QtCore.Qt.AlignCenter)
#     self.title.setFont(QFont('Helvetica', 24))
#     self.title_layout.addWidget(self.title)
#     self.title_layout.setContentsMargins(0, 0, 0, 50)

#     # add to general layout
#     self.layout.addLayout(self.title_layout, 0, 0)

# # load data button
# self.data_button = QtWidgets.QPushButton('Load Test Image')
# self.data_button.setStyleSheet('font-size: 18px')
# data_button_size = QtCore.QSize(500, 50)
# self.data_button.setMinimumSize(data_button_size)
# self.mode_control_layout.addWidget(self.data_button)
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