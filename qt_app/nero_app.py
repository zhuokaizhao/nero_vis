import enum
from gc import callbacks
from multiprocessing import reduction
import os
from selectors import EpollSelector
import sys
import glob
import time
import torch
import numpy as np
import flowiz as fz
from PIL import Image, ImageDraw
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui  import QPixmap, QFont
# from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QWidget, QLabel, QRadioButton

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import manifold
from sklearn.manifold import TSNE
import umap

import nero_transform
import nero_utilities
import nero_run_model

# globa configurations
pg.setConfigOptions(antialias=True, background='w')
# use pyside gpu acceleration if gpu detected
if torch.cuda.is_available():
    # pg.setConfigOption('useCupy', True)
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

        # load/initialize program cache
        self.use_cache = False
        self.cache_dir = os.path.join(os.getcwd(), 'cache')
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        self.cache_path = os.path.join(self.cache_dir, 'nero_cache.npz')
        # if not exist, creat one
        if not os.path.isfile(self.cache_path):
            np.savez(self.cache_path)

        self.cache = dict(np.load(self.cache_path, allow_pickle=True))

        # if we are doing real-time inference when dragging the field of view
        if torch.cuda.is_available():
            self.realtime_inference = True
        else:
            self.realtime_inference = False


    # helper functions on managing the database
    def load_from_cache(self, name):
        # if it exists
        if name in self.cache.keys():
            return self.cache[name]
        else:
            print(f'No precomputed result named {name}')
            return None
        # return getattr(self.cache, name)


    def save_to_cache(self, name, content):
        # replace if exists
        self.cache[name] = content
        np.savez(self.cache_path, **self.cache)


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

        # for cases where only image is loaded but no other things
        if self.image_existed and not self.aggregate_result_existed:
            self.clear_layout(self.single_result_layout)

        if self.aggregate_result_existed:
            self.clear_layout(self.aggregate_result_layout)
            self.single_result_existed = False
            self.aggregate_result_existed = False

        if self.run_button_existed:
            print(f'Cleaned previous run button')
            self.clear_layout(self.run_button_layout)
            self.run_button_existed = False

        if self.aggregate_result_existed:
            print(f'Cleaned {self.previous_mode} aggregate_result_layout')
            self.clear_layout(self.aggregate_result_layout)
            self.data_existed = False
            self.aggregate_result_existed = False
            self.single_result_existed = False

        if self.single_result_existed:
            print(f'Cleaned {self.previous_mode} single_result_layout')
            self.clear_layout(self.single_result_layout)
            self.image_existed = False
            self.single_result_existed = False


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

                # image size that is fed into the model
                self.image_size = 29
                # image size that is used for display
                self.display_image_size = 150
                # heatmap and detailed image plot size
                self.plot_size = 320
                # image (input data) modification mode
                self.rotation = True
                self.translation = False
                # rotation angles
                self.cur_rotation_angle = 0
                # rotation step for evaluation
                self.rotation_step = 5

                # preload model 1
                self.model_1_name = 'Original model'
                self.model_1_cache_name = self.model_1_name.split(' ')[0]
                self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
                self.model_1 = nero_run_model.load_model(self.mode, 'non_eqv', self.model_1_path)

                # preload model 2
                self.model_2_name = 'DA model'
                self.model_2_cache_name = self.model_2_name.split(' ')[0]
                self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
                self.model_2 = nero_run_model.load_model(self.mode, 'aug_eqv', self.model_2_path)
                # self.model_2_name = 'E2CNN model'
                # self.model_2_cache_name = self.model_2_name.split(' ')[0]
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

                # uncropped image size
                self.uncropped_image_size = 256
                # image (cropped) size that is fed into the model
                self.image_size = 128
                # image size that is used for display
                self.display_image_size = 256
                # heatmap and detailed image plot size
                self.plot_size = 320
                # image (input data) modification mode
                self.translation = False
                # translation step when evaluating
                self.translation_step_aggregate = 4
                self.translation_step_single = 4

                # predefined model paths
                self.model_1_name = 'FasterRCNN (0% jittering)'
                self.model_1_cache_name = self.model_1_name.split('(')[1].split(')')[0].split(' ')[0]
                self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_0-jittered', '*.pth'))[0]
                # pre-trained model does not need model path
                self.model_2_name = 'FasterRCNN (Pre-trained)'
                self.model_2_cache_name = self.model_2_name.split('(')[1].split(')')[0].split(' ')[0]
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
                self.last_clicked = None
                self.cur_line = None

                self.previous_mode = self.mode

        @QtCore.Slot()
        def piv_button_clicked():
            print('PIV button clicked')
            self.mode = 'piv'
            self.radio_button_3.setChecked(True)

            # below layouts depend on mode selection
            if self.previous_mode != self.mode or not self.previous_mode:
                # clear previous mode's layout
                self.switch_mode_cleanup()

                self.init_load_layout()

                # image (cropped) size that is fed into the model
                self.image_size = 256
                # image size that is used for display
                self.display_image_size = 256
                # heatmap and detailed image plot size
                self.plot_size = 320
                # image (input data) modification mode
                self.rotation = False
                self.flip = False
                self.time_reverse = False

                # predefined model paths
                self.model_1_name = 'PIV-LiteFlowNet-en'
                # LiteFlowNet
                self.model_1_cache_name = self.model_1_name.split('-')[1]
                self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'PIV-LiteFlowNet-en', f'*.paramOnly'))[0]
                # Horn-Schunck does not need model path
                self.model_2_name = 'Horn-Schunck'
                self.model_2_cache_name = self.model_2_name
                self.model_2_path = None
                # preload model
                self.model_1 = nero_run_model.load_model(self.mode, self.model_1_name, self.model_1_path)
                # Horn-Schunck is not a model
                # self.model_2 = nero_run_model.load_model(self.mode, self.model_1_name, self.model_1_path)
                self.model_2 = None

                # unique quantity of the result of current data
                self.all_quantities_1 = []
                self.all_quantities_2 = []

                # when doing highlighting
                self.last_clicked = None

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
        self.mode_label.setContentsMargins(0, 0, 0, 0)
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


    # load single mnist image from self.image_path
    def load_single_image(self):

        if self.mode == 'digit_recognition':
            self.loaded_image_label = int(self.image_path.split('/')[-1].split('_')[1])
            # load the image
            self.loaded_image_pt = torch.from_numpy(np.asarray(Image.open(self.image_path)))[:, :, None]
            self.loaded_image_name = self.image_path.split('/')[-1]
            # keep a copy to represent the current (rotated) version of the original images
            self.cur_image_pt = self.loaded_image_pt.clone()

        elif self.mode == 'object_detection':
            self.label_path = self.image_path.replace('images', 'labels').replace('jpg', 'npy').replace('jpg', 'npy')
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
            self.loaded_image_pt = torch.from_numpy(np.asarray(Image.open(self.image_path).convert('RGB'), dtype=np.uint8))
            self.loaded_image_name = self.image_path.split('/')[-1]

            # take the cropped part of the entire input image to put in display image
            self.cur_image_pt = self.loaded_image_pt[self.center_y-self.display_image_size//2:self.center_y+self.display_image_size//2, self.center_x-self.display_image_size//2:self.center_x+self.display_image_size//2, :]

        elif self.mode == 'piv':
            # keep the PIL image version of the loaded images in this mode because they are saved as gif using PIL
            # pre-trained PIV-LiteFlowNet-en takes 3 channel images
            self.loaded_image_1_pil = Image.open(self.image_1_path).convert('RGB')
            self.loaded_image_2_pil = Image.open(self.image_2_path).convert('RGB')

            # create a blank PIL image for gif purpose
            self.blank_image_pil = Image.fromarray(np.zeros((self.image_size, self.image_size, 3)), 'RGB')

            # convert to torch tensor
            self.loaded_image_1_pt = torch.from_numpy(np.asarray(self.loaded_image_1_pil))
            self.loaded_image_2_pt = torch.from_numpy(np.asarray(self.loaded_image_2_pil))
            self.loaded_image_1_name = self.image_1_path.split('/')[-1]
            self.loaded_image_2_name = self.image_2_path.split('/')[-1]
            # a separate copy to represent the transformed version of the original images
            self.cur_image_1_pt = self.loaded_image_1_pt.clone()
            self.cur_image_2_pt = self.loaded_image_2_pt.clone()

            # save the pil images as gif to cache
            # use multiple copies of image 1 and 2 to make blank smaller portion in time
            other_images_pil = [self.loaded_image_1_pil, self.loaded_image_2_pil, self.loaded_image_2_pil, self.blank_image_pil]
            self.gif_path = os.path.join(self.cache_dir, self.loaded_image_1_name.split('.')[0] + '.gif')
            self.loaded_image_1_pil.save(fp=self.gif_path,
                                         format='GIF',
                                         append_images=other_images_pil,
                                         save_all=True,
                                         duration=400,
                                         loop=0)

            # load the ground truth flow field
            self.loaded_image_label_pt = torch.from_numpy(fz.read_flow(self.label_path))


    def init_load_layout(self):

        # load aggregate dataset drop-down menu
        @QtCore.Slot()
        def aggregate_dataset_selection_changed(text):
            # filter out 0 selection signal
            if text == 'Input dataset':
                return

            # flag on if control has been set up
            self.aggregate_plot_control_existed = False

            # clear the single image selection
            self.image_menu.setCurrentIndex(0)

            # clear previous result layout
            if self.data_existed:
                self.clear_layout(self.run_button_layout)
                self.clear_layout(self.aggregate_result_layout)
                self.aggregate_result_existed = False
                self.data_existed = False
                self.run_button_existed = False
                print('Previous aggregate result layout deleted')

            # since single case starts with image loading
            if self.image_existed:
                # move model selection menus back to load menu
                self.load_menu_layout.addWidget(self.model_1_menu, 2, 3)
                self.load_menu_layout.addWidget(self.model_2_menu, 3, 3)
                self.clear_layout(self.run_button_layout)
                self.clear_layout(self.single_result_layout)
                self.single_result_existed = False
                self.image_existed = False
                self.run_button_existed = False
                print('Previous single result layout deleted')

            self.init_aggregate_result_layout()

            print('Loaded dataset:', text)
            self.data_mode = 'aggregate'
            self.dataset_index = int(text.split(' ')[-1])
            self.dataset_dir = self.aggregate_data_dirs[self.dataset_index]
            # in digit recognition, all the images are loaded
            if self.mode == 'digit_recognition':
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

            # in object detection, only all the image paths are loaded
            elif self.mode == 'object_detection':
                # all the images and labels paths
                self.all_images_paths = glob.glob(os.path.join(self.dataset_dir, 'images', '*.jpg'))
                self.all_labels_paths = []
                for cur_image_path in self.all_images_paths:
                    cur_label_path = cur_image_path.replace('images', 'labels').replace('jpg', 'npy')
                    self.all_labels_paths.append(cur_label_path)

                # name of the classes of each label
                self.loaded_images_labels = []
                for i, cur_label_path in enumerate(self.all_labels_paths):
                    cur_label = cur_label_path.split('/')[-1].split('_')[0]
                    self.loaded_images_labels.append(cur_label)

            # in piv, only the paths are loaded
            elif self.mode == 'piv':
                # all the image pairs and labels paths
                self.all_images_1_paths = glob.glob(os.path.join(self.dataset_dir, '*img1.tif'))
                self.all_images_2_paths = [cur_path.replace('img1', 'img2') for cur_path in self.all_images_1_paths]
                self.all_labels_paths = [cur_path.replace('img1', 'flow').replace('tif', 'flo') for cur_path in self.all_images_1_paths]
                # flow type of each image pair
                self.all_individual_flow_types = [cur_path.split('/')[-1].split('_')[0] for cur_path in self.all_images_1_paths]

            # check the data to be ready
            self.data_existed = True

            # show the run button when data is loaded
            if not self.run_button_existed:
                # run button
                # buttons layout for run model
                self.run_button_layout = QtWidgets.QGridLayout()
                # no displayed test image as in the single case so layout row number-1
                self.layout.addLayout(self.run_button_layout, 2, 0, 1, 2)

                self.run_button = QtWidgets.QPushButton('Analyze model')
                self.run_button.setStyleSheet('font-size: 18px')
                self.run_button.setFixedSize(QtCore.QSize(250, 50))
                self.run_button_layout.addWidget(self.run_button)
                self.run_button.clicked.connect(self.run_button_clicked)

                # instead of running, we can also load results from cache
                self.use_cache_checkbox = QtWidgets.QCheckBox('Use previously computed result')
                self.use_cache_checkbox.setStyleSheet('font-size: 18px')
                self.use_cache_checkbox.setFixedSize(QtCore.QSize(300, 50))
                self.use_cache_checkbox.stateChanged.connect(run_cache_checkbox_clicked)
                self.run_button_layout.addWidget(self.use_cache_checkbox)

                self.run_button_existed = True
            else:
                self.run_button.setText('Analyze model')

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
                # move model selection menus back to load menu
                self.load_menu_layout.addWidget(self.model_1_menu, 2, 3)
                self.load_menu_layout.addWidget(self.model_2_menu, 3, 3)
                # move run button back to run button layout
                self.run_button_layout.addWidget(self.run_button)
                self.run_button_layout.addWidget(self.use_cache_checkbox)
                # clear result layout
                self.clear_layout(self.aggregate_result_layout)
                self.aggregate_result_existed = False
                self.data_existed = False
            if self.image_existed:
                # move model selection menus back to load menu
                self.load_menu_layout.addWidget(self.model_1_menu, 2, 3)
                self.load_menu_layout.addWidget(self.model_2_menu, 3, 3)
                # move run button back to run button layout
                self.run_button_layout.addWidget(self.run_button)
                self.run_button_layout.addWidget(self.use_cache_checkbox)
                # clear result layout
                self.clear_layout(self.single_result_layout)
                self.single_result_existed = False
                self.image_existed = False

            self.data_mode = 'single'

            self.init_single_result_layout()

            print('Loaded image:', text)
            if self.mode == 'digit_recognition':
                # prepare image path
                self.image_index = int(text.split(' ')[-1])
                self.image_path = self.single_images_paths[self.image_index]
                self.load_single_image()

                # convert to QImage for display purpose
                self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt)
                # resize the display QImage
                self.cur_display_image = self.cur_display_image.scaledToWidth(self.display_image_size)
                # additional preparation required for MNIST
                self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)

            elif self.mode == 'object_detection':
                # prepare image path
                self.image_index = self.coco_classes.index(text.split(' ')[0])
                self.image_path = self.single_images_paths[self.image_index]
                self.load_single_image()

                # convert to QImage for display purpose
                self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt)
                # resize the display QImage
                self.cur_display_image = self.cur_display_image.scaledToWidth(self.display_image_size)

            elif self.mode == 'piv':
                # prepare image paths
                self.image_index = int(text.split(' ')[2])
                self.image_1_path = self.single_images_1_paths[self.image_index]
                self.image_2_path = self.single_images_2_paths[self.image_index]
                self.label_path = self.single_labels_paths[self.image_index]
                self.load_single_image()

            # display the image
            self.display_image()
            self.image_existed = True

            # show the run button when data is loaded
            if not self.run_button_existed:
                # run button
                # buttons layout for run model
                self.run_button_layout = QtWidgets.QVBoxLayout()
                self.layout.addLayout(self.run_button_layout, 3, 0, 1, 2)

                self.run_button_text = 'Analyze model'
                self.run_button = QtWidgets.QPushButton(self.run_button_text)
                self.run_button.setStyleSheet('font-size: 18px')
                self.run_button.setFixedSize(QtCore.QSize(250, 50))
                self.run_button.clicked.connect(self.run_button_clicked)
                self.run_button_layout.addWidget(self.run_button)

                # instead of running, we can also load results from cache
                self.use_cache_checkbox = QtWidgets.QCheckBox('Use previously computed result')
                self.use_cache_checkbox.setStyleSheet('font-size: 18px')
                self.use_cache_checkbox.setFixedSize(QtCore.QSize(300, 50))
                self.use_cache_checkbox.stateChanged.connect(run_cache_checkbox_clicked)
                self.run_button_layout.addWidget(self.use_cache_checkbox)

                self.run_button_existed = True
            else:
                self.run_button.setText(self.run_button_text)

        @QtCore.Slot()
        def run_cache_checkbox_clicked(state):
            if state == QtCore.Qt.Checked:
                self.use_cache = True
            else:
                self.use_cache = False

        # two drop down menus that let user choose models
        @QtCore.Slot()
        def model_1_selection_changed(text):
            print('Model 1:', text)
            self.model_1_name = text
            if self.mode == 'digit_recognition':
                # Original, E2CNN or DA
                self.model_1_cache_name = self.model_1_name.split(' ')[0]
                # load the mode
                if text == 'Original model':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'non-eqv', self.model_1_path)
                elif text == 'E2CNN model':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'rot-eqv', self.model_1_path)
                elif text == 'DA model':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'aug-eqv', self.model_1_path)

                print('Model 1 path:', self.model_1_path)

            elif self.mode == 'object_detection':
                self.model_1_cache_name = self.model_1_name.split('(')[1].split(')')[0].split(' ')[0]
                if text == 'FasterRCNN (0% jittering)':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_0-jittered', '*.pth'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_1_path)
                    print('Model 1 path:', self.model_1_path)
                elif text == 'FasterRCNN (20% jittering)':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_20-jittered', '*.pth'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_1_path)
                    print('Model 1 path:', self.model_1_path)
                elif text == 'FasterRCNN (40% jittering)':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_40-jittered', '*.pth'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_1_path)
                    print('Model 1 path:', self.model_1_path)
                elif text == 'FasterRCNN (60% jittering)':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_60-jittered', '*.pth'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_1_path)
                    print('Model 1 path:', self.model_1_path)
                elif text == 'FasterRCNN (80% jittering)':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_80-jittered', '*.pth'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_1_path)
                    print('Model 1 path:', self.model_1_path)
                elif text == 'FasterRCNN (100% jittering)':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_100-jittered', '*.pth'))[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_1_path)
                    print('Model 1 path:', self.model_1_path)
                elif text == 'FasterRCNN (Pre-trained)':
                    self.model_1_path = None
                    self.model_1 = nero_run_model.load_model(self.mode, 'pre_trained', self.model_1_path)
                    print('Model 1 path: Downloaded from PyTorch')

            elif self.mode == 'piv':
                self.model_1_cache_name = self.model_1_name.split('-')[1]
                if text == 'PIV-LiteFlowNet-en':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'PIV-LiteFlowNet-en', f'*.pt'))[0]
                    self.model_1 = nero_run_model.load_model(self.mode, self.model_1_name, self.model_1_path)
                    print('Model 1 path:', self.model_1_path)

                elif text == 'Horn-Schunck':
                    # Horn-Schunck does not need model path
                    self.model_2_path = None
                    self.model_2 = None

            # when loaded data is available, just show the result without clicking the button
            if self.use_cache:
                if self.data_mode == 'aggregate':
                    self.run_model_aggregated()
                    self.aggregate_result_existed = True

                    # run dimension reduction if previously run
                    if self.dr_result_existed:
                        self.run_dimension_reduction()

                elif self.data_mode == 'single':
                    self.run_model_all()
                    self.single_result_existed = True

        @QtCore.Slot()
        def model_2_selection_changed(text):
            print('Model 2:', text)
            self.model_2_name = text
            if self.mode == 'digit_recognition':
                # Original, E2CNN or DA
                self.model_2_cache_name = self.model_1_name.split(' ')[0]
                # load the mode
                if text == 'Original model':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model('non_eqv', self.model_2_path)
                elif text == 'E2CNN model':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model('rot_eqv', self.model_2_path)
                elif text == 'DA model':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model('aug_eqv', self.model_2_path)

                print('Model 2 path:', self.model_2_path)

            elif self.mode == 'object_detection':
                self.model_2_cache_name = self.model_2_name.split('(')[1].split(')')[0].split(' ')[0]
                if text == 'FasterRCNN (0% jittering)':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_0-jittered', '*.pth'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_2_path)
                    print('Model 2 path:', self.model_2_path)
                elif text == 'FasterRCNN (20% jittering)':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_20-jittered', '*.pth'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_2_path)
                    print('Model 2 path:', self.model_2_path)
                elif text == 'FasterRCNN (40% jittering)':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_40-jittered', '*.pth'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_2_path)
                    print('Model 2 path:', self.model_2_path)
                elif text == 'FasterRCNN (60% jittering)':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_60-jittered', '*.pth'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_2_path)
                    print('Model 2 path:', self.model_2_path)
                elif text == 'FasterRCNN (80% jittering)':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_80-jittered', '*.pth'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_2_path)
                    print('Model 2 path:', self.model_2_path)
                elif text == 'FasterRCNN (100% jittering)':
                    self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'custom_trained', f'object_100-jittered', '*.pth'))[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(self.mode, 'custom_trained', self.model_2_path)
                    print('Model 2 path:', self.model_2_path)
                elif text == 'FasterRCNN (Pre-trained)':
                    self.model_2_path = None
                    self.model_2 = nero_run_model.load_model(self.mode, 'pre_trained', self.model_2_path)
                    print('Model 2 path: Downloaded from PyTorch')

            elif self.mode == 'piv':
                self.model_1_cache_name = self.model_1_name.split('-')[1]
                if text == 'PIV-LiteFlowNet-en':
                    self.model_1_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'PIV-LiteFlowNet-en', f'*.pt'))[0]
                    self.model_1 = nero_run_model.load_model(self.mode, self.model_1_name, self.model_1_path)
                    print('Model 1 path:', self.model_1_path)

                elif text == 'Horn-Schunck':
                    # Horn-Schunck does not need model path
                    self.model_2_path = None
                    self.model_2 = None

            # when loaded data is available, just show the result without clicking the button
            if self.use_cache:
                if self.data_mode == 'aggregate':
                    self.run_model_aggregated()
                    self.aggregate_result_existed = True

                    # run dimension reduction if previously run
                    if self.dr_result_existed:
                        self.run_dimension_reduction()

                elif self.data_mode == 'single':
                    self.run_model_all()
                    self.single_result_existed = True

        # function used as model icon
        def draw_circle(painter, center_x, center_y, radius, color):
            # set up brush and pen
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setBrush(QtGui.QColor(color))
            painter.setPen(QtGui.QColor(color))
            center = QtCore.QPoint(center_x, center_y)
            # draw the circle
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
        self.model_label.setContentsMargins(0, 0, 0, 0)
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

        # data dir
        self.aggregate_data_dirs = glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, f'aggregate'))

        if self.mode == 'digit_recognition':
            # load all images in the folder
            for i in range(len(self.aggregate_data_dirs)):
                self.aggregate_image_menu.addItem(f'Test {i}')

            # set default to the prompt/description
            self.aggregate_image_menu.setCurrentIndex(0)

        elif self.mode == 'object_detection':
            # load all images in the folder
            for i in range(len(self.aggregate_data_dirs)):
                self.aggregate_image_menu.addItem(f'Test {i}')

            # set default to the prompt/description
            self.aggregate_image_menu.setCurrentIndex(0)

        elif self.mode == 'piv':
            # load all images in the folder
            for i in range(len(self.aggregate_data_dirs)):
                self.aggregate_image_menu.addItem(f'Test {i}')

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
                cur_image_path = glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, 'single', 'images', f'{cur_class}*.jpg'))[0]
                self.single_images_paths.append(cur_image_path)
                self.image_menu.addItem(QtGui.QIcon(cur_image_path), f'{cur_class} image')

            self.image_menu.setCurrentIndex(0)

        elif self.mode == 'piv':
            # different flow types
            self.flow_types = ['uniform', 'backstep', 'cylinder', 'SQG', 'DNS', 'JHTDB']
            self.single_images_1_paths = []
            self.single_images_2_paths = []
            self.single_labels_paths = []
            # image pairs
            for i, flow_type in enumerate(self.flow_types):
                cur_image_1_path = glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, 'single', f'{flow_type}*img1.tif'))[0]
                self.single_images_1_paths.append(cur_image_1_path)
                self.single_images_2_paths.append(cur_image_1_path.replace('img1', 'img2'))
                self.single_labels_paths.append(cur_image_1_path.replace('img1', 'flow').replace('tif', 'flo'))

                # add flow to the menu
                self.image_menu.addItem(f'Image pair {i} - {flow_type}')

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

        # load models choices
        # model 1
        # graphic representation
        self.model_1_label = QLabel(self)
        self.model_1_label.setContentsMargins(0, 0, 0, 0)
        self.model_1_label.setAlignment(QtCore.Qt.AlignCenter)
        model_1_icon = QPixmap(25, 25)
        model_1_icon.fill(QtCore.Qt.white)
        # draw model representation
        painter = QtGui.QPainter(model_1_icon)
        draw_circle(painter, 12, 12, 10, 'blue')

        # spacer item
        # self.mode_control_layout.addSpacing(30)

        self.model_1_menu = QtWidgets.QComboBox()
        self.model_1_menu.setFixedSize(QtCore.QSize(300, 50))
        self.model_1_menu.setStyleSheet('font-size: 18px')
        if self.mode == 'digit_recognition':
            self.model_1_menu.addItem(model_1_icon, 'Original model')
            self.model_1_menu.addItem(model_1_icon, 'E2CNN model')
            self.model_1_menu.addItem(model_1_icon, 'DA model')
            self.model_1_menu.setCurrentText('Original model')
        elif self.mode == 'object_detection':
            self.model_1_menu.addItem(model_1_icon, 'FasterRCNN (0% jittering)')
            self.model_1_menu.addItem(model_1_icon, 'FasterRCNN (20% jittering)')
            self.model_1_menu.addItem(model_1_icon, 'FasterRCNN (40% jittering)')
            self.model_1_menu.addItem(model_1_icon, 'FasterRCNN (60% jittering)')
            self.model_1_menu.addItem(model_1_icon, 'FasterRCNN (80% jittering)')
            self.model_1_menu.addItem(model_1_icon, 'FasterRCNN (100% jittering)')
            self.model_1_menu.addItem(model_1_icon, 'FasterRCNN (Pre-trained)')
            self.model_1_menu.setCurrentText('Custom-trained FasterRCNN')
        elif self.mode == 'piv':
            self.model_1_menu.addItem(model_1_icon, 'PIV-LiteFlowNet-en')
            self.model_1_menu.addItem(model_1_icon, 'Horn-Schunck')
            self.model_1_menu.setCurrentText('PIV-LiteFlowNet-en')

        # connect the drop down menu with actions
        self.model_1_menu.currentTextChanged.connect(model_1_selection_changed)
        self.model_1_menu.setEditable(True)
        self.model_1_menu.lineEdit().setReadOnly(True)
        self.model_1_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.load_menu_layout.addWidget(self.model_1_menu, 2, 3)

        # model 2
        # graphic representation
        self.model_2_label = QLabel(self)
        self.model_2_label.setContentsMargins(0, 0, 0, 0)
        self.model_2_label.setAlignment(QtCore.Qt.AlignCenter)
        model_2_icon = QPixmap(25, 25)
        model_2_icon.fill(QtCore.Qt.white)
        # draw model representation
        painter = QtGui.QPainter(model_2_icon)
        draw_circle(painter, 12, 12, 10, 'magenta')

        # spacer item
        # self.mode_control_layout.addSpacing(30)

        self.model_2_menu = QtWidgets.QComboBox()
        self.model_2_menu.setFixedSize(QtCore.QSize(300, 50))
        self.model_2_menu.setStyleSheet('font-size: 18px')
        self.model_2_menu.setEditable(True)
        self.model_2_menu.lineEdit().setReadOnly(True)
        self.model_2_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        if self.mode == 'digit_recognition':
            self.model_2_menu.addItem(model_2_icon, 'Original model')
            self.model_2_menu.addItem(model_2_icon, 'E2CNN model')
            self.model_2_menu.addItem(model_2_icon, 'DA model')
            # model_2_menu.setCurrentText('E2CNN model')
            self.model_2_menu.setCurrentText('DA model')
        elif self.mode == 'object_detection':
            self.model_2_menu.addItem(model_2_icon, 'FasterRCNN (0% jittering)')
            self.model_2_menu.addItem(model_2_icon, 'FasterRCNN (20% jittering)')
            self.model_2_menu.addItem(model_2_icon, 'FasterRCNN (40% jittering)')
            self.model_2_menu.addItem(model_2_icon, 'FasterRCNN (60% jittering)')
            self.model_2_menu.addItem(model_2_icon, 'FasterRCNN (80% jittering)')
            self.model_2_menu.addItem(model_2_icon, 'FasterRCNN (100% jittering)')
            self.model_2_menu.addItem(model_2_icon, 'FasterRCNN (Pre-trained)')
            self.model_2_menu.setCurrentText('FasterRCNN (Pre-trained)')
        elif self.mode == 'piv':
            self.model_2_menu.addItem(model_2_icon, 'PIV-LiteFlowNet-en')
            self.model_2_menu.addItem(model_2_icon, 'Horn-Schunck')
            self.model_2_menu.setCurrentText('Horn-Schunck')

        # connect the drop down menu with actions
        self.model_2_menu.currentTextChanged.connect(model_2_selection_changed)
        self.load_menu_layout.addWidget(self.model_2_menu, 3, 3)

        # add this layout to the general layout
        self.layout.addLayout(self.load_menu_layout, 0, 1)


    def init_aggregate_result_layout(self):

        # loaded images and model result layout
        self.aggregate_result_layout = QtWidgets.QGridLayout()
        # self.aggregate_result_layout.setContentsMargins(30, 50, 30, 50)

        # if model result ever existed
        self.aggregate_result_existed = False

        # batch size when running in aggregate mode
        if self.mode == 'digit_recognition':
            self.batch_size = 100
            # add to general layout
            self.layout.addLayout(self.aggregate_result_layout, 1, 0, 3, 3)
        elif self.mode == 'object_detection':
            self.batch_size = 64
            # add to general layout
            self.layout.addLayout(self.aggregate_result_layout, 1, 0, 3, 3)
        elif self.mode == 'piv':
            self.batch_size = 16
            self.layout.addLayout(self.aggregate_result_layout, 1, 0, 3, 3)


    def init_single_result_layout(self):
        # loaded images and model result layout
        self.single_result_layout = QtWidgets.QGridLayout()
        self.single_result_layout.setContentsMargins(30, 50, 30, 50)
        # self.single_result_layout.setContentsMargins(0, 0, 0, 0)

        # add to general layout
        if self.data_mode == 'single':
            # take up two columns in UI layout
            self.layout.addLayout(self.single_result_layout, 1, 0, 1, 2)
        elif self.data_mode == 'aggregate':
            self.layout.addLayout(self.single_result_layout, 1, 1)

        # if model result ever existed
        self.single_result_existed = False


    # run button execution that could be used by all modes
    @QtCore.Slot()
    def run_button_clicked(self):

        # aggregate (dataset selected) case
        if self.data_mode == 'aggregate':
            self.run_model_aggregated()
            self.aggregate_result_existed = True

        # single (single image selected) case
        elif self.data_mode == 'single':
            if self.mode == 'digit_recognition':
                # run model once and display bar result
                self.run_model_once()
                # run model all and display results (Individual NERO polar plot)
                self.run_model_all()

            elif self.mode == 'object_detection' or self.mode == 'piv':
                # run model all and display results (Individual NERO plot)
                self.run_model_all()

            self.single_result_existed = True


    # initialize digit selection control drop down menu
    def init_aggregate_plot_control(self):

        self.aggregate_plot_control_existed = True
        # mark if dimension reduction algorithm has been run
        self.dr_result_existed = False

        # aggregate class selection drop-down menu
        @QtCore.Slot()
        def aggregate_class_selection_changed(text):
            # update the current digit selection
            # first two cases are for digit recognition (MNIST)
            if self.mode == 'digit_recognition':
                if text.split(' ')[0] == 'Averaged':
                    self.class_selection = 'all'
                elif text.split(' ')[0] == 'Digit':
                    self.class_selection = int(text.split(' ')[-1])

                # display the plot
                self.display_mnist_aggregate_result()

            # for object detection (COCO)
            elif self.mode == 'object_detection':
                if text.split(' ')[0] == 'Averaged':
                    self.class_selection = 'all'
                else:
                    self.class_selection = text

                # display the plot
                self.display_coco_aggregate_result()

            # for piv
            elif self.mode == 'piv':
                # select different flows
                if text.split(' ')[0] == 'Averaged':
                    self.class_selection = 'all'
                else:
                    self.class_selection = text

                # display the plot
                self.display_piv_aggregate_result()

            # after change class, run new dimension reduction if previously run
            if self.dr_result_existed:
                self.run_dimension_reduction()

        # change different dimension reduction algorithms
        def dr_selection_changed(text):
            self.dr_selection = text

            # re-run dimension reduction and show result
            if self.dr_result_existed:
                self.run_dimension_reduction()

        # layout that controls the plotting items
        self.aggregate_plot_control_layout = QtWidgets.QGridLayout()
        # add plot control layout to general layout
        self.aggregate_result_layout.addLayout(self.aggregate_plot_control_layout, 0, 0, 3, 1)

        # drop down menu on choosing the display class
        self.class_selection_menu = QtWidgets.QComboBox()
        self.class_selection_menu.setFixedSize(QtCore.QSize(250, 50))
        self.class_selection_menu.setStyleSheet('font-size: 18px')
        if self.mode == 'digit_recognition':
            self.class_selection_menu.addItem(f'Averaged over all digits')
            # add all digits as items
            for i in range(10):
                self.class_selection_menu.addItem(f'Digit {i}')
        elif self.mode == 'object_detection':
            self.class_selection_menu.addItem(f'Averaged over all classes')
            # add all classes as items
            for cur_class in self.coco_classes:
                self.class_selection_menu.addItem(f'{cur_class}')
        elif self.mode == 'piv':
            self.class_selection_menu.addItem(f'Averaged over all types of flows')
            # add all classes as items
            for cur_type in self.flow_types:
                self.class_selection_menu.addItem(f'{cur_type}')

        # set default to 'all', which means the average one
        self.class_selection = 'all'
        self.class_selection_menu.setCurrentIndex(0)
        # connect the drop down menu with actions
        self.class_selection_menu.currentTextChanged.connect(aggregate_class_selection_changed)
        self.class_selection_menu.setEditable(True)
        self.class_selection_menu.lineEdit().setReadOnly(True)
        self.class_selection_menu.lineEdit().setAlignment(QtCore.Qt.AlignRight)
        # add to local layout
        self.aggregate_plot_control_layout.addWidget(self.class_selection_menu, 0, 0)

        # drop down menu on choosing the dimension reduction method
        self.dr_selection_menu = QtWidgets.QComboBox()
        self.dr_selection_menu.setFixedSize(QtCore.QSize(250, 50))
        self.dr_selection_menu.setStyleSheet('font-size: 18px')
        dr_algorithms = ['PCA', 'ICA', 'ISOMAP', 't-SNE', 'UMAP']
        for algo in dr_algorithms:
            self.dr_selection_menu.addItem(f'{algo}')
        # set default to digit 0, which means PCA
        self.dr_selection = dr_algorithms[0]
        self.dr_selection_menu.setCurrentIndex(0)
        # connect the drop down menu with actions
        self.dr_selection_menu.currentTextChanged.connect(dr_selection_changed)
        self.dr_selection_menu.setEditable(True)
        self.dr_selection_menu.lineEdit().setReadOnly(True)
        self.dr_selection_menu.lineEdit().setAlignment(QtCore.Qt.AlignRight)
        # add to local layout
        self.aggregate_plot_control_layout.addWidget(self.dr_selection_menu, 2, 0)

        # push button on running PCA
        self.run_dr_button = QtWidgets.QPushButton('See Overview')
        self.run_dr_button.setStyleSheet('font-size: 18px')
        self.run_dr_button.setFixedSize(QtCore.QSize(250, 50))
        self.run_dr_button.clicked.connect(self.run_dimension_reduction)
        self.aggregate_plot_control_layout.addWidget(self.run_dr_button, 3, 0)


    # run PCA on demand
    @QtCore.Slot()
    def run_dimension_reduction(self):

        self.dr_result_existed = True

        # helper function on computing dimension reductions
        def dimension_reduce(high_dim, target_dim):

            if self.dr_selection == 'PCA':
                pca = PCA(n_components=target_dim, svd_solver='full')
                low_dim = pca.fit_transform(high_dim)
            elif self.dr_selection == 'ICA':
                ica = FastICA(n_components=target_dim, random_state=12)
                low_dim = ica.fit_transform(high_dim)
            elif self.dr_selection == 'ISOMAP':
                low_dim = manifold.Isomap(n_neighbors=5, n_components=target_dim, n_jobs=-1).fit_transform(high_dim)
            elif self.dr_selection == 't-SNE':
                low_dim = TSNE(n_components=target_dim, n_iter=250).fit_transform(high_dim)
            elif self.dr_selection == 'UMAP':
                low_dim = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=target_dim).fit_transform(high_dim)

            return low_dim

        # helper function for clicking inside the scatter plot
        def low_dim_scatter_clicked(item, points):

            # clear previous visualization
            if self.last_clicked:
                self.last_clicked.resetPen()
                self.last_clicked.setBrush(self.old_brush)

            # only allow clicking one point at a time
            # save the old brush, use color to determine which plot gets the click
            self.old_brush = points[0].brush()

            # create new brush and brush the newly clicked point
            new_brush = pg.mkBrush(255, 0, 0, 255)
            points[0].setBrush(new_brush)
            points[0].setPen(5)

            self.last_clicked = points[0]

            # get the clicked scatter item's information
            self.image_index = int(item.opts['name'])

            # start single result view from here
            # if not self.image_existed:
            #     self.init_single_result_layout()

            # get the corresponding image path
            self.image_path = self.all_images_paths[self.image_index]
            print(f'Selected image at {self.image_path}')

            # load the image
            if self.mode == 'digit_recognition':
                self.load_single_image()
            elif self.mode == 'object_detection':
                self.load_single_image()

            # convert to QImage for display purpose
            self.cur_display_image = nero_utilities.tensor_to_qt_image(self.cur_image_pt)
            # resize the display QImage
            self.cur_display_image = self.cur_display_image.scaledToWidth(self.display_image_size)

            if self.mode == 'digit_recognition':
                # display the image
                self.display_image()
                # prepare image tensor for model purpose
                self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)
                # run model once and display results (Detailed bar plot)
                self.run_model_once()

            # run model all and display results (Individual NERO plot)
            self.run_model_all()

        # helper function on normalizing low dimension points within [-1, 1] sqaure
        def normalize_low_dim_result(low_dim):
            # to preserve shape, largest bound are taken from either axis
            all_min = np.min(low_dim)
            all_max = np.max(low_dim)

            # lerp each point's x and y into [-1, 1]
            new_low_dim = np.zeros(low_dim.shape)
            new_low_dim[:, 0] = nero_utilities.lerp(low_dim[:, 0], all_min, all_max, -1, 1)
            new_low_dim[:, 1] = nero_utilities.lerp(low_dim[:, 1], all_min, all_max, -1, 1)

            return new_low_dim

        # run dimension reduction of all images on the selected digit
        # each image has tensor with length being the number of translations
        cur_class_indices = []
        if self.class_selection == 'all':
            # all the indices
            cur_class_indices = list(range(len(self.loaded_images_labels)))
        else:
            for i in range(len(self.loaded_images_labels)):
                if self.class_selection == self.loaded_images_labels[i]:
                    cur_class_indices.append(i)

        if self.mode == 'digit_recognition':
            num_transformations = len(self.all_aggregate_angles)
        elif self.mode == 'object_detection':
            num_transformations = len(self.x_translation) * len(self.y_translation)

        all_high_dim_points_1 = np.zeros((len(cur_class_indices), num_transformations))
        all_high_dim_points_2 = np.zeros((len(cur_class_indices), num_transformations))

        for i, index in enumerate(cur_class_indices):
            # go through all the transfomations
            for j in range(num_transformations):
                if self.mode == 'digit_recognition':
                    # all_outputs has shape (num_rotations, num_samples, 10)
                    all_high_dim_points_1[i, j] = int(self.all_outputs_1[j][index].argmax() == self.loaded_images_labels[index])
                    all_high_dim_points_2[i, j] = int(self.all_outputs_2[j][index].argmax() == self.loaded_images_labels[index])

                elif self.mode == 'object_detection':

                    y = int(j//len(self.x_translation))
                    x = int(j%len(self.x_translation))

                    # aggregate_outputs_1 has shape (num_y_translations, num_x_translations, num_samples, 7)
                    cur_conf_1 = self.aggregate_outputs_1[y, x][index][0, 4]
                    cur_iou_1= self.aggregate_outputs_1[y, x][index][0, 6]

                    cur_conf_2 = self.aggregate_outputs_2[y, x][index][0, 4]
                    cur_iou_2 = self.aggregate_outputs_2[y, x][index][0, 6]

                    if self.quantity_name == 'Confidence*IOU':
                        cur_value_1 = cur_conf_1 * cur_iou_1
                        cur_value_2 = cur_conf_2 * cur_iou_2
                    elif self.quantity_name == 'Confidence':
                        cur_value_1 = cur_conf_1
                        cur_value_2 = cur_conf_2
                    elif self.quantity_name == 'IOU':
                        cur_value_1 = cur_iou_1
                        cur_value_2 = cur_iou_2
                    elif self.quantity_name == 'Precision':
                        cur_value_1 = self.aggregate_precision_1[y, x][index]
                        cur_value_2 = self.aggregate_precision_2[y, x][index]
                    elif self.quantity_name == 'Recall':
                        cur_value_1 = self.aggregate_recall_1[y, x][index]
                        cur_value_2 = self.aggregate_recall_2[y, x][index]
                    elif self.quantity_name == 'F1 Score':
                        cur_value_1 = self.aggregate_F_measure_1[y, x][index]
                        cur_value_2 = self.aggregate_F_measure_2[y, x][index]
                    elif self.quantity_name == 'mAP':
                        cur_value_1 = 0
                        cur_value_2 = 0

                    all_high_dim_points_1[i, j] = cur_value_1
                    all_high_dim_points_2[i, j] = cur_value_2

        # get the average intensity of each sample
        all_intensity_1 = np.mean(all_high_dim_points_1, axis=1)
        all_intensity_2 = np.mean(all_high_dim_points_2, axis=1)

        # run dimension reduction algorithm
        low_dim_1 = dimension_reduce(all_high_dim_points_1, target_dim=2)
        low_dim_1 = normalize_low_dim_result(low_dim_1)
        low_dim_2 = dimension_reduce(all_high_dim_points_2, target_dim=2)
        low_dim_2 = normalize_low_dim_result(low_dim_2)

        # scatter plot on low-dim points
        low_dim_scatter_view_1 = pg.GraphicsLayoutWidget()
        low_dim_scatter_view_1.setBackground('white')
        low_dim_scatter_view_1.setFixedSize(self.plot_size, self.plot_size)

        # set axis range
        self.low_dim_scatter_plot_1 = low_dim_scatter_view_1.addPlot()
        # self.low_dim_scatter_plot_1
        self.low_dim_scatter_plot_1.setXRange(-1.2, 1.2, padding=0)
        self.low_dim_scatter_plot_1.setYRange(-1.2, 1.2, padding=0)
        # Not letting user zoom out past axis limit
        self.low_dim_scatter_plot_1.vb.setLimits(xMin=-1.2, xMax=1.2, yMin=-1.2, yMax=1.2)

        low_dim_scatter_view_2 = pg.GraphicsLayoutWidget()
        low_dim_scatter_view_2.setBackground('white')
        low_dim_scatter_view_2.setFixedSize(self.plot_size, self.plot_size)
        self.low_dim_scatter_plot_2 = low_dim_scatter_view_2.addPlot()
        self.low_dim_scatter_plot_2.setXRange(-1.2, 1.2, padding=0)
        self.low_dim_scatter_plot_2.setYRange(-1.2, 1.2, padding=0)
        # Not letting user zoom out past axis limit
        self.low_dim_scatter_plot_2.vb.setLimits(xMin=-1.2, xMax=1.2, yMin=-1.2, yMax=1.2)

        # save colorbar as used in aggregate NERO plot, to be used in color encode scatter points
        scatter_color_map = pg.colormap.get('viridis')
        scatter_lut = scatter_color_map.getLookupTable(start=0, stop=1, nPts=101, alpha=False)
        # quantize all the intensity into color
        color_indices_1 = []
        color_indices_2 = []
        for i in range(len(all_intensity_1)):
            color_indices_1.append(scatter_lut[int(all_intensity_1[i]*100)])
            color_indices_2.append(scatter_lut[int(all_intensity_2[i]*100)])

        for i, index in enumerate(cur_class_indices):
            # all the points to be plotted
            # add individual items for getting the item's name later when clicking
            # Set pxMode=False to allow spots to transform with the view
            self.low_dim_scatter_item_1 = pg.ScatterPlotItem(pxMode=False)
            self.low_dim_scatter_item_1.setSymbol('o')
            self.low_dim_scatter_item_2 = pg.ScatterPlotItem(pxMode=False)
            self.low_dim_scatter_item_2.setSymbol('o')

            low_dim_point_1 = [{'pos': (low_dim_1[i, 0], low_dim_1[i, 1]),
                                'size': 0.1,
                                'pen': {'color': 'w', 'width': 0.1},
                                'brush': QtGui.QColor(color_indices_1[i][0], color_indices_1[i][1], color_indices_1[i][2])}]

            low_dim_point_2 = [{'pos': (low_dim_2[i, 0], low_dim_2[i, 1]),
                                'size': 0.1,
                                'pen': {'color': 'w', 'width': 0.1},
                                'brush': QtGui.QColor(color_indices_2[i][0], color_indices_2[i][1], color_indices_2[i][2])}]

            # add points to the item
            self.low_dim_scatter_item_1.addPoints(low_dim_point_1, name=str(index))
            self.low_dim_scatter_item_2.addPoints(low_dim_point_2, name=str(index))

            # add points to the plot
            self.low_dim_scatter_plot_1.addItem(self.low_dim_scatter_item_1)
            self.low_dim_scatter_plot_2.addItem(self.low_dim_scatter_item_2)

            # connect click events on scatter items
            self.low_dim_scatter_item_1.sigClicked.connect(low_dim_scatter_clicked)
            self.low_dim_scatter_item_2.sigClicked.connect(low_dim_scatter_clicked)

        if self.mode == 'digit_recognition':
            self.aggregate_result_layout.addWidget(low_dim_scatter_view_1, 1, 3)
            self.aggregate_result_layout.addWidget(low_dim_scatter_view_2, 2, 3)
        elif self.mode == 'object_detection':
            # aggregate result layout at the very left
            self.aggregate_result_layout.addWidget(low_dim_scatter_view_1, 2, 1)
            self.aggregate_result_layout.addWidget(low_dim_scatter_view_2, 2, 2)


    # run model on the aggregate dataset
    def run_model_aggregated(self):

        if not self.aggregate_plot_control_existed:
            # initialize digit selection control
            self.init_aggregate_plot_control()

        if self.mode == 'digit_recognition':
            # all the rotation angles applied to the aggregated dataset
            self.all_aggregate_angles = list(range(0, 365, 5))

            # load from cache if available
            if self.use_cache:
                self.all_avg_accuracy_1 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_avg_accuracy')
                self.all_avg_accuracy_per_digit_1 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_avg_accuracy_per_digit')
                self.all_outputs_1 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_outputs')

                self.all_avg_accuracy_2 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_avg_accuracy')
                self.all_avg_accuracy_per_digit_2 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_avg_accuracy_per_digit')
                self.all_outputs_2 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_outputs')
            else:
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

                # save to cache
                self.save_to_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_avg_accuracy', self.all_avg_accuracy_1)
                self.save_to_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_avg_accuracy_per_digit', self.all_avg_accuracy_per_digit_1)
                self.save_to_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_outputs', self.all_outputs_1)

                self.save_to_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_avg_accuracy', self.all_avg_accuracy_2)
                self.save_to_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_avg_accuracy_per_digit', self.all_avg_accuracy_per_digit_2)
                self.save_to_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_outputs', self.all_outputs_2)


            # display the result
            self.display_mnist_aggregate_result()

        elif self.mode == 'object_detection':
            # all the translations in x and y applied to the aggregated dataset
            self.x_translation = list(range(-self.image_size//2, self.image_size//2, self.translation_step_aggregate))
            self.y_translation = list(range(-self.image_size//2, self.image_size//2, self.translation_step_aggregate))

            # output of each sample for all translations, has shape (num_y_trans, num_x_trans, num_samples, num_samples, 7)
            self.aggregate_outputs_1 = np.zeros((len(self.y_translation), len(self.x_translation)), dtype=np.ndarray)
            self.aggregate_outputs_2 = np.zeros((len(self.y_translation), len(self.x_translation)), dtype=np.ndarray)

            # individual precision, recall, F measure and AP
            self.aggregate_precision_1 = np.zeros((len(self.y_translation), len(self.x_translation)), dtype=np.ndarray)
            self.aggregate_recall_1 = np.zeros((len(self.y_translation), len(self.x_translation)), dtype=np.ndarray)
            self.aggregate_F_measure_1 = np.zeros((len(self.y_translation), len(self.x_translation)), dtype=np.ndarray)
            # mAP does not have individuals
            self.aggregate_mAP_1 = np.zeros((len(self.y_translation), len(self.x_translation)))

            self.aggregate_precision_2 = np.zeros((len(self.y_translation), len(self.x_translation)), dtype=np.ndarray)
            self.aggregate_recall_2 = np.zeros((len(self.y_translation), len(self.x_translation)), dtype=np.ndarray)
            self.aggregate_F_measure_2 = np.zeros((len(self.y_translation), len(self.x_translation)), dtype=np.ndarray)
            # mAP does not have individuals
            self.aggregate_mAP_2 = np.zeros((len(self.y_translation), len(self.x_translation)))

            if self.use_cache:
                self.aggregate_outputs_1 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_outputs')
                self.aggregate_precision_1 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_precision')
                self.aggregate_recall_1 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_recall')
                self.aggregate_mAP_1 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_mAP')
                self.aggregate_F_measure_1 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_F_measure')

                self.aggregate_outputs_2 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_outputs')
                self.aggregate_precision_2 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_precision')
                self.aggregate_recall_2 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_recall')
                self.aggregate_mAP_2 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_mAP')
                self.aggregate_F_measure_2 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_F_measure')
            else:
                # for all the loaded images
                for y, y_tran in enumerate(self.y_translation):
                    for x, x_tran in enumerate(self.x_translation):
                        print(f'y_tran = {y_tran}, x_tran = {x_tran}')
                        # model 1 output
                        cur_qualified_output_1, \
                        cur_precision_1, \
                        cur_recall_1, \
                        cur_F_measure_1 = nero_run_model.run_coco_once('aggregate',
                                                                        self.model_1_name,
                                                                        self.model_1,
                                                                        self.all_images_paths,
                                                                        self.custom_coco_names,
                                                                        self.pytorch_coco_names,
                                                                        batch_size=self.batch_size,
                                                                        x_tran=x_tran,
                                                                        y_tran=y_tran,
                                                                        coco_names=self.original_coco_names)

                        # save to result arrays
                        self.aggregate_outputs_1[y, x] = cur_qualified_output_1
                        self.aggregate_precision_1[y, x] = cur_precision_1
                        self.aggregate_recall_1[y, x] = cur_recall_1
                        self.aggregate_F_measure_1[y, x] = cur_F_measure_1
                        self.aggregate_mAP_1[y, x] = nero_utilities.compute_ap(cur_recall_1, cur_precision_1)

                        # model 2 output
                        cur_qualified_output_2, \
                        cur_precision_2, \
                        cur_recall_2, \
                        cur_F_measure_2 = nero_run_model.run_coco_once('aggregate',
                                                                        self.model_2_name,
                                                                        self.model_2,
                                                                        self.all_images_paths,
                                                                        self.custom_coco_names,
                                                                        self.pytorch_coco_names,
                                                                        batch_size=self.batch_size,
                                                                        x_tran=x_tran,
                                                                        y_tran=y_tran,
                                                                        coco_names=self.original_coco_names)

                        # save to result arrays
                        self.aggregate_outputs_2[y, x] = cur_qualified_output_2
                        self.aggregate_precision_2[y, x] = cur_precision_2
                        self.aggregate_recall_2[y, x] = cur_recall_2
                        self.aggregate_F_measure_2[y, x] = cur_F_measure_2
                        self.aggregate_mAP_2[y, x] = nero_utilities.compute_ap(cur_recall_2, cur_precision_2)

                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_outputs', content=self.aggregate_outputs_1)
                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_precision', content=self.aggregate_precision_1)
                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_recall', content=self.aggregate_recall_1)
                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_mAP', content=self.aggregate_mAP_1)
                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_F_measure', content=self.aggregate_F_measure_1)

                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_outputs', content=self.aggregate_outputs_2)
                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_precision', content=self.aggregate_precision_2)
                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_recall', content=self.aggregate_recall_2)
                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_mAP', content=self.aggregate_mAP_2)
                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_F_measure', content=self.aggregate_F_measure_2)


            # display the result
            self.display_coco_aggregate_result()

        elif self.mode == 'piv':
            # Dihedral group4 transformations
            all_rotation_degrees = [0, 90, 180, 270]
            # 0 means no flip/time reverse, 1 means flip/time reverse
            all_flip = [0, 1]
            all_time_reversals = [0, 1]
            self.num_transformations = len(all_rotation_degrees) * len(all_flip) * len(all_time_reversals)

            # output are dense 2D velocity field of the input image pairs
            # output for all transformation, has shape (num_transformations, num_samples, image_size, image_size, 2)
            self.aggregate_outputs_1 = torch.zeros((self.num_transformations, len(self.all_images_1_paths),  self.image_size, self.image_size, 2))
            self.aggregate_outputs_2 = torch.zeros((self.num_transformations, len(self.all_images_2_paths), self.image_size, self.image_size, 2))
            self.aggregate_ground_truths = torch.zeros((self.num_transformations, len(self.all_labels_paths), self.image_size, self.image_size, 2))

            if self.use_cache:
                self.aggregate_outputs_1 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_outputs')
                self.aggregate_outputs_2 = self.load_from_cache(f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_outputs')
                self.aggregate_ground_truths = self.load_from_cache(f'{self.mode}_{self.data_mode}_ground_truths')
            else:
                transformation_index = 0
                # take a batch of images
                num_batches = int(len(self.all_images_1_paths) / self.batch_size)
                if len(self.all_images_1_paths) % self.batch_size != 0:
                    num_batches += 1
                # construct sample indices for each batch
                batch_indices = []
                for i in range(num_batches):
                    batch_indices.append((i*self.batch_size, min((i+1)*self.batch_size, len(self.all_images_1_paths))))

                # go through each transformation type
                for is_time_reversed in all_time_reversals:
                    for is_flipped in all_flip:
                        for cur_rotation_angle in all_rotation_degrees:

                            print(f'Transformation {transformation_index}')

                            # modify the data and run model in batch
                            for index_range in batch_indices:
                                cur_images_1_paths = self.all_images_1_paths[index_range[0]:index_range[1]]
                                cur_images_2_paths = self.all_images_2_paths[index_range[0]:index_range[1]]
                                cur_labels_paths = self.all_labels_paths[index_range[0]:index_range[1]]

                                cur_images_1_pt = torch.zeros((len(cur_images_1_paths), self.image_size, self.image_size, 3))
                                cur_images_2_pt = torch.zeros((len(cur_images_2_paths), self.image_size, self.image_size, 3))
                                cur_labels = torch.zeros((len(cur_images_1_paths), self.image_size, self.image_size, 2))

                                # load and modify data of the current batch
                                for i in range(len(cur_images_1_paths)):
                                    # load the data
                                    cur_image_1_pil = Image.open(cur_images_1_paths[i]).convert('RGB')
                                    cur_image_2_pil = Image.open(cur_images_2_paths[i]).convert('RGB')
                                    # convert to torch tensor
                                    cur_image_1_pt = torch.from_numpy(np.asarray(cur_image_1_pil))
                                    cur_image_2_pt = torch.from_numpy(np.asarray(cur_image_2_pil))
                                    # load the ground truth flow field
                                    cur_label = torch.from_numpy(fz.read_flow(cur_labels_paths[i]))

                                    # modify the data
                                    if is_time_reversed:
                                        cur_image_1_pt, cur_image_2_pt, cur_label = nero_transform.reverse_piv_data(cur_image_1_pt, cur_image_2_pt, cur_label)

                                    if is_flipped:
                                        cur_image_1_pt, cur_image_2_pt, cur_label = nero_transform.flip_piv_data(cur_image_1_pt, cur_image_2_pt, cur_label)

                                    if cur_rotation_angle != 0:
                                        cur_image_1_pt, cur_image_2_pt, cur_label = nero_transform.rotate_piv_data(cur_image_1_pt, cur_image_2_pt, cur_label, cur_rotation_angle)

                                    # add to batch
                                    cur_images_1_pt[i] = cur_image_1_pt
                                    cur_images_2_pt[i] = cur_image_2_pt
                                    cur_labels[i] = cur_label

                                # run models on the current batch
                                cur_outputs_1 = nero_run_model.run_piv_once('aggregate',
                                                                            self.model_1_name,
                                                                            self.model_1,
                                                                            cur_images_1_pt,
                                                                            cur_images_2_pt)

                                cur_outputs_2 = nero_run_model.run_piv_once('aggregate',
                                                                            self.model_2_name,
                                                                            self.model_2,
                                                                            cur_images_1_pt,
                                                                            cur_images_2_pt)

                            # add to all outputs
                            self.aggregate_outputs_1[transformation_index, index_range[0]:index_range[1]] = cur_outputs_1 / self.image_size
                            self.aggregate_outputs_1[transformation_index, index_range[0]:index_range[1]] = cur_outputs_2 / self.image_size
                            self.aggregate_ground_truths[transformation_index, index_range[0]:index_range[1]] = cur_labels

                            transformation_index += 1

                # save to cache
                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_outputs', content=self.aggregate_outputs_1)
                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_outputs', content=self.aggregate_outputs_2)
                self.save_to_cache(name=f'{self.mode}_{self.data_mode}_ground_truths', content=self.aggregate_ground_truths)

            # display the result
            self.display_piv_aggregate_result()


    # run model on a single test sample with no transfomations
    def run_model_once(self):
        if self.mode == 'digit_recognition':

            self.output_1 = nero_run_model.run_mnist_once(self.model_1, self.cur_image_pt)
            self.output_2 = nero_run_model.run_mnist_once(self.model_2, self.cur_image_pt)

            # display result
            self.display_mnist_single_result(type='bar', boundary_width=3)

        elif self.mode == 'object_detection':

            self.output_1 = nero_run_model.run_coco_once('single',
                                                            self.model_1_name,
                                                            self.model_1,
                                                            self.cropped_image_pt,
                                                            self.custom_coco_names,
                                                            self.pytorch_coco_names,
                                                            test_label=self.cur_image_label)

            self.output_2 = nero_run_model.run_coco_once('single',
                                                            self.model_2_name,
                                                            self.model_2,
                                                            self.cropped_image_pt,
                                                            self.custom_coco_names,
                                                            self.pytorch_coco_names,
                                                            test_label=self.cur_image_label)


    # helper function that computes labels for cut out images
    def compute_label(self, cur_bounding_box, x_min, y_min, image_size):
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


    # run model on all the available transformations on a single sample
    def run_model_all(self):

        # quantity that is displayed in the individual NERO plot
        self.all_quantities_1 = []
        self.all_quantities_2 = []

        if self.mode == 'digit_recognition':
            self.all_angles = []
            # run all rotation test with 5 degree increment
            for self.cur_rotation_angle in range(0, 360+self.rotation_step, self.rotation_step):
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
            # when this is called in the single case
            if self.data_mode == 'single':
                # all the x and y translations
                # x translates on columns, y translates on rows
                self.x_translation = list(range(-self.image_size//2, self.image_size//2, self.translation_step_single))
                self.y_translation = list(range(-self.image_size//2, self.image_size//2, self.translation_step_single))
                num_x_translations = len(self.x_translation)
                num_y_translations = len(self.y_translation)
                self.all_translations = np.zeros((num_y_translations, num_x_translations, 2))

                if self.use_cache:
                    self.all_quantities_1 = self.load_from_cache(name=f'single_{self.model_1_cache_name}_{self.image_index}')
                    self.all_quantities_2 = self.load_from_cache(name=f'single_{self.model_2_cache_name}_{self.image_index}')
                else:
                    self.all_quantities_1 = np.zeros((num_y_translations, num_x_translations, 7))
                    self.all_quantities_2 = np.zeros((num_y_translations, num_x_translations, 7))

                for y, y_tran in enumerate(self.y_translation):
                    for x, x_tran in enumerate(self.x_translation):
                        # translation amout
                        # cur_x_tran and cur_y_tran are used to draw points on the heatmap to indicate translation amount
                        self.cur_x_tran = x_tran - self.x_translation[0]
                        # y axis needs to be converted from image axis to heatmap axis
                        self.cur_y_tran = -y_tran - self.y_translation[0]
                        # all_translations are for book keeping
                        self.all_translations[y, x] = [x_tran, y_tran]

                        # modify the underlying image tensor accordingly
                        # take the cropped part of the entire input image
                        cur_center_x = self.center_x - x_tran
                        cur_center_y = self.center_y - y_tran
                        self.x_min = cur_center_x - self.image_size//2
                        self.x_max = cur_center_x + self.image_size//2
                        self.y_min = cur_center_y - self.image_size//2
                        self.y_max = cur_center_y + self.image_size//2
                        # model takes image between [0, 1]
                        self.cropped_image_pt = self.loaded_image_pt[self.y_min:self.y_max, self.x_min:self.x_max, :] / 255

                        self.cur_image_label = np.zeros((len(self.loaded_image_label), 6))
                        for i in range(len(self.cur_image_label)):
                            # object index
                            self.cur_image_label[i, 0] = i
                            # since PyTorch FasterRCNN has 0 as background
                            self.cur_image_label[i, 5] = self.loaded_image_label[i, 4] + 1
                            # modify the label accordingly
                            self.cur_image_label[i, 1:5] = self.compute_label(self.loaded_image_label[i, :4], self.x_min, self.y_min, (self.image_size, self.image_size))

                        if self.use_cache:
                            continue

                        # re-display image for each rectangle drawn every 8 steps
                        if (x_tran)%2 == 0 and (y_tran)%2 == 0:
                            self.display_image()

                            display_rect_width = self.display_image_size/2
                            display_rect_height = self.display_image_size/2
                            # since the translation measures on the movement of object instead of the point of view, the sign is reversed
                            rect_center_x = self.display_image_size/2 - x_tran * (self.display_image_size/self.uncropped_image_size)
                            rect_center_y = self.display_image_size/2 - y_tran * (self.display_image_size/self.uncropped_image_size)

                            # draw rectangles on the displayed image to indicate scanning process
                            painter = QtGui.QPainter(self.image_pixmap)
                            # draw the rectangles
                            cover_color = QtGui.QColor(65, 65, 65, 225)
                            self.draw_fov_mask(painter, rect_center_x, rect_center_y, display_rect_width, display_rect_height, cover_color)

                            # end the painter
                            painter.end()

                            # draw ground truth label on the display image
                            # draw rectangle on the displayed image to indicate scanning process
                            painter = QtGui.QPainter(self.image_pixmap)
                            # draw the ground truth label
                            gt_display_center_x = (self.cur_image_label[0, 1] + self.cur_image_label[0, 3]) / 2 * (display_rect_width/self.image_size) + (rect_center_x - display_rect_width/2)
                            gt_display_center_y = (self.cur_image_label[0, 2] + self.cur_image_label[0, 4]) / 2 * (display_rect_height/self.image_size) + (rect_center_y - display_rect_height/2)
                            gt_display_rect_width = (self.cur_image_label[0, 3] - self.cur_image_label[0, 1]) * (display_rect_width/self.image_size)
                            gt_display_rect_height = (self.cur_image_label[0, 4] - self.cur_image_label[0, 2]) * (display_rect_height/self.image_size)
                            self.draw_rectangle(painter, gt_display_center_x, gt_display_center_y, gt_display_rect_width, gt_display_rect_height, color='yellow', label='Ground Truth')
                            painter.end()

                            # update pixmap with the label
                            self.image_label.setPixmap(self.image_pixmap)

                            # force repaint
                            self.image_label.repaint()

                        # run the model
                        # update the model output
                        self.output_1 = nero_run_model.run_coco_once('single',
                                                                        self.model_1_name,
                                                                        self.model_1,
                                                                        self.cropped_image_pt,
                                                                        self.custom_coco_names,
                                                                        self.pytorch_coco_names,
                                                                        test_label=self.cur_image_label)

                        self.output_2 = nero_run_model.run_coco_once('single',
                                                                        self.model_2_name,
                                                                        self.model_2,
                                                                        self.cropped_image_pt,
                                                                        self.custom_coco_names,
                                                                        self.pytorch_coco_names,
                                                                        test_label=self.cur_image_label)

                        # plotting the quantity regarding the correct label
                        quantity_1 = self.output_1[0][0][0]
                        quantity_2 = self.output_2[0][0][0]
                        self.all_quantities_1[y, x] = quantity_1
                        self.all_quantities_2[y, x] = quantity_2

                # display as the final x_tran, y_tran
                if self.use_cache:
                    self.display_image()
                    display_rect_width = self.display_image_size/2
                    display_rect_height = self.display_image_size/2
                    # since the translation measures on the movement of object instead of the point of view, the sign is reversed
                    rect_center_x = self.display_image_size/2 - x_tran * (self.display_image_size/self.uncropped_image_size)
                    rect_center_y = self.display_image_size/2 - y_tran * (self.display_image_size/self.uncropped_image_size)

                    # draw rectangles on the displayed image to indicate scanning process
                    painter = QtGui.QPainter(self.image_pixmap)
                    # draw the rectangles
                    cover_color = QtGui.QColor(65, 65, 65, 225)
                    self.draw_fov_mask(painter, rect_center_x, rect_center_y, display_rect_width, display_rect_height, cover_color)

                    # draw ground truth label on the display image
                    gt_display_center_x = (self.cur_image_label[0, 1] + self.cur_image_label[0, 3]) / 2 * (display_rect_width/self.image_size) + (rect_center_x - display_rect_width/2)
                    gt_display_center_y = (self.cur_image_label[0, 2] + self.cur_image_label[0, 4]) / 2 * (display_rect_height/self.image_size) + (rect_center_y - display_rect_height/2)
                    gt_display_rect_width = (self.cur_image_label[0, 3] - self.cur_image_label[0, 1]) * (display_rect_width/self.image_size)
                    gt_display_rect_height = (self.cur_image_label[0, 4] - self.cur_image_label[0, 2]) * (display_rect_height/self.image_size)
                    self.draw_rectangle(painter, gt_display_center_x, gt_display_center_y, gt_display_rect_width, gt_display_rect_height, color='yellow', label='Ground Truth')
                    painter.end()

                    # update pixmap with the label
                    self.image_label.setPixmap(self.image_pixmap)

                    # force repaint
                    self.image_label.repaint()

                # save to cache
                else:
                    self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_{self.image_index}', content=self.all_quantities_1)
                    self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_{self.image_index}', content=self.all_quantities_2)

            # when this is in aggregate mode, all the computations have been done
            elif self.data_mode == 'aggregate':
                # all the label paths
                cur_label_path = self.all_labels_paths[self.image_index]
                # load the label of the current selected image
                self.loaded_image_label = np.load(cur_label_path)
                self.cur_image_label = np.zeros((len(self.loaded_image_label), 6))
                for i in range(len(self.cur_image_label)):
                    # object index
                    self.cur_image_label[i, 0] = i
                    # since PyTorch FasterRCNN has 0 as background
                    self.cur_image_label[i, 5] = self.custom_coco_names.index(self.original_coco_names[int(self.loaded_image_label[i, -1])]) + 1
                    # modify the label accordingly
                    self.cur_image_label[i, 1:5] = self.loaded_image_label[i, :4]

                # information needed for interactive display
                # bounding box center of the key object
                self.center_x = int((self.cur_image_label[0, 1] + self.cur_image_label[0, 3]) // 2)
                self.center_y = int((self.cur_image_label[0, 2] + self.cur_image_label[0, 4]) // 2)
                # just need information for interactive display
                self.x_min = self.center_x - self.image_size//2
                self.x_max = self.center_x + self.image_size//2
                self.y_min = self.center_y - self.image_size//2
                self.y_max = self.center_y + self.image_size//2
                # no transformation to start
                self.cur_x_tran = self.image_size//2
                self.cur_y_tran = self.image_size//2

                # display the image and plot mask + ground truth
                self.display_image()

                display_rect_width = self.display_image_size/2
                display_rect_height = self.display_image_size/2
                # since the translation measures on the movement of object instead of the point of view, the sign is reversed
                rect_center_x = self.display_image_size/2
                rect_center_y = self.display_image_size/2
                # draw rectangles on the displayed image to indicate scanning process
                painter = QtGui.QPainter(self.image_pixmap)
                # draw the rectangles
                cover_color = QtGui.QColor(65, 65, 65, 225)
                self.draw_fov_mask(painter, rect_center_x, rect_center_y, display_rect_width, display_rect_height, cover_color)

                # re-compute the ground truth label bounding boxes of the cropped image
                for i in range(len(self.cur_image_label)):
                    self.cur_image_label[i, 1:5] = self.compute_label(self.loaded_image_label[i, :4], self.x_min, self.y_min, (self.image_size, self.image_size))

                # draw the ground truth label
                gt_display_center_x = (self.cur_image_label[0, 1] + self.cur_image_label[0, 3]) / 2 * (self.display_image_size/self.uncropped_image_size) + (rect_center_x - display_rect_width/2)
                gt_display_center_y = (self.cur_image_label[0, 4] + self.cur_image_label[0, 2]) / 2 * (self.display_image_size/self.uncropped_image_size) + (rect_center_y - display_rect_height/2)
                gt_display_rect_width = (self.cur_image_label[0, 3] - self.cur_image_label[0, 1]) * (self.display_image_size/self.uncropped_image_size)
                gt_display_rect_height = (self.cur_image_label[0, 4] - self.cur_image_label[0, 2]) * (self.display_image_size/self.uncropped_image_size)
                self.draw_rectangle(painter, gt_display_center_x, gt_display_center_y, gt_display_rect_width, gt_display_rect_height, color='yellow', label='Ground Truth')
                painter.end()

                # update pixmap with the label
                self.image_label.setPixmap(self.image_pixmap)

                # force repaint
                self.image_label.repaint()

            # display the individual NERO plot
            self.display_coco_single_result()

        elif self.mode == 'piv':
            # button clicked counter
            self.cur_rotation_angle = 0
            self.is_flipped = 0
            self.is_time_reversed = 0

            # modify the gif as user rotates, flips or time-reverses
            def modify_display_gif():
                display_image_1_pt = self.loaded_image_1_pt.clone()
                display_image_2_pt = self.loaded_image_2_pt.clone()

                if self.cur_rotation_angle != 0:
                    display_image_1_pt = torch.rot90(display_image_1_pt, self.cur_rotation_angle//90)
                    display_image_2_pt = torch.rot90(display_image_2_pt, self.cur_rotation_angle//90)

                if self.is_flipped:
                    display_image_1_pt = torch.flip(display_image_1_pt, [0])
                    display_image_2_pt = torch.flip(display_image_2_pt, [0])

                if self.is_time_reversed:
                    temp = display_image_1_pt.clone()
                    display_image_1_pt = display_image_2_pt.clone()
                    display_image_2_pt = temp

                # create new GIF
                display_image_1_pil = Image.fromarray(display_image_1_pt.numpy(), 'RGB')
                display_image_2_pil = Image.fromarray(display_image_2_pt.numpy(), 'RGB')
                other_images_pil = [display_image_1_pil, display_image_2_pil, display_image_2_pil, self.blank_image_pil]
                self.gif_path = os.path.join(self.cache_dir, self.loaded_image_1_name.split('.')[0] + '.gif')
                display_image_1_pil.save(fp=self.gif_path,
                                            format='GIF',
                                            append_images=other_images_pil,
                                            save_all=True,
                                            duration=600,
                                            loop=0)

                # compute the triangle index by comparing current matrix and the D4 orbit matrix
                for i, cur_image_1_pt in enumerate(self.all_images_1_pt):
                    print(i, torch.eq(display_image_1_pt, cur_image_1_pt))
                    if torch.eq(display_image_1_pt, cur_image_1_pt).all():
                        self.triangle_index = i
                        print('matched', self.triangle_index)
                        break
                exit()
                print(self.triangle_index)


            @QtCore.Slot()
            def rotate_90_ccw():
                self.cur_rotation_angle = (self.cur_rotation_angle + 90) % 360
                print(f'Cur rotation angle: {self.cur_rotation_angle}')
                # modify the image, display and current triangle index
                modify_display_gif()
                self.display_image()
                # showing indication on the individual NERO plot
                # if -self.cur_rotation_angle//90 < 0:
                #     rot_index = 4 - self.cur_rotation_angle//90
                # else:
                #     rot_index = -self.cur_rotation_angle//90
                # self.triangle_index = self.is_time_reversed*8 + self.is_flipped*4 + rot_index
                # print(f'triangle index {self.triangle_index}, nero number {self.cur_plot_quantity_1[self.triangle_index]}')
                # redraw the nero plot with new triangle display
                self.draw_d4_nero('single')
                # update detailed plot of PIV
                self.draw_piv_details()

            @QtCore.Slot()
            def rotate_90_cw():
                self.cur_rotation_angle = (self.cur_rotation_angle - 90) % 360
                print(f'Cur rotation angle: {self.cur_rotation_angle}')
                # modify the image and display
                modify_display_gif()
                self.display_image()
                # showing indication on the individual NERO plot
                # if -self.cur_rotation_angle//90 < 0:
                #     rot_index = 4 - self.cur_rotation_angle//90
                # else:
                #     rot_index = -self.cur_rotation_angle//90
                # self.triangle_index = self.is_time_reversed*8 + self.is_flipped*4 + rot_index
                # print(f'triangle index {self.triangle_index}, nero number {self.cur_plot_quantity_1[self.triangle_index]}')
                # redraw the nero plot with new triangle display
                self.draw_d4_nero('single')
                # update detailed plot of PIV
                self.draw_piv_details()

            @QtCore.Slot()
            def vertical_flip():
                # self.is_flipped = not self.is_flipped
                print(f'Vertical flipped')
                # modify the image and display
                modify_display_gif()
                self.display_image()
                # showing indication on the individual NERO plot
                # if -self.cur_rotation_angle//90 < 0:
                #     rot_index = 4 - self.cur_rotation_angle//90
                # else:
                #     rot_index = -self.cur_rotation_angle//90
                # self.triangle_index = self.is_time_reversed*8 + self.is_flipped*4 + rot_index
                # print(f'triangle index {self.triangle_index}, nero number {self.cur_plot_quantity_1[self.triangle_index]}')
                # redraw the nero plot with new triangle display
                self.draw_d4_nero('single')
                # update detailed plot of PIV
                self.draw_piv_details()

            @QtCore.Slot()
            def horizontal_flip():
                # self.is_flipped = not self.is_flipped
                print(f'Horizontal flipped')
                # modify the image and display
                modify_display_gif()
                self.display_image()
                # showing indication on the individual NERO plot
                # if -self.cur_rotation_angle//90 < 0:
                #     rot_index = 4 - self.cur_rotation_angle//90
                # else:
                #     rot_index = -self.cur_rotation_angle//90
                # self.triangle_index = self.is_time_reversed*8 + self.is_flipped*4 + rot_index
                # print(f'triangle index {self.triangle_index}, nero number {self.cur_plot_quantity_1[self.triangle_index]}')
                # redraw the nero plot with new triangle display
                self.draw_d4_nero('single')
                # update detailed plot of PIV
                self.draw_piv_details()

            @QtCore.Slot()
            def time_reverse():
                self.is_time_reversed = not self.is_time_reversed
                print(f'Image time-reversed: {self.is_time_reversed}')
                # modify the image and display
                modify_display_gif()
                self.display_image()
                # showing indication on the individual NERO plot
                # if -self.cur_rotation_angle//90 < 0:
                #     rot_index = 4 - self.cur_rotation_angle//90
                # else:
                #     rot_index = -self.cur_rotation_angle//90
                # self.triangle_index = self.is_time_reversed*8 + self.is_flipped*4 + rot_index
                # print(f'triangle index {self.triangle_index}, nero number {self.cur_plot_quantity_1[self.triangle_index]}')
                # redraw the nero plot with new triangle display
                self.draw_d4_nero('single')
                # update detailed plot of PIV
                self.draw_piv_details()


            # add buttons for controlling the GIF
            self.gif_control_layout = QtWidgets.QHBoxLayout()
            self.gif_control_layout.setAlignment(QtGui.Qt.AlignTop)
            self.gif_control_layout.setContentsMargins(50, 0, 50, 50)
            if self.data_mode == 'single':
                self.single_result_layout.addLayout(self.gif_control_layout, 2, 0)

            # rotate 90 degrees counter-closewise
            self.rotate_90_ccw_button = QtWidgets.QPushButton(self)
            self.rotate_90_ccw_button.setFixedSize(QtCore.QSize(50, 50))
            self.rotate_90_ccw_button.setIcon(QtGui.QIcon('symbols/rotate_90_ccw.png'))
            self.rotate_90_ccw_button.setIconSize(QtCore.QSize(40, 40))
            self.rotate_90_ccw_button.clicked.connect(rotate_90_ccw)
            self.gif_control_layout.addWidget(self.rotate_90_ccw_button)

            # rotate 90 degrees closewise
            self.rotate_90_cw_button = QtWidgets.QPushButton(self)
            self.rotate_90_cw_button.setFixedSize(QtCore.QSize(50, 50))
            self.rotate_90_cw_button.setIcon(QtGui.QIcon('symbols/rotate_90_cw.png'))
            self.rotate_90_cw_button.setIconSize(QtCore.QSize(40, 40))
            self.rotate_90_cw_button.clicked.connect(rotate_90_cw)
            self.gif_control_layout.addWidget(self.rotate_90_cw_button)

            # flip the iamge vertically (by x axis)
            self.vertical_flip_button = QtWidgets.QPushButton(self)
            self.vertical_flip_button.setFixedSize(QtCore.QSize(50, 50))
            self.vertical_flip_button.setIcon(QtGui.QIcon('symbols/vertical_flip.png'))
            self.vertical_flip_button.setIconSize(QtCore.QSize(40, 40))
            self.vertical_flip_button.clicked.connect(vertical_flip)
            self.gif_control_layout.addWidget(self.vertical_flip_button)

            # flip the iamge horizontally (by y axis)
            self.horizontal_flip_button = QtWidgets.QPushButton(self)
            self.horizontal_flip_button.setFixedSize(QtCore.QSize(50, 50))
            self.horizontal_flip_button.setIcon(QtGui.QIcon('symbols/horizontal_flip.png'))
            self.horizontal_flip_button.setIconSize(QtCore.QSize(40, 40))
            self.horizontal_flip_button.clicked.connect(horizontal_flip)
            self.gif_control_layout.addWidget(self.horizontal_flip_button)

            # time reverse
            self.time_reverse_button = QtWidgets.QPushButton(self)
            self.time_reverse_button.setFixedSize(QtCore.QSize(50, 50))
            self.time_reverse_button.setIcon(QtGui.QIcon('symbols/time_reverse.png'))
            self.time_reverse_button.setIconSize(QtCore.QSize(40, 40))
            self.time_reverse_button.clicked.connect(time_reverse)
            self.gif_control_layout.addWidget(self.time_reverse_button)

            # Dihedral group4 transformations
            all_rotation_degrees = [0, 90, 180, 270]
            # 0 means no flip/time reverse, 1 means flip/time reverse
            all_flip = [0, 1]
            all_time_reversals = [0, 1]
            self.num_transformations = len(all_rotation_degrees) * len(all_flip) * len(all_time_reversals)

            if self.data_mode == 'single':
                # keep track for all transformation's output
                self.all_images_1_pt = torch.zeros((self.num_transformations, self.image_size, self.image_size, 3))
                self.all_images_2_pt = torch.zeros((self.num_transformations, self.image_size, self.image_size, 3))
                transformation_index = 0
                for is_time_reversed in all_time_reversals:
                        for is_flipped in all_flip:
                            for cur_rot_degree in all_rotation_degrees:
                                print(f'Transformation {transformation_index}')
                                # modify the input image tensor alone with its ground truth
                                # rotation
                                if cur_rot_degree:
                                    self.cur_image_1_pt, \
                                    self.cur_image_2_pt, \
                                    self.cur_image_label_pt = nero_transform.rotate_piv_data(self.loaded_image_1_pt, self.loaded_image_2_pt, self.loaded_image_label_pt, cur_rot_degree)

                                # flip
                                elif is_flipped:
                                    self.cur_image_1_pt, \
                                    self.cur_image_2_pt, \
                                    self.cur_image_label_pt = nero_transform.flip_piv_data(self.loaded_image_1_pt, self.loaded_image_2_pt, self.loaded_image_label_pt)

                                # time reverse
                                elif is_time_reversed:
                                    self.cur_image_1_pt, \
                                    self.cur_image_2_pt, \
                                    self.cur_image_label_pt = nero_transform.reverse_piv_data(self.loaded_image_1_pt, self.loaded_image_2_pt, self.loaded_image_label_pt)

                                # no transformation at all
                                else:
                                    self.cur_image_1_pt = self.loaded_image_1_pt
                                    self.cur_image_2_pt = self.loaded_image_2_pt
                                    self.cur_image_label_pt = self.loaded_image_label_pt

                                # keep track of the different tranformed image and ground truths
                                self.all_images_1_pt[transformation_index] = self.cur_image_1_pt
                                self.all_images_2_pt[transformation_index] = self.cur_image_2_pt

                                transformation_index += 0

                # all_quantities has shape (16, 256, 256, 2)
                if self.use_cache:
                    self.all_quantities_1 = torch.from_numpy(self.load_from_cache(name=f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_{self.image_index}'))
                    self.all_quantities_2 = torch.from_numpy(self.load_from_cache(name=f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_{self.image_index}'))
                    self.all_ground_truths = torch.from_numpy(self.load_from_cache(name=f'{self.mode}_{self.data_mode}_{self.image_index}_ground_truths'))
                else:
                    # each model output are dense 2D velocity field of the input image
                    self.all_quantities_1 = torch.zeros((self.num_transformations, self.image_size, self.image_size, 2))
                    self.all_quantities_2 = torch.zeros((self.num_transformations, self.image_size, self.image_size, 2))
                    self.all_ground_truths = torch.zeros((self.num_transformations, self.image_size, self.image_size, 2))

                    # keep track for all transformation's output
                    self.all_images_1_pt = torch.zeros((self.num_transformations, self.image_size, self.image_size, 3))
                    self.all_images_2_pt = torch.zeros((self.num_transformations, self.image_size, self.image_size, 3))
                    transformation_index = 0

                    for is_time_reversed in all_time_reversals:
                        for is_flipped in all_flip:
                            for cur_rot_degree in all_rotation_degrees:
                                print(f'Transformation {transformation_index}')
                                # modify the input image tensor alone with its ground truth
                                # rotation
                                if cur_rot_degree:
                                    self.cur_image_1_pt, \
                                    self.cur_image_2_pt, \
                                    self.cur_image_label_pt = nero_transform.rotate_piv_data(self.loaded_image_1_pt, self.loaded_image_2_pt, self.loaded_image_label_pt, cur_rot_degree)

                                # flip
                                elif is_flipped:
                                    self.cur_image_1_pt, \
                                    self.cur_image_2_pt, \
                                    self.cur_image_label_pt = nero_transform.flip_piv_data(self.loaded_image_1_pt, self.loaded_image_2_pt, self.loaded_image_label_pt)

                                # time reverse
                                elif is_time_reversed:
                                    self.cur_image_1_pt, \
                                    self.cur_image_2_pt, \
                                    self.cur_image_label_pt = nero_transform.reverse_piv_data(self.loaded_image_1_pt, self.loaded_image_2_pt, self.loaded_image_label_pt)

                                # no transformation at all
                                else:
                                    self.cur_image_1_pt = self.loaded_image_1_pt
                                    self.cur_image_2_pt = self.loaded_image_2_pt
                                    self.cur_image_label_pt = self.loaded_image_label_pt

                                # keep track of the different tranformed image and ground truths
                                self.all_images_1_pt[transformation_index] = self.cur_image_1_pt
                                self.all_images_2_pt[transformation_index] = self.cur_image_2_pt
                                self.all_ground_truths[transformation_index] = self.cur_image_label_pt

                                # run the model
                                quantity_1 = nero_run_model.run_piv_once('single',
                                                                            self.model_1_name,
                                                                            self.model_1,
                                                                            self.cur_image_1_pt,
                                                                            self.cur_image_2_pt)

                                quantity_2 = nero_run_model.run_piv_once('single',
                                                                            self.model_2_name,
                                                                            self.model_2,
                                                                            self.cur_image_1_pt,
                                                                            self.cur_image_2_pt)

                                self.all_quantities_1[transformation_index] = quantity_1 / self.image_size
                                self.all_quantities_2[transformation_index] = quantity_2 / self.image_size

                                transformation_index += 1

                    # save to cache
                    self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_1_cache_name}_{self.image_index}', content=self.all_quantities_1.numpy())
                    self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.model_2_cache_name}_{self.image_index}', content=self.all_quantities_2.numpy())
                    self.save_to_cache(name=f'{self.mode}_{self.data_mode}_{self.image_index}_ground_truths', content=self.all_ground_truths.numpy())

                # display the piv single case result
                self.triangle_index = 0
                self.display_piv_single_result()

            elif self.data_mode == 'aggregate':
                raise NotImplementedError


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
    def draw_rectangle(self, painter, center_x, center_y, width, height, color=None, alpha=255, fill=None, boundary_width=5, label=None):
        if center_x == 0 and center_y == 0 and width == 0 and height == 0:
            return

        # left, top, width, height for QRect
        rectangle = QtCore.QRect(center_x-width//2, center_y-height//2, width, height)

        if color:
            pen = QtGui.QPen()
            pen.setWidth(boundary_width)
            pen_color = QtGui.QColor(color)
            pen_color.setAlpha(alpha)
            pen.setColor(pen_color)
            painter.setPen(pen)
            painter.drawRect(rectangle)

        if fill:
            brush = QtGui.QBrush()
            brush.setColor(fill)
            brush.setStyle(QtCore.Qt.SolidPattern)
            # painter.setBrush(brush)
            painter.fillRect(rectangle, brush)

        if label:
            # label background area
            text_rect = QtCore.QRect(center_x-width//2, center_y-height//2-20, 100, 20)
            brush = QtGui.QBrush()
            brush.setStyle(QtCore.Qt.SolidPattern)
            brush.setColor(color)
            painter.fillRect(text_rect, brush)
            # black text
            pen = QtGui.QPen()
            pen.setColor(QtGui.QColor('black'))
            painter.setPen(pen)
            painter.drawText(text_rect, QtGui.Qt.AlignCenter, label)


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
        self.circle_1.setPen(pg.mkPen('blue', width=7))
        self.polar_plot.addItem(self.circle_1)

        # transform to x and y coordinate
        cur_quantity_2_x = self.output_2[self.loaded_image_label] * np.cos(self.cur_rotation_angle/180*np.pi)
        cur_quantity_2_y = self.output_2[self.loaded_image_label] * np.sin(self.cur_rotation_angle/180*np.pi)
        # plot a circle item
        self.circle_2 = pg.QtGui.QGraphicsEllipseItem(cur_quantity_2_x-r/2, cur_quantity_2_y-r/2, r, r)
        self.circle_2.setPen(pg.mkPen('magenta', width=7))
        self.polar_plot.addItem(self.circle_2)


    # draw detailed look of COCO models output on cropped regions
    def draw_model_output(self):
        def draw_detailed_plot(detailed_display_image, model_output, color):

            # prepare a pixmap for the image
            detailed_image_pixmap = QPixmap(detailed_display_image)

            # add a new label for loadeds image
            detailed_image_label = QLabel(self)
            detailed_image_label.setFixedSize(self.plot_size+20, self.plot_size)
            # left top right bottom
            detailed_image_label.setContentsMargins(20, 0, 0, 0)
            detailed_image_label.setAlignment(QtCore.Qt.AlignCenter)

            # draw bounding boxes on the enlarged view
            # draw ground truth
            painter = QtGui.QPainter(detailed_image_pixmap)
            # draw the ground truth label
            gt_display_center_x = (self.cur_image_label[0, 1] + self.cur_image_label[0, 3]) // 2 * (self.plot_size/self.image_size)
            gt_display_center_y = (self.cur_image_label[0, 2] + self.cur_image_label[0, 4]) // 2 * (self.plot_size/self.image_size)
            gt_display_rect_width = (self.cur_image_label[0, 3] - self.cur_image_label[0, 1]) * (self.plot_size/self.image_size)
            gt_display_rect_height = (self.cur_image_label[0, 4] - self.cur_image_label[0, 2]) * (self.plot_size/self.image_size)
            self.draw_rectangle(painter, gt_display_center_x, gt_display_center_y, gt_display_rect_width, gt_display_rect_height, color='yellow', alpha=166, label='Ground Truth')

            # box from model 1
            bounding_boxes = model_output[0][0][:, :4]
            confidences = model_output[0][0][:, 4]
            ious = model_output[0][0][:, 6]
            # showing a maximum of 3 bounding boxes
            num_boxes_1 = min(3, len(bounding_boxes))
            for i in range(num_boxes_1):
                center_x = (bounding_boxes[i, 0] + bounding_boxes[i, 2]) // 2 * (self.plot_size/self.image_size)
                center_y = (bounding_boxes[i, 1] + bounding_boxes[i, 3]) // 2 * (self.plot_size/self.image_size)
                model_display_rect_width = (bounding_boxes[i, 2] - bounding_boxes[i, 0]) * (self.plot_size/self.image_size)
                model_display_rect_height = (bounding_boxes[i, 3] - bounding_boxes[i, 1]) * (self.plot_size/self.image_size)

                # compute alpha value based on confidence
                cur_alpha = nero_utilities.lerp(confidences[i], 0, 1, 255/4, 255)
                # compute boundary width based on IOU
                cur_boundary_width = nero_utilities.lerp(ious[i], 0, 1, 2, 5)

                self.draw_rectangle(painter, center_x, center_y, model_display_rect_width, model_display_rect_height, color, alpha=cur_alpha, boundary_width=cur_boundary_width, label=f'Prediction {i+1}')

            painter.end()

            # put pixmap in the label
            detailed_image_label.setPixmap(detailed_image_pixmap)

            # force repaint
            detailed_image_label.repaint()

            # detailed information showed beneath the image
            # add a new label for text
            detailed_text_label = QLabel(self)
            detailed_text_label.setFixedSize(self.plot_size+20, 100)
            # left top right bottom
            detailed_text_label.setContentsMargins(20, 0, 0, 0)
            detailed_text_label.setAlignment(QtCore.Qt.AlignTop)
            # font and size
            detailed_text_label.setFont(QFont('Helvetica', 12))

            # display_text = f'Ground Truth: {self.custom_coco_names[int(self.loaded_image_label[0][4])]}\n'
            display_text = ''
            for i in range(num_boxes_1):
                display_text += f'Prediction {i+1}: {self.custom_coco_names[int(model_output[0][0][i, 5]-1)]}, Conf: {model_output[0][0][i, 4]:.3f}, IOU: {model_output[0][0][i, 6]:.3f}\n'
            display_text += '\n'

            detailed_text_label.setText(display_text)

            return detailed_image_label, detailed_text_label

        # size of the enlarged image
        # convert and resize current selected FOV to QImage for display purpose
        self.detailed_display_image = nero_utilities.tensor_to_qt_image(self.loaded_image_pt[self.y_min:self.y_max, self.x_min:self.x_max, :]).scaledToWidth(self.plot_size)
        # run model with the cropped view
        self.cropped_image_pt = self.loaded_image_pt[self.y_min:self.y_max, self.x_min:self.x_max, :] / 255
        self.run_model_once()

        # display for model 1
        self.detailed_image_label_1, self.detailed_text_label_1 = draw_detailed_plot(self.detailed_display_image, self.output_1, 'blue')
        # display for model 2
        self.detailed_image_label_2, self.detailed_text_label_2 = draw_detailed_plot(self.detailed_display_image, self.output_2, 'magenta')
        # spacer item between image and text
        image_text_spacer = QtWidgets.QSpacerItem(self.plot_size, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        if self.data_mode == 'single':
            self.single_result_layout.addWidget(self.detailed_image_label_1, 3, 1)
            self.single_result_layout.addItem(image_text_spacer, 4, 2)
            self.single_result_layout.addWidget(self.detailed_text_label_1, 4, 1)
            self.single_result_layout.addWidget(self.detailed_image_label_2, 3, 2)
            self.single_result_layout.addWidget(self.detailed_text_label_2, 4, 2)
        elif self.data_mode == 'aggregate':
            self.aggregate_result_layout.addWidget(self.detailed_image_label_1, 2, 4)
            self.aggregate_result_layout.addItem(image_text_spacer, 3, 5)
            self.aggregate_result_layout.addWidget(self.detailed_text_label_1, 3, 4)
            self.aggregate_result_layout.addWidget(self.detailed_image_label_2, 2, 5)
            self.aggregate_result_layout.addWidget(self.detailed_text_label_2, 3, 5)


    def display_image(self):
        print('displaying image')
        # add a new label for loaded image if no imager has existed
        if not self.image_existed:
            self.image_label = QLabel(self)
            self.image_label.setAlignment(QtCore.Qt.AlignCenter)
            self.image_existed = True
            # no additional content margin to prevent cutoff on images
            self.image_label.setContentsMargins(0, 0, 0, 0)

            # add the image label to the layout
            if self.data_mode == 'single':
                self.single_result_layout.addWidget(self.image_label, 1, 0)

            elif self.data_mode == 'aggregate':
                if self.mode == 'digit_recognition':
                    self.aggregate_result_layout.addWidget(self.image_label, 1, 4, 2, 1)
                elif self.mode == 'object_detection':
                    self.aggregate_result_layout.addWidget(self.image_label, 1, 3, 3, 1)
                elif self.mode == 'piv':
                    self.aggregate_result_layout.addWidget(self.image_label, 1, 3, 3, 1)


        if self.mode == 'digit_recognition' or self.mode == 'object_detection':

            # prepare a pixmap for the image
            self.image_pixmap = QPixmap(self.cur_display_image)

            # single pixmap in the label
            self.image_label.setPixmap(self.image_pixmap)

            if self.mode == 'digit_recognition':
                # plot_size should be bigger than the display_size, so that some margins exist
                self.image_label.setFixedSize(self.plot_size, self.plot_size)
            elif self.mode == 'object_detection':
                # set label to the size of pixmap so that when clicked it is wrt image
                self.image_label.setFixedSize(self.image_pixmap.size())

                # pixel mouse over for object detection mode
                def start_moving(event):
                    self.moving_pov = True

                def update_model_pov(event):
                    if self.moving_pov:
                        rect_center_x = int(event.position().x())
                        rect_center_y = int(event.position().y())

                        # when the click selection is valid and model has been run before
                        if self.single_result_existed:

                            # re-load clean image
                            self.image_pixmap = QPixmap(self.cur_display_image)
                            self.image_label.setPixmap(self.image_pixmap)

                            # draw the new FOV rectangle
                            # width and height of the rectangle
                            display_rect_width = self.display_image_size//2
                            display_rect_height = self.display_image_size//2

                            # restrict x and y value
                            if rect_center_x + display_rect_width//2 >= self.display_image_size:
                                rect_center_x = self.display_image_size - display_rect_width//2
                            elif rect_center_x - display_rect_width//2 < 0:
                                rect_center_x = display_rect_width//2

                            if rect_center_y + display_rect_height//2 >= self.display_image_size:
                                rect_center_y = self.display_image_size - display_rect_height//2
                            elif rect_center_y - display_rect_height//2 < 0:
                                rect_center_y = display_rect_height//2

                            # draw rectangle on the displayed image to indicate scanning process
                            painter = QtGui.QPainter(self.image_pixmap)
                            # draw the rectangles
                            cover_color = QtGui.QColor(65, 65, 65, 225)
                            self.draw_fov_mask(painter, rect_center_x, rect_center_y, display_rect_width, display_rect_height, cover_color)

                            # how much the clicked point is away from the image center
                            x_dist = (rect_center_x - self.display_image_size/2) / (self.display_image_size/self.uncropped_image_size)
                            y_dist = (rect_center_y - self.display_image_size/2) / (self.display_image_size/self.uncropped_image_size)
                            # compute rectangle center wrt to the original image
                            cur_center_x = self.center_x + x_dist
                            cur_center_y = self.center_y + y_dist
                            self.x_min = int(cur_center_x - display_rect_width/2)
                            self.x_max = int(cur_center_x + display_rect_width/2)
                            self.y_min = int(cur_center_y - display_rect_height/2)
                            self.y_max = int(cur_center_y + display_rect_height/2)

                            # re-compute the ground truth label bounding boxes of the cropped image
                            for i in range(len(self.cur_image_label)):
                                self.cur_image_label[i, 1:5] = self.compute_label(self.loaded_image_label[i, :4], self.x_min, self.y_min, (self.image_size, self.image_size))

                            # draw the ground truth label
                            gt_display_center_x = (self.cur_image_label[0, 1] + self.cur_image_label[0, 3]) / 2 * (self.display_image_size/self.uncropped_image_size) + (rect_center_x - display_rect_width/2)
                            gt_display_center_y = (self.cur_image_label[0, 4] + self.cur_image_label[0, 2]) / 2 * (self.display_image_size/self.uncropped_image_size) + (rect_center_y - display_rect_height/2)
                            gt_display_rect_width = (self.cur_image_label[0, 3] - self.cur_image_label[0, 1]) * (self.display_image_size/self.uncropped_image_size)
                            gt_display_rect_height = (self.cur_image_label[0, 4] - self.cur_image_label[0, 2]) * (self.display_image_size/self.uncropped_image_size)
                            self.draw_rectangle(painter, gt_display_center_x, gt_display_center_y, gt_display_rect_width, gt_display_rect_height, color='yellow', label='Ground Truth')
                            painter.end()

                            # update pixmap with the label
                            self.image_label.setPixmap(self.image_pixmap)

                            # force repaint
                            self.image_label.repaint()

                            # show corresponding translation amount on the heatmap
                            # translation amout for plotting in heatmap
                            self.cur_x_tran = -x_dist - self.x_translation[0]
                            self.cur_y_tran = y_dist - self.y_translation[0]
                            self.draw_heatmaps(mode='single')

                            # run inference only when in the realtime mode
                            if self.realtime_inference:
                                # redisplay model output
                                self.draw_model_output()

                def end_moving(event):
                    self.moving_pov = False
                    # when not realtime-inferencing, run once after stop moving
                    if not self.realtime_inference:
                        # redisplay model output
                        self.draw_model_output()

                self.image_label.mousePressEvent = start_moving
                self.image_label.mouseMoveEvent = update_model_pov
                self.image_label.mouseReleaseEvent = end_moving

        elif self.mode == 'piv':

            # plot_size should be bigger than the display_size, so that some margins exist
            self.image_label.setFixedSize(self.plot_size, self.plot_size)
            image_gif = QtGui.QMovie(self.gif_path)
            # add to the label
            self.image_label.setMovie(image_gif)
            image_gif.start()


    # helper function on drawing mask on input COCO image (to highlight the current FOV)
    def draw_fov_mask(self, painter, rect_center_x, rect_center_y, display_rect_width, display_rect_height, cover_color):
        # draw the rectangles
        # top
        top_rect_center_x = (0 + self.display_image_size) / 2
        top_rect_center_y = (0 + rect_center_y - display_rect_height/2) / 2
        top_display_rect_width = self.display_image_size
        top_display_rect_height = top_rect_center_y * 2
        self.draw_rectangle(painter, top_rect_center_x, top_rect_center_y, top_display_rect_width, top_display_rect_height, fill=cover_color)
        # bottom
        bottom_rect_center_x = (0 + self.display_image_size) / 2
        bottom_rect_center_y = (rect_center_y + display_rect_height/2 + self.display_image_size) / 2
        bottom_display_rect_width = self.display_image_size
        bottom_display_rect_height = (self.display_image_size-bottom_rect_center_y) * 2
        self.draw_rectangle(painter, bottom_rect_center_x, bottom_rect_center_y, bottom_display_rect_width, bottom_display_rect_height, fill=cover_color)
        # left
        left_rect_center_x = (0 + rect_center_x - display_rect_width/2) / 2
        left_rect_center_y = rect_center_y
        left_display_rect_width = rect_center_x - display_rect_width/2
        left_display_rect_height = self.display_image_size-top_display_rect_height-bottom_display_rect_height
        self.draw_rectangle(painter, left_rect_center_x, left_rect_center_y, left_display_rect_width, left_display_rect_height, fill=cover_color)
        # right
        right_rect_center_x = (rect_center_x + display_rect_width/2 + self.display_image_size) / 2
        right_rect_center_y = rect_center_y
        right_display_rect_width = self.display_image_size - (rect_center_x + display_rect_width/2)
        right_display_rect_height = self.display_image_size-top_display_rect_height-bottom_display_rect_height
        self.draw_rectangle(painter, right_rect_center_x, right_rect_center_y, right_display_rect_width, right_display_rect_height, fill=cover_color)


    # helper function on drawing individual heatmap (called by both individual and aggregate cases)
    def draw_individual_heatmap(self, mode, data, title=None, range=(0, 1)):
        # actuall heatmap
        heatmap = pg.ImageItem()
        heatmap.setOpts(axisOrder='row-major')
        heatmap.setImage(data)

        # create view box to contain the heatmap
        view_box = pg.ViewBox()
        view_box.setAspectLocked(lock=True)
        view_box.addItem(heatmap)

        if self.mode == 'object_detection' and mode == 'single':
            # small indicator on where the translation is at
            scatter_item = pg.ScatterPlotItem(pxMode=False)
            scatter_point = []

            scatter_point.append({'pos': (self.cur_x_tran, self.cur_y_tran),
                                    'size': 3,
                                    'pen': {'color': 'red', 'width': 0.1},
                                    'brush': (255, 0, 0, 255)})

            # add points to the item
            scatter_item.addPoints(scatter_point)
            view_box.addItem(scatter_item)

        heatmap_plot = pg.PlotItem(viewBox=view_box, title=title)
        if self.mode == 'object_detection':
            heatmap_plot.getAxis('bottom').setLabel('Translation in x')
            heatmap_plot.getAxis('bottom').setStyle(tickLength=0, showValues=False)
            heatmap_plot.getAxis('left').setLabel('Translation in y')
            heatmap_plot.getAxis('left').setStyle(tickLength=0, showValues=False)
        elif self.mode == 'piv':
            heatmap_plot.getAxis('bottom').setLabel('x')
            heatmap_plot.getAxis('bottom').setStyle(tickLength=0, showValues=False)
            heatmap_plot.getAxis('left').setLabel('y')
            heatmap_plot.getAxis('left').setStyle(tickLength=0, showValues=False)

        # disable being able to move plot around
        heatmap_plot.setMouseEnabled(x=False, y=False)

        # create colorbar
        color_map = pg.colormap.get('viridis')
        color_bar = pg.ColorBarItem(values=range, colorMap=color_map)
        color_bar.setImageItem(heatmap, insert_in=heatmap_plot)

        return heatmap_plot


    # draw heatmaps that displays the individual NERO plots
    def draw_heatmaps(self, mode):

        # helper function for clicking inside the heatmap
        def heatmap_mouse_clicked(event, heatmap_plot):
            self.heatmap_clicked = not self.heatmap_clicked
            heatmap_plot.scene().items(event.scenePos())
            # check if the click is within the polar plot
            if heatmap_plot.sceneBoundingRect().contains(event._scenePos):
                mouse_pos_on_heatmap = self.polar_plot.vb.mapSceneToView(event._scenePos)
                x_pos = mouse_pos_on_heatmap.x()
                y_pos = mouse_pos_on_heatmap.y()

                print(x_pos, y_pos)

                # update the model output
                # if self.result_existed:
                #     self.run_model_once()

                # remove old point
                # if self.cur_line:
                #     heatmap_plot.removeItem(self.cur_point)

                # draw a point(rect) that represents current selection of location

        # check if the data is in shape (self.image_size, self.image_size)
        if self.cur_plot_quantity_1.shape != (self.image_size, self.image_size):
            # repeat in row
            temp = np.repeat(self.cur_plot_quantity_1, self.image_size/self.cur_plot_quantity_1.shape[1], axis=0)
            # repeat in column
            data_1 = np.repeat(temp, self.image_size/self.cur_plot_quantity_1.shape[0], axis=1)
        else:
            data_1 = self.cur_plot_quantity_1

        if self.cur_plot_quantity_2.shape != (self.image_size, self.image_size):
            # repeat in row
            temp = np.repeat(self.cur_plot_quantity_2, self.image_size/self.cur_plot_quantity_2.shape[1], axis=0)
            # repeat in column
            data_2 = np.repeat(temp, self.image_size/self.cur_plot_quantity_2.shape[0], axis=1)
        else:
            data_2 = self.cur_plot_quantity_2

        # add to general layout
        if mode == 'single':
            # heatmap view
            self.heatmap_view_1 = pg.GraphicsLayoutWidget()
            # left top right bottom
            self.heatmap_view_1.ci.layout.setContentsMargins(0, 20, 0, 0)
            self.heatmap_view_1.setFixedSize(self.plot_size*1.3, self.plot_size*1.3)
            self.heatmap_view_2 = pg.GraphicsLayoutWidget()
            # left top right bottom
            self.heatmap_view_2.ci.layout.setContentsMargins(0, 20, 0, 0)
            self.heatmap_view_2.setFixedSize(self.plot_size*1.3, self.plot_size*1.3)
            self.heatmap_plot_1 = self.draw_individual_heatmap('single', data_1)
            self.heatmap_plot_2 = self.draw_individual_heatmap('single', data_2)
            # self.view_box_1.scene().sigMouseClicked.connect(heatmap_mouse_clicked(self.view_box_1))
            # self.view_box_2.scene().sigMouseClicked.connect(heatmap_mouse_clicked(self.view_box_2))

            # add to view
            self.heatmap_view_1.addItem(self.heatmap_plot_1)
            self.heatmap_view_2.addItem(self.heatmap_plot_2)

            if self.data_mode == 'single':
                self.single_result_layout.addWidget(self.heatmap_view_1, 1, 1)
                self.single_result_layout.addWidget(self.heatmap_view_2, 1, 2)
            elif self.data_mode == 'aggregate':
                self.aggregate_result_layout.addWidget(self.heatmap_view_1, 1, 4)
                self.aggregate_result_layout.addWidget(self.heatmap_view_2, 1, 5)

        elif mode == 'aggregate':
            # heatmap view
            self.aggregate_heatmap_view_1 = pg.GraphicsLayoutWidget()
            # left top right bottom
            self.aggregate_heatmap_view_1.ci.layout.setContentsMargins(0, 20, 0, 0)
            self.aggregate_heatmap_view_1.setFixedSize(self.plot_size*1.3, self.plot_size*1.3)
            self.aggregate_heatmap_view_2 = pg.GraphicsLayoutWidget()
            # left top right bottom
            self.aggregate_heatmap_view_2.ci.layout.setContentsMargins(0, 20, 0, 0)
            self.aggregate_heatmap_view_2.setFixedSize(self.plot_size*1.3, self.plot_size*1.3)
            self.aggregate_heatmap_plot_1 = self.draw_individual_heatmap('aggregate', data_1)
            self.aggregate_heatmap_plot_2 = self.draw_individual_heatmap('aggregate', data_2)
            # self.view_box_1.scene().sigMouseClicked.connect(heatmap_mouse_clicked(self.view_box_1))
            # self.view_box_2.scene().sigMouseClicked.connect(heatmap_mouse_clicked(self.view_box_2))

            # add to view
            self.aggregate_heatmap_view_1.addItem(self.aggregate_heatmap_plot_1)
            self.aggregate_heatmap_view_2.addItem(self.aggregate_heatmap_plot_2)

            self.aggregate_result_layout.addWidget(self.aggregate_heatmap_view_1, 1, 1)
            self.aggregate_result_layout.addWidget(self.aggregate_heatmap_view_2, 1, 2)


    def draw_aggregate_polar(self):
        # initialize view and plot
        polar_view = pg.GraphicsLayoutWidget()
        polar_view.setBackground('white')
        # polar plot larger than others because it occupies two rows
        polar_view.setFixedSize(self.plot_size*1.7, self.plot_size*1.7)
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
        # plot selected digit's average accuracy/confidence across all rotations
        for i in range(len(self.all_aggregate_angles)):
            radian = self.all_aggregate_angles[i] / 180 * np.pi
            # model 1 accuracy
            if self.quantity_name == 'Accuracy':
                if self.class_selection == 'all':
                    cur_quantity_1 = self.all_avg_accuracy_1[i]
                else:
                    cur_quantity_1 = self.all_avg_accuracy_per_digit_1[i][self.class_selection]
            # elif self.quantity_name == 'Confidence':

            # Transform to cartesian and plot
            x_1 = cur_quantity_1 * np.cos(radian)
            y_1 = cur_quantity_1 * np.sin(radian)
            all_x_1.append(x_1)
            all_y_1.append(y_1)
            all_points_1.append({'pos': (x_1, y_1),
                                'size': 0.05,
                                'pen': {'color': 'w', 'width': 0.1},
                                'brush': QtGui.QColor('blue')})

            # model 2 quantity
            if self.quantity_name == 'Accuracy':
                if self.class_selection == 'all':
                    cur_quantity_2 = self.all_avg_accuracy_2[i]
                else:
                    cur_quantity_2 = self.all_avg_accuracy_per_digit_2[i][self.class_selection]

            # Transform to cartesian and plot
            x_2 = cur_quantity_2 * np.cos(radian)
            y_2 = cur_quantity_2 * np.sin(radian)
            all_x_2.append(x_2)
            all_y_2.append(y_2)
            all_points_2.append({'pos': (x_2, y_2),
                                'size': 0.05,
                                'pen': {'color': 'w', 'width': 0.1},
                                'brush': QtGui.QColor('magenta')})

        # draw lines to better show shape
        line_1 = self.aggregate_polar_plot.plot(all_x_1, all_y_1, pen = QtGui.QPen(QtGui.Qt.blue, 0.03))
        line_2 = self.aggregate_polar_plot.plot(all_x_2, all_y_2, pen = QtGui.QPen(QtGui.Qt.magenta, 0.03))

        # add points to the item
        self.aggregate_scatter_items.addPoints(all_points_1)
        self.aggregate_scatter_items.addPoints(all_points_2)

        # add points to the plot
        self.aggregate_polar_plot.addItem(self.aggregate_scatter_items)

        # fix zoom level
        # self.polar_plot.vb.scaleBy((0.5, 0.5))
        self.aggregate_polar_plot.setMouseEnabled(x=False, y=False)

        # add the plot view to the layout
        self.aggregate_result_layout.addWidget(polar_view, 1, 1, 2, 2)


    def draw_triangle(self, painter, points, pen_color=None, brush_color=None, boundary_width=None):

        point_1, point_2, point_3 = points
        # define pen and brush
        pen = QtGui.QPen()
        if pen_color:
            pen.setWidth(boundary_width)
            pen.setColor(pen_color)
        painter.setPen(pen)

        brush = QtGui.QBrush()
        if brush_color:
            brush.setColor(brush_color)
            brush.setStyle(QtCore.Qt.SolidPattern)
        else:
            brush.setStyle(QtCore.Qt.NoBrush)
        painter.setBrush(brush)

        # define triangle
        triangle = QtGui.QPolygonF()
        triangle.append(QtCore.QPointF(point_1))
        triangle.append(QtCore.QPointF(point_2))
        triangle.append(QtCore.QPointF(point_3))

        # draw triangle
        painter.drawPolygon(triangle)


    # draws a single D4 vis
    def draw_individual_d4(self, data):

        # initialize label and its pixmap
        d4_label = QLabel(self)
        d4_label.setContentsMargins(0, 0, 0, 0)
        d4_label.setAlignment(QtCore.Qt.AlignCenter)

        d4_pixmap = QPixmap(self.plot_size, self.plot_size)
        d4_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(d4_pixmap)

        # define a colormap and lookup table to use as fill color
        d4_color_map = pg.colormap.get('viridis')
        d4_lut = d4_color_map.getLookupTable(start=0, stop=1, nPts=101, alpha=False)

        # each d4 NERO plot has 16 triangles
        p = self.plot_size
        # points of all triangles
        all_triangle_points = [[QtCore.QPointF(0, p/2), QtCore.QPointF(p/4, p/4), QtCore.QPointF(p/2, p/2)], # original
                                [QtCore.QPointF(p/2, p/2), QtCore.QPointF(p/2, 0), QtCore.QPointF(p*3/4, p*1/4)], # rotated 90
                                [QtCore.QPointF(p/2, p/2), QtCore.QPointF(p, p/2), QtCore.QPointF(p*3/4, p*3/4)], # rotated 180
                                [QtCore.QPointF(p/4, p*3/4), QtCore.QPointF(p/2, p/2), QtCore.QPointF(p/2, p)], # rotated 270
                                [QtCore.QPointF(p/4, p/4), QtCore.QPointF(p/2, 0), QtCore.QPointF(p/2, p/2)], # original + flip
                                [QtCore.QPointF(p/2, p/2), QtCore.QPointF(p*3/4, p/4), QtCore.QPointF(p, p/2)], # rotated 90 + flip
                                [QtCore.QPointF(p/2, p/2), QtCore.QPointF(p/2, p), QtCore.QPointF(p*3/4, p*3/4)], # rotated 180 + flip
                                [QtCore.QPointF(0, p/2), QtCore.QPointF(p/4, p*3/4), QtCore.QPointF(p/2, p/2)], # rotated 270 + flip
                                [QtCore.QPointF(0, 0), QtCore.QPointF(p/4, p/4), QtCore.QPointF(0, p/2)], # time reverse
                                [QtCore.QPointF(p/2, 0), QtCore.QPointF(p*3/4, p/4), QtCore.QPointF(p, 0)], # time reverse + rot 90
                                [QtCore.QPointF(p, p/2), QtCore.QPointF(p*3/4, p*3/4), QtCore.QPointF(p, p)], # time reverse + rot 180
                                [QtCore.QPointF(0, p), QtCore.QPointF(p/4, p*3/4), QtCore.QPointF(p/2, p)], # time reverse + rot 270
                                [QtCore.QPointF(0, 0), QtCore.QPointF(p/4, p/4), QtCore.QPointF(p/2, 0)], # time reverse + flip
                                [QtCore.QPointF(p, 0), QtCore.QPointF(p*3/4, p/4), QtCore.QPointF(p, p/2)], # time reverse + rot 90 + flip
                                [QtCore.QPointF(p/2, p), QtCore.QPointF(p*3/4, p*3/4), QtCore.QPointF(p, p)], # time reverse + rot 180 + flip
                                [QtCore.QPointF(0, p), QtCore.QPointF(0, p/2), QtCore.QPointF(p/4, p*3/4)]] # time reverse + rot 270 + flip

        for i, cur_points in enumerate(all_triangle_points):
            self.draw_triangle(painter, cur_points, brush_color=QtGui.QColor(d4_lut[int(data[i]*100)][0], d4_lut[int(data[i]*100)][1], d4_lut[int(data[i]*100)][2]))

        # draw the red triangle selector
        self.draw_triangle(painter, all_triangle_points[self.triangle_index], pen_color=QtGui.QColor('red'), boundary_width=3)

        painter.end()

        # set pixmap to label
        d4_label.setPixmap(d4_pixmap)

        return d4_label


    # draws dihedral 4 visualization to be the NERO plot for PIV experiment
    def draw_d4_nero(self, mode):

        # NERO plot
        self.d4_label_1 = self.draw_individual_d4(self.cur_plot_quantity_1)
        self.d4_label_2 = self.draw_individual_d4(self.cur_plot_quantity_2)

        # single mode layout can be used for both single and aggresingle
        if mode == 'single':
            if self.data_mode == 'aggregate':
                self.aggregate_result_layout.addWidget(self.d4_label_1, 1, 4)
                self.aggregate_result_layout.addWidget(self.d4_label_2, 1, 5)
            elif self.data_mode == 'single':
                self.single_result_layout.addWidget(self.d4_label_1, 1, 1)
                self.single_result_layout.addWidget(self.d4_label_2, 1, 2)

        # aggregate
        elif mode == 'aggregate':
            self.aggregate_result_layout.addWidget(self.d4_label_1, 1, 1)
            self.aggregate_result_layout.addWidget(self.d4_label_2, 1, 2)


    # draw error plots of PIV
    def draw_piv_details(self):
        # the detailed view are two heatmaps showing the error
        # heatmap view
        self.piv_detail_view_1 = pg.GraphicsLayoutWidget()
        # left top right bottom
        self.piv_detail_view_1.ci.layout.setContentsMargins(0, 20, 0, 0)
        self.piv_detail_view_1.setFixedSize(self.plot_size*1.3, self.plot_size*1.3)
        self.piv_detail_view_2 = pg.GraphicsLayoutWidget()
        # left top right bottom
        self.piv_detail_view_2.ci.layout.setContentsMargins(0, 20, 0, 0)
        self.piv_detail_view_2.setFixedSize(self.plot_size*1.3, self.plot_size*1.3)

        # prepare data for the visualization, using current triangle index
        detail_data_1 = self.loss_module(self.all_ground_truths[self.triangle_index], self.all_quantities_1[self.triangle_index], reduction='none').numpy().mean(axis=2)
        detail_data_2 = self.loss_module(self.all_ground_truths[self.triangle_index], self.all_quantities_2[self.triangle_index], reduction='none').numpy().mean(axis=2)

        # normalize the detailed data and take flip
        detail_data_1 = 1 - nero_utilities.lerp(detail_data_1, self.error_min, self.error_max, 0, 1)
        detail_data_2 = 1 - nero_utilities.lerp(detail_data_2, self.error_min, self.error_max, 0, 1)

        # draw the heatmap that represent detailed view
        self.piv_detail_plot_1 = self.draw_individual_heatmap('single', detail_data_1)
        self.piv_detail_plot_2 = self.draw_individual_heatmap('single', detail_data_2)
        # self.view_box_1.scene().sigMouseClicked.connect(heatmap_mouse_clicked(self.view_box_1))
        # self.view_box_2.scene().sigMouseClicked.connect(heatmap_mouse_clicked(self.view_box_2))

        # add to view
        self.piv_detail_view_1.addItem(self.piv_detail_plot_1)
        self.piv_detail_view_2.addItem(self.piv_detail_plot_2)

        if self.data_mode == 'single':
            self.single_result_layout.addWidget(self.piv_detail_view_1, 2, 1)
            self.single_result_layout.addWidget(self.piv_detail_view_2, 2, 2)
        elif self.data_mode == 'aggregate':
            self.aggregate_result_layout.addWidget(self.piv_detail_view_1, 2, 4)
            self.aggregate_result_layout.addWidget(self.piv_detail_view_2, 2, 5)


    # display MNIST aggregated results
    def display_mnist_aggregate_result(self):

        @QtCore.Slot()
        def polar_quantity_changed(text):
            print('Plotting:', text, 'on polar NERO')
            self.quantity_name = text
            self.draw_aggregate_polar()

        # move the model menu on top of the each aggregate NERO plot
        self.aggregate_result_layout.addWidget(self.model_1_menu, 0, 1, 1, 1, QtCore.Qt.AlignCenter)
        self.aggregate_result_layout.addWidget(self.model_2_menu, 0, 2, 1, 1, QtCore.Qt.AlignCenter)

        # move run button in the first column (after aggregate heatmap control)
        self.aggregate_plot_control_layout.addWidget(self.run_button, 4, 0)
        self.aggregate_plot_control_layout.addWidget(self.use_cache_checkbox, 5, 0)

        self.aggregate_result_existed = True

        # drop down menu on selection which quantity to plot
        quantity_menu = QtWidgets.QComboBox()
        quantity_menu.setFixedSize(QtCore.QSize(250, 50))
        quantity_menu.setStyleSheet('font-size: 18px')
        quantity_menu.setEditable(True)
        quantity_menu.lineEdit().setReadOnly(True)
        quantity_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)

        quantity_menu.addItem('Accuracy')
        # quantity_menu.addItem('Confidence')
        quantity_menu.setCurrentText('Accuracy')
        self.quantity_name = 'Accuracy'

        # connect the drop down menu with actions
        quantity_menu.currentTextChanged.connect(polar_quantity_changed)
        self.aggregate_plot_control_layout.addWidget(quantity_menu, 1, 0)

        # draw the aggregate polar plot
        self.draw_aggregate_polar()


    # display MNIST single results
    def display_mnist_single_result(self, type, boundary_width):
        self.single_result_existed = True
        # aggregate mode does not draw arrow
        # if self.data_mode == 'single':
        #     # draw arrow
        #     # add a new label for result if no result has existed
        #     if not self.single_result_existed:
        #         self.arrow_label = QLabel(self)
        #         self.arrow_label.setContentsMargins(0, 0, 0, 0)
        #         self.arrow_label.setAlignment(QtCore.Qt.AlignCenter)
        #         self.arrow_label.setWordWrap(True)
        #         self.arrow_label.setTextFormat(QtGui.Qt.AutoText)
        #         self.single_result_existed = True

        #     arrow_pixmap = QPixmap(100, 50)
        #     arrow_pixmap.fill(QtCore.Qt.white)
        #     painter = QtGui.QPainter(arrow_pixmap)
        #     # set pen (used to draw outlines of shapes) and brush (draw the background of a shape)
        #     pen = QtGui.QPen()
        #     # draw arrow to indicate feeding
        #     self.draw_arrow(painter, pen, 100, 50, boundary_width)

        #     # add to the label and layout
        #     self.arrow_label.setPixmap(arrow_pixmap)
        #     self.single_result_layout.addWidget(self.arrow_label, 1, 1)
        #     painter.end()

        # draw result using bar plot
        if type == 'bar':
            # create individual bar (item) for individual hover/click control
            class InteractiveBarItem(pg.BarGraphItem):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    if self.opts['brush'] == 'blue':
                        cur_class = int(self.opts.get('x0')[0] + 0.2)
                        cur_value = self.opts.get('height')
                    elif self.opts['brush'] == 'magenta':
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
            self.bar_plot.setFixedSize(self.plot_size, self.plot_size)
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
            #                                      brush='magenta')

            #     self.bar_plot.addItem(cur_graph_1)
            #     self.bar_plot.addItem(cur_graph_2)

            graph_1 = pg.BarGraphItem(x=np.arange(len(self.output_1))-0.2, height = list(self.output_1), width = 0.4, brush ='blue')
            graph_2 = pg.BarGraphItem(x=np.arange(len(self.output_1))+0.2, height = list(self.output_2), width = 0.4, brush ='magenta')
            self.bar_plot.addItem(graph_1)
            self.bar_plot.addItem(graph_2)
            # disable moving around
            self.bar_plot.setMouseEnabled(x=False, y=False)
            if self.data_mode == 'single':
                self.single_result_layout.addWidget(self.bar_plot, 1, 2)
            elif self.data_mode == 'aggregate':
                self.aggregate_result_layout.addWidget(self.bar_plot, 2, 6)

        elif type == 'polar':

            # helper function for clicking inside demension reduced scatter plot
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
                if self.single_result_existed:
                    self.run_model_once()

                # only allow clicking one point at a time
                # save the old brush
                if points[0].brush() == pg.mkBrush(QtGui.QColor('blue')):
                    self.old_brush = pg.mkBrush(QtGui.QColor('blue'))

                elif points[0].brush() == pg.mkBrush(QtGui.QColor('magenta')):
                    self.old_brush = pg.mkBrush(QtGui.QColor('magenta'))

                # create new brush
                new_brush = pg.mkBrush(255, 0, 0, 255)
                points[0].setBrush(new_brush)
                points[0].setPen(5)

                self.last_clicked = points[0]

            # initialize view and plot
            polar_view = pg.GraphicsLayoutWidget()
            polar_view.setBackground('white')
            polar_view.setFixedSize(self.plot_size, self.plot_size)
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
                                    'brush': QtGui.QColor('blue')})

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
                                    'brush': QtGui.QColor('magenta')})

            # draw lines to better show shape
            line_1 = self.polar_plot.plot(all_x_1, all_y_1, pen = QtGui.QPen(QtGui.Qt.blue, 0.03))
            line_2 = self.polar_plot.plot(all_x_2, all_y_2, pen = QtGui.QPen(QtGui.QColor('magenta'), 0.03))

            # add points to the item
            self.scatter_items.addPoints(all_points_1)
            self.scatter_items.addPoints(all_points_2)

            # add points to the plot
            self.polar_plot.addItem(self.scatter_items)
            # connect click events on scatter items (disabled)
            # self.scatter_items.sigClicked.connect(clicked)

            # used for clicking on the polar plot
            def polar_mouse_clicked(event):
                self.polar_clicked = not self.polar_clicked

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
                    if self.single_result_existed:
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
                    self.cur_line = self.polar_plot.plot(line_x, line_y, pen = QtGui.QPen(QtGui.Qt.green, 0.02))

                    # display current results on the line
                    self.draw_circle_on_polar()

            self.polar_clicked = False
            self.polar_plot.scene().sigMouseClicked.connect(polar_mouse_clicked)
            self.polar_plot.scene().sigMouseMoved.connect(polar_mouse_moved)

            # fix zoom level
            # self.polar_plot.vb.scaleBy((0.5, 0.5))
            self.polar_plot.setMouseEnabled(x=False, y=False)

            # add the plot view to the layout
            if self.data_mode == 'single':
                self.single_result_layout.addWidget(polar_view, 1, 3)
            elif self.data_mode == 'aggregate':
                self.aggregate_result_layout.addWidget(polar_view, 1, 6)

        else:
            raise Exception('Unsupported display mode')


    # display COCO aggregate results
    def display_coco_aggregate_result(self):
        # move the model menu on top of the each aggregate NERO plot
        self.aggregate_result_layout.addWidget(self.model_1_menu, 0, 1, 1, 1, QtCore.Qt.AlignCenter)
        self.aggregate_result_layout.addWidget(self.model_2_menu, 0, 2, 1, 1, QtCore.Qt.AlignCenter)

        # move run button in the first column (after aggregate heatmap control)
        self.aggregate_plot_control_layout.addWidget(self.run_button, 4, 0)
        self.aggregate_plot_control_layout.addWidget(self.use_cache_checkbox, 5, 0)

        self.aggregate_result_existed = True

        @QtCore.Slot()
        def heatmap_quantity_changed(text):
            print('Plotting:', text, 'on heatmap')
            self.quantity_name = text
            if text == 'Confidence*IOU':
                self.cur_plot_quantity_1 = self.aggregate_avg_conf_1 * self.aggregate_avg_iou_1
                self.cur_plot_quantity_2 = self.aggregate_avg_conf_2 * self.aggregate_avg_iou_2
            elif text == 'Confidence':
                self.cur_plot_quantity_1 = self.aggregate_avg_conf_1
                self.cur_plot_quantity_2 = self.aggregate_avg_conf_2
            elif text == 'IOU':
                self.cur_plot_quantity_1 = self.aggregate_avg_iou_1
                self.cur_plot_quantity_2 = self.aggregate_avg_iou_2
            elif text == 'Precision':
                self.cur_plot_quantity_1 = self.aggregate_avg_precision_1
                self.cur_plot_quantity_2 = self.aggregate_avg_precision_2
            elif text == 'Recall':
                self.cur_plot_quantity_1 = self.aggregate_avg_recall_1
                self.cur_plot_quantity_2 = self.aggregate_avg_recall_2
            elif text == 'F1 score':
                self.cur_plot_quantity_1 = self.aggregate_avg_F_measure_1
                self.cur_plot_quantity_2 = self.aggregate_avg_F_measure_2
            elif text == 'AP':
                self.cur_plot_quantity_1 = self.aggregate_mAP_1
                self.cur_plot_quantity_2 = self.aggregate_mAP_2

            # re-display the heatmap
            self.draw_heatmaps(mode='aggregate')

            # re-run dimension reduction and show result
            if self.dr_result_existed:
                self.run_dimension_reduction()

        # drop down menu on selection which quantity to plot
        quantity_menu = QtWidgets.QComboBox()
        quantity_menu.setFixedSize(QtCore.QSize(250, 50))
        quantity_menu.setStyleSheet('font-size: 18px')
        quantity_menu.setEditable(True)
        quantity_menu.lineEdit().setReadOnly(True)
        quantity_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)

        quantity_menu.addItem('Confidence*IOU')
        quantity_menu.addItem('Confidence')
        quantity_menu.addItem('IOU')
        quantity_menu.addItem('Precision')
        quantity_menu.addItem('Recall')
        quantity_menu.addItem('AP')
        quantity_menu.addItem('F1 Score')
        # self.quantity_menu.setCurrentIndex(0)
        quantity_menu.setCurrentText('Confidence*IOU')

        # connect the drop down menu with actions
        quantity_menu.currentTextChanged.connect(heatmap_quantity_changed)
        self.aggregate_plot_control_layout.addWidget(quantity_menu, 1, 0)

        # define default plotting quantity (IOU*Confidence)
        self.quantity_name = 'Confidence*IOU'
        # averaged (depends on selected class) confidence and iou of the top results (ranked by IOU)
        self.aggregate_avg_conf_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_iou_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_precision_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_recall_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_F_measure_1 = np.zeros((len(self.y_translation), len(self.x_translation)))

        self.aggregate_avg_conf_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_iou_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_precision_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_recall_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_F_measure_2 = np.zeros((len(self.y_translation), len(self.x_translation)))

        for y in range(len(self.y_translation)):
            for x in range(len(self.x_translation)):
                all_samples_conf_sum_1 = []
                all_samples_iou_sum_1 = []
                all_samples_conf_sum_2 = []
                all_samples_iou_sum_2 = []
                all_samples_precision_sum_1 = []
                all_samples_precision_sum_2 = []
                all_samples_recall_sum_1 = []
                all_samples_recall_sum_2 = []
                all_samples_F_measure_sum_1 = []
                all_samples_F_measure_sum_2 = []
                for i in range(len(self.aggregate_outputs_1[y, x])):
                    # either all the classes or one specific class
                    if self.class_selection == 'all' or self.class_selection == self.loaded_images_labels[i]:
                        all_samples_conf_sum_1.append(self.aggregate_outputs_1[y, x][i][0, 4])
                        all_samples_iou_sum_1.append(self.aggregate_outputs_1[y, x][i][0, 6])
                        all_samples_conf_sum_2.append(self.aggregate_outputs_2[y, x][i][0, 4])
                        all_samples_iou_sum_2.append(self.aggregate_outputs_2[y, x][i][0, 6])
                        all_samples_precision_sum_1.append(self.aggregate_precision_1[y, x][i])
                        all_samples_precision_sum_2.append(self.aggregate_precision_2[y, x][i])
                        all_samples_recall_sum_1.append(self.aggregate_recall_1[y, x][i])
                        all_samples_recall_sum_2.append(self.aggregate_recall_2[y, x][i])
                        all_samples_F_measure_sum_1.append(self.aggregate_F_measure_1[y, x][i])
                        all_samples_F_measure_sum_2.append(self.aggregate_F_measure_2[y, x][i])

                # take the average result
                self.aggregate_avg_conf_1[y, x] = np.mean(all_samples_conf_sum_1)
                self.aggregate_avg_iou_1[y, x] = np.mean(all_samples_iou_sum_1)
                self.aggregate_avg_precision_1[y, x] = np.mean(all_samples_precision_sum_1)
                self.aggregate_avg_recall_1[y, x] = np.mean(all_samples_recall_sum_1)
                self.aggregate_avg_F_measure_1[y, x] = np.mean(all_samples_F_measure_sum_1)

                self.aggregate_avg_conf_2[y, x] = np.mean(all_samples_conf_sum_2)
                self.aggregate_avg_iou_2[y, x] = np.mean(all_samples_iou_sum_2)
                self.aggregate_avg_precision_2[y, x] = np.mean(all_samples_precision_sum_2)
                self.aggregate_avg_recall_2[y, x] = np.mean(all_samples_recall_sum_2)
                self.aggregate_avg_F_measure_2[y, x] = np.mean(all_samples_F_measure_sum_2)

        # default plotting quantity is Confidence*IOU
        self.cur_plot_quantity_1 = self.aggregate_avg_conf_1 * self.aggregate_avg_iou_1
        self.cur_plot_quantity_2 = self.aggregate_avg_conf_2 * self.aggregate_avg_iou_2

        # draw the heatmap
        self.draw_heatmaps(mode='aggregate')


    # display COCO single results
    def display_coco_single_result(self):

        # if single mode, change control menus' locations
        if self.data_mode == 'single':
            # move the model menu on top of the each individual NERO plot when in single mode
            self.single_result_layout.addWidget(self.model_1_menu, 0, 1, 1, 1, QtCore.Qt.AlignCenter)
            self.single_result_layout.addWidget(self.model_2_menu, 0, 2, 1, 1, QtCore.Qt.AlignCenter)

            # move run button below the displayed image
            self.single_result_layout.addWidget(self.run_button, 2, 0)
            self.single_result_layout.addWidget(self.use_cache_checkbox, 3, 0)

        # plot current field-of-view's detailed prediction results
        self.draw_model_output()

        self.single_result_existed = True

        # draw result using heatmaps
        @QtCore.Slot()
        def realtime_inference_checkbox_clicked(state):
            if state == QtCore.Qt.Checked:
                self.realtime_inference = True
            else:
                self.realtime_inference = False

        @QtCore.Slot()
        def heatmap_quantity_changed(text):
            print('Plotting:', text, 'on heatmap')
            self.quantity_name = text

            if text == 'Confidence*IOU':
                if self.data_mode == 'single':
                    self.cur_plot_quantity_1 = self.all_quantities_1[:, :, 4] * self.all_quantities_1[:, :, 6]
                    self.cur_plot_quantity_2 = self.all_quantities_2[:, :, 4] * self.all_quantities_2[:, :, 6]
                elif self.data_mode == 'aggregate':
                    # current selected individual images' result on all transformations
                    for y in range(len(self.y_translation)):
                        for x in range(len(self.x_translation)):
                            self.cur_plot_quantity_1[y, x] = self.aggregate_outputs_1[y, x][self.image_index][0, 4] * self.aggregate_outputs_1[y, x][self.image_index][0, 6]
                            self.cur_plot_quantity_2[y, x] = self.aggregate_outputs_2[y, x][self.image_index][0, 4] * self.aggregate_outputs_2[y, x][self.image_index][0, 6]

            elif text == 'Confidence':
                if self.data_mode == 'single':
                    self.cur_plot_quantity_1 = self.all_quantities_1[:, :, 4]
                    self.cur_plot_quantity_2 = self.all_quantities_2[:, :, 4]
                elif self.data_mode == 'aggregate':
                    # current selected individual images' result on all transformations
                    for y in range(len(self.y_translation)):
                        for x in range(len(self.x_translation)):
                            self.cur_plot_quantity_1[y, x] = self.aggregate_outputs_1[y, x][self.image_index][0, 4]
                            self.cur_plot_quantity_2[y, x] = self.aggregate_outputs_2[y, x][self.image_index][0, 4]

            elif text == 'IOU':
                if self.data_mode == 'single':
                    self.cur_plot_quantity_1 = self.all_quantities_1[:, :, 6]
                    self.cur_plot_quantity_2 = self.all_quantities_2[:, :, 6]
                elif self.data_mode == 'aggregate':
                    # current selected individual images' result on all transformations
                    for y in range(len(self.y_translation)):
                        for x in range(len(self.x_translation)):
                            self.cur_plot_quantity_1[y, x] = self.aggregate_outputs_1[y, x][self.image_index][0, 6]
                            self.cur_plot_quantity_2[y, x] = self.aggregate_outputs_2[y, x][self.image_index][0, 6]

            elif text == 'Consensus':
                self.cur_plot_quantity_1 = np.zeros((self.all_quantities_1.shape[0], self.all_quantities_1.shape[1]))
                self.cur_plot_quantity_2 = np.zeros((self.all_quantities_2.shape[0], self.all_quantities_2.shape[1]))
                # for each position, compute its bounding box center
                # x_ratio = int(self.image_size // self.all_quantities_1.shape[0])
                # y_ratio = int(self.image_size // self.all_quantities_1.shape[1])
                for i in range(self.all_quantities_1.shape[0]):
                    for j in range(self.all_quantities_1.shape[1]):
                        # correct translation amount
                        x_tran = self.all_translations[i, j, 0]
                        y_tran = self.all_translations[i, j, 1]

                        # current bounding box center from model 1 and 2
                        cur_center_x_1 = (self.all_quantities_1[i, j, 0] + self.all_quantities_1[i, j, 2]) / 2
                        cur_center_y_1 = (self.all_quantities_1[i, j, 1] + self.all_quantities_1[i, j, 3]) / 2
                        cur_center_x_2 = (self.all_quantities_2[i, j, 0] + self.all_quantities_2[i, j, 2]) / 2
                        cur_center_y_2 = (self.all_quantities_2[i, j, 1] + self.all_quantities_2[i, j, 3]) / 2

                        # model output translation
                        x_tran_model_1 = cur_center_x_1 - self.image_size//2 - 1
                        y_tran_model_1 = cur_center_y_1 - self.image_size//2 - 1
                        x_tran_model_2 = cur_center_x_2 - self.image_size//2 - 1
                        y_tran_model_2 = cur_center_y_2 - self.image_size//2 - 1

                        # compute percentage
                        if np.sqrt(x_tran**2 + y_tran**2) == 0:
                            self.cur_plot_quantity_1[i, j] = 1
                            self.cur_plot_quantity_2[i, j] = 1
                        else:
                            self.cur_plot_quantity_1[i, j] = 1 - np.sqrt((x_tran_model_1-x_tran)**2 + (y_tran_model_1-y_tran)**2) / np.sqrt(x_tran**2 + y_tran**2)
                            self.cur_plot_quantity_2[i, j] = 1 - np.sqrt((x_tran_model_2-x_tran)**2 + (y_tran_model_2-y_tran)**2) / np.sqrt(x_tran**2 + y_tran**2)

            # re-display the heatmap
            self.draw_heatmaps(mode='single')

        # drop down menu on selection which quantity to plot
        # layout that controls the plotting items
        self.single_plot_control_layout = QtWidgets.QVBoxLayout()
        quantity_menu = QtWidgets.QComboBox()
        quantity_menu.setFixedSize(QtCore.QSize(250, 50))
        quantity_menu.setStyleSheet('font-size: 18px')
        quantity_menu.setEditable(True)
        quantity_menu.lineEdit().setReadOnly(True)
        quantity_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)

        quantity_menu.addItem('Confidence*IOU')
        quantity_menu.addItem('Confidence')
        quantity_menu.addItem('IOU')
        quantity_menu.addItem('Consensus')
        # self.quantity_menu.setCurrentIndex(0)
        quantity_menu.setCurrentText('Confidence*IOU')

        # connect the drop down menu with actions
        quantity_menu.currentTextChanged.connect(heatmap_quantity_changed)
        self.single_plot_control_layout.addWidget(quantity_menu)

        # checkbox on if doing real-time inference
        self.realtime_inference_checkbox = QtWidgets.QCheckBox('Realtime inference when dragging')
        self.realtime_inference_checkbox.setStyleSheet('font-size: 18px')
        self.realtime_inference_checkbox.setFixedSize(QtCore.QSize(300, 50))
        self.realtime_inference_checkbox.stateChanged.connect(realtime_inference_checkbox_clicked)
        if self.realtime_inference:
            self.realtime_inference_checkbox.setChecked(True)
        else:
            self.realtime_inference_checkbox.setChecked(False)

        self.single_plot_control_layout.addWidget(self.realtime_inference_checkbox)


        # define default plotting quantity
        if self.data_mode == 'single':
            # add plot control layout to general layout
            self.single_result_layout.addLayout(self.single_plot_control_layout, 0, 0)
            self.cur_plot_quantity_1 = self.all_quantities_1[:, :, 4] * self.all_quantities_1[:, :, 6]
            self.cur_plot_quantity_2 = self.all_quantities_2[:, :, 4] * self.all_quantities_2[:, :, 6]
        elif self.data_mode == 'aggregate':
            # add plot control layout to general layout
            self.aggregate_result_layout.addLayout(self.single_plot_control_layout, 0, 3)
            # current selected individual images' result on all transformations
            self.cur_plot_quantity_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
            self.cur_plot_quantity_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
            for y in range(len(self.y_translation)):
                for x in range(len(self.x_translation)):
                    self.cur_plot_quantity_1[y, x] = self.aggregate_outputs_1[y, x][self.image_index][0, 4] * self.aggregate_outputs_1[y, x][self.image_index][0, 6]
                    self.cur_plot_quantity_2[y, x] = self.aggregate_outputs_2[y, x][self.image_index][0, 4] * self.aggregate_outputs_2[y, x][self.image_index][0, 6]

        # draw the heatmap
        self.draw_heatmaps(mode='single')


    # display COCO aggregate result
    def display_piv_aggregate_result(self):
        # move the model menu on top of the each aggregate NERO plot
        self.aggregate_result_layout.addWidget(self.model_1_menu, 0, 1, 1, 1, QtCore.Qt.AlignCenter)
        self.aggregate_result_layout.addWidget(self.model_2_menu, 0, 2, 1, 1, QtCore.Qt.AlignCenter)

        # move run button in the first column (after aggregate heatmap control)
        self.aggregate_plot_control_layout.addWidget(self.run_button, 4, 0)
        self.aggregate_plot_control_layout.addWidget(self.use_cache_checkbox, 5, 0)

        self.aggregate_result_existed = True

        # helper function on compute, normalize the loss and display quantity
        def compute_nero_display_quantity():

            if self.cur_plot_quantity == 'RMSE':
                self.loss_module = nero_utilities.RMSELoss()
            elif self.cur_plot_quantity == 'MSE':
                self.loss_module = torch.nn.MSELoss()
            elif self.cur_plot_quantity == 'MAE':
                self.loss_module = torch.nn.L1Loss()
            elif self.cur_plot_quantity == 'AEE':
                self.loss_module = nero_utilities.AEELoss()

            # compute loss using torch loss module
            for i in range(self.num_transformations):
                loss_1 = self.loss_module(self.all_ground_truths[i], self.all_quantities_1[i], reduction='none').numpy()
                loss_2 = self.loss_module(self.all_ground_truths[i], self.all_quantities_2[i], reduction='none').numpy()

                # average the x and y
                self.all_losses_1[i] = loss_1.mean(axis=2)
                self.all_losses_2[i] = loss_2.mean(axis=2)

            # get the max and min for normalization purpose
            self.error_min = np.min(self.all_losses_1)
            self.error_max = np.max(self.all_losses_2)

            # normalize the losses
            self.all_losses_1 = nero_utilities.lerp(self.all_losses_1.mean(axis=(1, 2)), self.error_min, self.error_max, 0, 1)
            self.all_losses_2 = nero_utilities.lerp(self.all_losses_2.mean(axis=(1, 2)), self.error_min, self.error_max, 0, 1)

            # average element-wise loss to scalar and normalize between 0 and 1
            self.cur_plot_quantity_1 = 1 - self.all_losses_1
            self.cur_plot_quantity_2 = 1 - self.all_losses_2

        @QtCore.Slot()
        def piv_plot_quantity_changed(text):
            print('Plotting:', text, 'on heatmap')
            self.quantity_name = text

            if text == 'RMSE':
                self.cur_plot_quantity_1 = text
            elif text == 'MSE':
                self.cur_plot_quantity_1 = text
            elif text == 'MAE':
                self.cur_plot_quantity_1 = text
            elif text == 'AEE':
                self.cur_plot_quantity_1 = text

            # compute the quantity to plot
            compute_nero_display_quantity()

            # re-display the heatmap
            self.draw_d4_nero(mode='aggregate')

            # re-run dimension reduction and show result
            if self.dr_result_existed:
                self.run_dimension_reduction()

        # drop down menu on selection which quantity to plot
        quantity_menu = QtWidgets.QComboBox()
        quantity_menu.setFixedSize(QtCore.QSize(250, 50))
        quantity_menu.setStyleSheet('font-size: 18px')
        quantity_menu.setEditable(True)
        quantity_menu.lineEdit().setReadOnly(True)
        quantity_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)

        quantity_menu.addItem('Confidence*IOU')
        quantity_menu.addItem('Confidence')
        quantity_menu.addItem('IOU')
        quantity_menu.addItem('Precision')
        quantity_menu.addItem('Recall')
        quantity_menu.addItem('AP')
        quantity_menu.addItem('F1 Score')
        # self.quantity_menu.setCurrentIndex(0)
        quantity_menu.setCurrentText('Confidence*IOU')

        # connect the drop down menu with actions
        quantity_menu.currentTextChanged.connect(piv_plot_quantity_changed)
        self.aggregate_plot_control_layout.addWidget(quantity_menu, 1, 0)

        # define default plotting quantity (IOU*Confidence)
        self.quantity_name = 'Confidence*IOU'
        # averaged (depends on selected class) confidence and iou of the top results (ranked by IOU)
        self.aggregate_avg_conf_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_iou_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_precision_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_recall_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_F_measure_1 = np.zeros((len(self.y_translation), len(self.x_translation)))

        self.aggregate_avg_conf_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_iou_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_precision_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_recall_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
        self.aggregate_avg_F_measure_2 = np.zeros((len(self.y_translation), len(self.x_translation)))

        for y in range(len(self.y_translation)):
            for x in range(len(self.x_translation)):
                all_samples_conf_sum_1 = []
                all_samples_iou_sum_1 = []
                all_samples_conf_sum_2 = []
                all_samples_iou_sum_2 = []
                all_samples_precision_sum_1 = []
                all_samples_precision_sum_2 = []
                all_samples_recall_sum_1 = []
                all_samples_recall_sum_2 = []
                all_samples_F_measure_sum_1 = []
                all_samples_F_measure_sum_2 = []
                for i in range(len(self.aggregate_outputs_1[y, x])):
                    # either all the classes or one specific class
                    if self.class_selection == 'all' or self.class_selection == self.loaded_images_labels[i]:
                        all_samples_conf_sum_1.append(self.aggregate_outputs_1[y, x][i][0, 4])
                        all_samples_iou_sum_1.append(self.aggregate_outputs_1[y, x][i][0, 6])
                        all_samples_conf_sum_2.append(self.aggregate_outputs_2[y, x][i][0, 4])
                        all_samples_iou_sum_2.append(self.aggregate_outputs_2[y, x][i][0, 6])
                        all_samples_precision_sum_1.append(self.aggregate_precision_1[y, x][i])
                        all_samples_precision_sum_2.append(self.aggregate_precision_2[y, x][i])
                        all_samples_recall_sum_1.append(self.aggregate_recall_1[y, x][i])
                        all_samples_recall_sum_2.append(self.aggregate_recall_2[y, x][i])
                        all_samples_F_measure_sum_1.append(self.aggregate_F_measure_1[y, x][i])
                        all_samples_F_measure_sum_2.append(self.aggregate_F_measure_2[y, x][i])

                # take the average result
                self.aggregate_avg_conf_1[y, x] = np.mean(all_samples_conf_sum_1)
                self.aggregate_avg_iou_1[y, x] = np.mean(all_samples_iou_sum_1)
                self.aggregate_avg_precision_1[y, x] = np.mean(all_samples_precision_sum_1)
                self.aggregate_avg_recall_1[y, x] = np.mean(all_samples_recall_sum_1)
                self.aggregate_avg_F_measure_1[y, x] = np.mean(all_samples_F_measure_sum_1)

                self.aggregate_avg_conf_2[y, x] = np.mean(all_samples_conf_sum_2)
                self.aggregate_avg_iou_2[y, x] = np.mean(all_samples_iou_sum_2)
                self.aggregate_avg_precision_2[y, x] = np.mean(all_samples_precision_sum_2)
                self.aggregate_avg_recall_2[y, x] = np.mean(all_samples_recall_sum_2)
                self.aggregate_avg_F_measure_2[y, x] = np.mean(all_samples_F_measure_sum_2)

        # default plotting quantity is Confidence*IOU
        self.cur_plot_quantity_1 = self.aggregate_avg_conf_1 * self.aggregate_avg_iou_1
        self.cur_plot_quantity_2 = self.aggregate_avg_conf_2 * self.aggregate_avg_iou_2

        # draw the heatmap
        self.draw_heatmaps(mode='aggregate')


    # display PIV single results
    def display_piv_single_result(self):
        # if single mode, change control menus' locations
        if self.data_mode == 'single':
            # move the model menu on top of the each individual NERO plot when in single mode
            self.single_result_layout.addWidget(self.model_1_menu, 0, 1, 1, 1, QtCore.Qt.AlignCenter)
            self.single_result_layout.addWidget(self.model_2_menu, 0, 2, 1, 1, QtCore.Qt.AlignCenter)

            # move run button below the displayed image
            self.single_result_layout.addWidget(self.run_button, 3, 0)
            self.single_result_layout.addWidget(self.use_cache_checkbox, 4, 0)

        # helper function on compute, normalize the loss and display quantity
        def compute_nero_plot_quantity():
            # compute loss using torch loss module
            if self.cur_plot_quantity == 'RMSE':
                self.loss_module = nero_utilities.RMSELoss()
            elif self.cur_plot_quantity == 'MSE':
                self.loss_module = torch.nn.MSELoss()
            elif self.cur_plot_quantity == 'MAE':
                self.loss_module = torch.nn.L1Loss()
            elif self.cur_plot_quantity == 'AEE':
                self.loss_module = nero_utilities.AEELoss()

            for i in range(self.num_transformations):
                self.cur_plot_quantity_1[i] = self.loss_module(self.all_ground_truths[i], self.all_quantities_1[i]).numpy()
                self.cur_plot_quantity_1[i] = self.loss_module(self.all_ground_truths[i], self.all_quantities_2[i]).numpy()

            # normalize these errors between 0 and 1
            self.cur_plot_quantity_1 = nero_utilities.lerp(self.cur_plot_quantity_1, self.error_min, self.error_max, 0, 1)
            self.cur_plot_quantity_2 = nero_utilities.lerp(self.cur_plot_quantity_2, self.error_min, self.error_max, 0, 1)
            self.cur_plot_quantity_1 = 1 - self.cur_plot_quantity_1
            self.cur_plot_quantity_2 = 1 - self.cur_plot_quantity_2

        @QtCore.Slot()
        def piv_plot_quantity_changed(text):
            print('Plotting:', text, 'on detailed PIV plots')
            self.cur_plot_quantity = text

            # compute the quantity needed to plot individual NERO plot
            compute_nero_plot_quantity()

            # plot/update the individual NERO plot
            self.draw_d4_nero(mode='single')

            # update detailed plot of PIV
            self.draw_piv_details()

        # drop down menu on selection which quantity to plot
        # layout that controls the plotting items
        self.single_plot_control_layout = QtWidgets.QVBoxLayout()
        quantity_menu = QtWidgets.QComboBox()
        quantity_menu.setFixedSize(QtCore.QSize(250, 50))
        quantity_menu.setStyleSheet('font-size: 18px')
        quantity_menu.setEditable(True)
        quantity_menu.lineEdit().setReadOnly(True)
        quantity_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)

        # all the different plot quantities (losses)
        self.all_plot_quantities = ['RMSE', 'MSE', 'MAE', 'AEE']
        for cur_quantity in self.all_plot_quantities:
            quantity_menu.addItem(cur_quantity)

        quantity_menu.setCurrentText(self.all_plot_quantities[0])
        # by default the loss is RMSE
        self.cur_plot_quantity = 'RMSE'
        self.loss_module = nero_utilities.RMSELoss()

        # connect the drop down menu with actions
        quantity_menu.currentTextChanged.connect(piv_plot_quantity_changed)
        self.single_plot_control_layout.addWidget(quantity_menu)

        # define default plotting quantity
        if self.data_mode == 'single':
            # add plot control layout to general layout
            self.single_result_layout.addLayout(self.single_plot_control_layout, 0, 0)
            # plot quantity in individual nero plot
            self.cur_plot_quantity_1 = np.zeros(self.num_transformations)
            self.cur_plot_quantity_2 = np.zeros(self.num_transformations)

            # compute average error as the largest and smallest for normalization
            # not using individual pixel min/max because some have too large perks
            for i in range(len(self.all_ground_truths)):
                self.cur_plot_quantity_1[i] = self.loss_module(self.all_ground_truths[i], self.all_quantities_1[i], reduction='mean').numpy()
                self.cur_plot_quantity_2[i] = self.loss_module(self.all_ground_truths[i], self.all_quantities_2[i], reduction='mean').numpy()

            # get the max and min for normalization purpose
            self.error_min = min(np.min(self.cur_plot_quantity_1), np.min(self.cur_plot_quantity_2))
            self.error_max = max(np.max(self.cur_plot_quantity_1), np.max(self.cur_plot_quantity_2))

            # normalize the losses
            self.cur_plot_quantity_1 = nero_utilities.lerp(self.cur_plot_quantity_1, self.error_min, self.error_max, 0, 1)
            self.cur_plot_quantity_2 = nero_utilities.lerp(self.cur_plot_quantity_2, self.error_min, self.error_max, 0, 1)

            # average element-wise loss to scalar and normalize between 0 and 1
            self.cur_plot_quantity_1 = 1 - self.cur_plot_quantity_1
            self.cur_plot_quantity_2 = 1 - self.cur_plot_quantity_2

        elif self.data_mode == 'aggregate':
            raise NotImplementedError

        # visualize the individual NERO plot of the current input
        self.draw_d4_nero(mode='single')

        # the detailed plot of PIV
        self.draw_piv_details()


    # mouse move event only applies in the MNIST case
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
                self.cur_line = self.polar_plot.plot(line_x, line_y, pen = QtGui.QPen(QtGui.Qt.green, 0.02))

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


    # print help message
    def print_help(self):
        print('Ah Oh, help not available')


if __name__ == "__main__":

    app = QtWidgets.QApplication([])
    widget = UI_MainWindow()
    # widget.resize(1920, 1080)
    widget.show()

    sys.exit(app.exec())
