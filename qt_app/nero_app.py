from operator import truediv
import os
import sys
import glob
import PySide6
import torch
import argparse
import numpy as np
import flowiz as fz
from PIL import Image
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import QWidget, QLabel, QRadioButton

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import manifold
from sklearn.manifold import TSNE
import umap

import nero_transform
import nero_utilities
import nero_run_model

import warnings

import teem
import qiv

warnings.filterwarnings('ignore')

# globa configurations
pg.setConfigOptions(antialias=True, background='w')
# use pyside gpu acceleration if gpu detected
if torch.cuda.is_available():
    # pg.setConfigOption('useCupy', True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    pg.setConfigOption('useCupy', False)


class UI_MainWindow(QWidget):
    def __init__(self, pre_selected_mode, demo, cache_path):
        super().__init__()
        # window size
        if pre_selected_mode == 'digit_recognition':
            self.resize(1920, 1080)
        elif pre_selected_mode == 'object_detection':
            self.resize(2280, 1080)
        elif pre_selected_mode == 'piv':
            self.resize(2280, 1080)
        # set window title
        self.setWindowTitle('Non-Equivariance Revealed on Orbits')
        # white background color
        self.setStyleSheet('background-color: rgb(255, 255, 255);')
        # general layout
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignCenter)
        # left, top, right, and bottom margins
        self.layout.setContentsMargins(10, 10, 10, 10)

        # individual laytout for different widgets
        # mode selections
        # self.mode = 'digit_recognition'
        self.mode = pre_selected_mode
        # save the previous mode selection for layout swap
        self.previous_mode = pre_selected_mode
        if self.mode != None:
            self.pre_selected = True
        else:
            self.pre_selected = False
        # input data determines data mode
        self.data_mode = None

        # if we are in the demo mode
        self.demo = demo

        # initialize control panel on mode selection
        self.image_existed = False
        self.data_existed = False
        self.run_button_existed = False
        self.aggregate_result_existed = False
        self.single_result_existed = False
        self.dr_result_existed = False

        # load/initialize program cache
        self.use_cache = False

        if cache_path != None:
            self.cache_path = cache_path
            self.cache_dir = self.cache_path.removesuffix(self.cache_path.split('/')[-1])
        else:
            self.cache_dir = os.path.join(os.getcwd(), 'cache')
            if not os.path.isdir(self.cache_dir):
                os.mkdir(self.cache_dir)

            self.cache_path = os.path.join(self.cache_dir, f'{self.mode}', 'nero_cache.npz')
            # if not exist, creat one
            if not os.path.isfile(self.cache_path):
                np.savez(self.cache_path)

        self.cache = dict(np.load(self.cache_path, allow_pickle=True))

        # if we are doing real-time inference when dragging the field of view
        if torch.cuda.is_available():
            self.realtime_inference = True
        else:
            self.realtime_inference = False

        # start initializing control layout
        self.init_mode_control_layout()

        print(f'\nFinished rendering main layout')

    # helper functions on managing the database
    def load_from_cache(self, name):
        # if it exists
        if name in self.cache.keys():
            self.load_successfully = True
            return self.cache[name]
        else:
            print(f'No precomputed result named {name}')
            self.load_successfully = False
            return np.zeros(0)

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

            if self.previous_mode != self.mode or not self.previous_mode or self.pre_selected:
                # clear previous mode's layout
                if not self.pre_selected:
                    self.switch_mode_cleanup()

                # input image size
                self.image_size = 28
                # image size that is used for display
                self.display_image_size = 380
                # heatmap and detailed image plot size
                self.plot_size = 320
                # image (input data) modification mode
                self.rotation = True
                self.translation = False
                # rotation angles
                self.cur_rotation_angle = 0
                # rotation step for evaluation
                self.rotation_step = 5
                # batch size when running in aggregate mode
                self.batch_size = 100

                # preload model 1
                self.model_1_name = 'Original model'
                self.model_1_cache_name = self.model_1_name.split(' ')[0]
                self.model_1_path = glob.glob(
                    os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt')
                )[0]
                self.model_1 = nero_run_model.load_model(self.mode, 'non_eqv', self.model_1_path)

                # preload model 2
                self.model_2_name = 'Data Aug'
                self.model_2_cache_name = self.model_2_name.split(' ')[0]
                self.model_2_path = glob.glob(
                    os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt')
                )[0]
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

                # since later needs some updates, thus initiated here
                self.dr_result_sliders_existed = False

                # start initialing load layout
                self.init_load_layout()

        @QtCore.Slot()
        def object_detection_button_clicked():
            print('Object detection button clicked')
            self.mode = 'object_detection'

            # below layouts depend on mode selection
            if self.previous_mode != self.mode or not self.previous_mode or self.pre_selected:
                # clear previous mode's layout
                if not self.pre_selected:
                    self.switch_mode_cleanup()

                # uncropped image size
                self.uncropped_image_size = 256
                # image (cropped) size that is fed into the model
                self.image_size = 128
                # image size that is used for display
                self.display_image_size = 320
                # heatmap and detailed image plot size
                self.plot_size = 320
                # image (input data) modification mode
                self.translation = False
                # translation step when evaluating
                self.translation_step_aggregate = 4
                self.translation_step_single = 4
                # batch size when running in aggregate mode
                self.batch_size = 64

                # predefined model paths
                self.model_1_name = '0% jittering'
                self.model_1_cache_name = self.model_1_name.split('%')[0]
                self.model_1_path = glob.glob(
                    os.path.join(
                        os.getcwd(),
                        'example_models',
                        self.mode,
                        'custom_trained',
                        f'object_0-jittered',
                        '*.pth',
                    )
                )[0]
                # pre-trained model does not need model path
                self.model_2_name = 'Pre-trained'
                self.model_2_cache_name = self.model_2_name.split('-')[0]
                self.model_2_path = None
                # preload model
                self.model_1 = nero_run_model.load_model(
                    self.mode, 'custom_trained', self.model_1_path
                )
                self.model_2 = nero_run_model.load_model(
                    self.mode, 'pre_trained', self.model_2_path
                )

                # different class names (original COCO classes, custom 5-class and the one that pretrained PyTorch model uses)
                self.original_coco_names_path = os.path.join(
                    os.getcwd(), 'example_data', self.mode, 'coco.names'
                )
                self.custom_coco_names_path = os.path.join(
                    os.getcwd(), 'example_data', self.mode, 'custom.names'
                )
                self.pytorch_coco_names_path = os.path.join(
                    os.getcwd(), 'example_data', self.mode, 'pytorch_coco.names'
                )

                # load these name files
                self.original_coco_names = nero_utilities.load_coco_classes_file(
                    self.original_coco_names_path
                )
                self.custom_coco_names = nero_utilities.load_coco_classes_file(
                    self.custom_coco_names_path
                )
                self.pytorch_coco_names = nero_utilities.load_coco_classes_file(
                    self.pytorch_coco_names_path
                )

                print(f'Custom 5 classes: {self.custom_coco_names}')

                # unique quantity of the result of current data
                self.all_quantities_1 = []
                self.all_quantities_2 = []

                # when doing highlighting
                self.last_clicked = None
                self.cur_line = None

                self.previous_mode = self.mode

                # since later needs some updates, thus initiated here
                self.dr_result_sliders_existed = False

                # start initialing load layout
                self.init_load_layout()

        @QtCore.Slot()
        def piv_button_clicked():
            print('PIV button clicked')
            self.mode = 'piv'

            # below layouts depend on mode selection
            if self.previous_mode != self.mode or not self.previous_mode or self.pre_selected:
                # clear previous mode's layout
                if not self.pre_selected:
                    self.switch_mode_cleanup()

                # image (cropped) size that is fed into the model
                self.image_size = 256
                # image size that is used for display
                self.display_image_size = 256
                # heatmap and detailed image plot size
                self.plot_size = 320
                # batch size when running in aggregate mode
                self.batch_size = 4
                # image (input data) modification mode
                self.rotation = False
                self.flip = False
                self.time_reverse = False

                # predefined model paths
                self.model_1_name = 'PIV-LiteFlowNet-en'
                # LiteFlowNet
                self.model_1_cache_name = self.model_1_name.split('-')[1]
                self.model_1_path = glob.glob(
                    os.path.join(
                        os.getcwd(),
                        'example_models',
                        self.mode,
                        'PIV-LiteFlowNet-en',
                        f'*.paramOnly',
                    )
                )[0]
                # Gunnar-Farneback does not need model path
                self.model_2_name = 'Gunnar-Farneback'
                self.model_2_cache_name = self.model_2_name
                self.model_2_path = None
                # preload model
                self.model_1 = nero_run_model.load_model(
                    self.mode, self.model_1_name, self.model_1_path
                )
                # Gunnar-Farneback is not a model
                # self.model_2 = nero_run_model.load_model(self.mode, self.model_1_name, self.model_1_path)
                self.model_2 = None

                # cayley table that shows the group orbit result
                self.cayley_table = np.zeros((8, 8), dtype=np.int8)
                # the columns represents current state, rows represent new action
                # 0: no transformation
                self.cayley_table[0] = [0, 1, 2, 3, 4, 5, 6, 7]
                # 1: / diagonal flip
                self.cayley_table[1] = [1, 0, 7, 6, 5, 4, 3, 2]
                # 2: counter-clockwise 90 rotation
                self.cayley_table[2] = [2, 3, 4, 5, 6, 7, 0, 1]
                # 3: horizontal flip (by y axis)
                self.cayley_table[3] = [3, 2, 1, 0, 7, 6, 5, 4]
                # 4: counter-clockwise 180 rotation
                self.cayley_table[4] = [4, 5, 6, 7, 0, 1, 2, 3]
                # 5: \ diagnal flip
                self.cayley_table[5] = [5, 4, 3, 2, 1, 0, 7, 6]
                # 6: counter-clockwise 270 rotation
                self.cayley_table[6] = [6, 7, 0, 1, 2, 3, 4, 5]
                # 7: vertical flip (by x axis)
                self.cayley_table[7] = [7, 6, 5, 4, 3, 2, 1, 0]

                # piv nero layout in terms of transformation
                """
                2'  2(Rot90)            1(right diag flip)   1'
                3'  3(hori flip)        0(original)          0'
                4'  4(Rot180)           7(vert flip)         7'
                5'  5(left diag flip)   6(Rot270)            6'
                """
                self.piv_nero_layout = np.array(
                    [[10, 2, 1, 9], [11, 3, 0, 8], [12, 4, 7, 15], [13, 5, 6, 14]]
                )

                self.piv_nero_layout_names = [
                    [
                        'Time-reversed Rot90 (ccw)',
                        'Rot90 (ccw)',
                        'Diagonal (/) flip',
                        'Time-reversed diagonal flip',
                    ],
                    [
                        'Time-reversed horizontal flip',
                        'Horizontal flip',
                        'Original',
                        'Time-reversed original',
                    ],
                    [
                        'Time-reversed Rot180 (ccw)',
                        'Rot180 (ccw)',
                        'Vertical flip',
                        'Time-reversed vertical flip',
                    ],
                    [
                        'Time-reversed anti-diagonal (\) flip',
                        'Anti-diagonal (\) flip',
                        'Rot270 (ccw)',
                        'Time-reversed Rot270 (ccw)',
                    ],
                ]

                # unique quantity of the result of current data
                self.all_quantities_1 = []
                self.all_quantities_2 = []

                # when doing highlighting
                self.last_clicked = None

                self.previous_mode = self.mode

                # since later needs some updates, thus initiated here
                self.dr_result_sliders_existed = False

                # start initialing load layout
                self.init_load_layout()

        # mode selection radio buttons
        self.mode_control_layout = QtWidgets.QGridLayout()
        self.mode_control_layout.setContentsMargins(50, 0, 0, 50)
        # radio buttons on mode selection (digit_recognition, object detection, PIV)
        if not self.pre_selected:
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
            self.radio_button_1.setStyleSheet(
                'QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};'
            )
            self.radio_button_1.pressed.connect(digit_recognition_button_clicked)
            self.mode_control_layout.addWidget(self.radio_button_1, 0, 1)

            self.radio_button_2 = QRadioButton('Object detection')
            self.radio_button_2.setFixedSize(QtCore.QSize(400, 50))
            self.radio_button_2.setStyleSheet(
                'QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};'
            )
            self.radio_button_2.pressed.connect(object_detection_button_clicked)
            self.mode_control_layout.addWidget(self.radio_button_2, 1, 1)

            self.radio_button_3 = QRadioButton('Particle Image Velocimetry (PIV)')
            self.radio_button_3.setFixedSize(QtCore.QSize(400, 50))
            self.radio_button_3.setStyleSheet(
                'QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};'
            )
            self.radio_button_3.pressed.connect(piv_button_clicked)
            self.mode_control_layout.addWidget(self.radio_button_3, 2, 1)

            self.layout.addLayout(self.mode_control_layout, 0, 0)

        else:
            # used for default state, if applicable
            if self.mode == 'digit_recognition':
                digit_recognition_button_clicked()
            elif self.mode == 'object_detection':
                object_detection_button_clicked()
            elif self.mode == 'piv':
                piv_button_clicked()

    # load single mnist image from self.image_path
    def load_single_image(self):

        if self.mode == 'digit_recognition':
            self.loaded_image_label = int(self.image_path.split('/')[-1].split('_')[1])
            # load the image
            self.loaded_image_pt = torch.from_numpy(np.asarray(Image.open(self.image_path)))[
                :, :, None
            ]
            self.loaded_image_name = self.image_path.split('/')[-1]
            # keep a copy to represent the current (rotated) version of the original images
            self.cur_image_pt = self.loaded_image_pt.clone()

        elif self.mode == 'object_detection':
            self.label_path = (
                self.image_path.replace('images', 'labels')
                .replace('jpg', 'npy')
                .replace('jpg', 'npy')
            )
            self.loaded_image_label = np.load(self.label_path)
            # loaded image label is in original coco classes defined by original_coco_names
            # convert to custom names
            for i in range(len(self.loaded_image_label)):
                self.loaded_image_label[i, -1] = self.custom_coco_names.index(
                    self.original_coco_names[int(self.loaded_image_label[i, -1])]
                )

            # the center of the bounding box is the center of cropped image
            # we know that we only have one object, x is column, y is row
            self.center_x = int(
                (self.loaded_image_label[0, 0] + self.loaded_image_label[0, 2]) // 2
            )
            self.center_y = int(
                (self.loaded_image_label[0, 1] + self.loaded_image_label[0, 3]) // 2
            )

            # load the image
            self.loaded_image_pt = torch.from_numpy(
                np.asarray(Image.open(self.image_path).convert('RGB'), dtype=np.uint8)
            )
            self.loaded_image_name = self.image_path.split('/')[-1]

            # take the cropped part of the entire input image to put in display image
            self.cur_image_pt = self.loaded_image_pt[
                self.center_y - self.image_size : self.center_y + self.image_size,
                self.center_x - self.image_size : self.center_x + self.image_size,
                :,
            ]

        elif self.mode == 'piv':
            # keep the PIL image version of the loaded images in this mode because they are saved as gif using PIL
            # pre-trained PIV-LiteFlowNet-en takes 3 channel images
            self.loaded_image_1_pil = Image.open(self.image_1_path).convert('RGB')
            self.loaded_image_2_pil = Image.open(self.image_2_path).convert('RGB')

            # create a blank PIL image for gif purpose
            self.blank_image_pil = Image.fromarray(
                np.zeros((self.image_size, self.image_size, 3)), 'RGB'
            )

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
            other_images_pil = [
                self.loaded_image_1_pil,
                self.loaded_image_2_pil,
                self.loaded_image_2_pil,
                self.blank_image_pil,
            ]
            self.gif_path = os.path.join(
                self.cache_dir, self.loaded_image_1_name.split('.')[0] + '.gif'
            )
            self.loaded_image_1_pil.save(
                fp=self.gif_path,
                format='GIF',
                append_images=other_images_pil,
                save_all=True,
                duration=300,
                loop=0,
            )

            # load the ground truth flow field
            self.label_path = self.image_1_path.replace('img1', 'flow').replace('tif', 'flo')
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

            if not self.demo:
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

                # initialize layout
                self.init_aggregate_result_layout()

            self.dataset_name = text
            print('Loaded dataset:', self.dataset_name)
            self.data_mode = 'aggregate'
            # index 0 is the prompt
            self.dataset_index = self.aggregate_image_menu.currentIndex() - 1
            self.dataset_dir = self.aggregate_data_dirs[self.dataset_index]
            print(f'Loaded data from {self.dataset_dir}')
            # in digit recognition, all the images are loaded
            if self.mode == 'digit_recognition':
                # load the image and scale the size
                # get all the image paths from the directory
                self.all_images_paths = glob.glob(os.path.join(self.dataset_dir, '*.png'))
                self.loaded_images_pt = torch.zeros(
                    len(
                        self.all_images_paths,
                    ),
                    self.image_size,
                    self.image_size,
                    1,
                )
                self.loaded_images_names = []
                self.loaded_images_labels = torch.zeros(
                    len(self.all_images_paths), dtype=torch.int64
                )

                for i, cur_image_path in enumerate(self.all_images_paths):
                    self.loaded_images_pt[i] = torch.from_numpy(
                        np.asarray(Image.open(cur_image_path))
                    )[:, :, None]
                    self.loaded_images_names.append(cur_image_path.split('/')[-1])
                    self.loaded_images_labels[i] = int(cur_image_path.split('/')[-1].split('_')[1])

            # in object detection, only all the image paths are loaded
            elif self.mode == 'object_detection':
                # all the images and labels paths
                self.all_images_paths = glob.glob(
                    os.path.join(self.dataset_dir, 'images', '*.jpg')
                )
                self.all_labels_paths = []
                for cur_image_path in self.all_images_paths:
                    cur_label_path = cur_image_path.replace('images', 'labels').replace(
                        'jpg', 'npy'
                    )
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
                self.all_images_2_paths = [
                    cur_path.replace('img1', 'img2') for cur_path in self.all_images_1_paths
                ]
                self.all_labels_paths = [
                    cur_path.replace('img1', 'flow').replace('tif', 'flo')
                    for cur_path in self.all_images_1_paths
                ]
                # flow type of each image pair
                self.loaded_images_labels = [
                    cur_path.split('/')[-1].split('_')[0] for cur_path in self.all_images_1_paths
                ]

            # check the data to be ready
            self.data_existed = True

            # show the run button when data is loaded
            if not self.run_button_existed:
                # run button
                # buttons layout for run model
                self.run_button_layout = QtWidgets.QGridLayout()
                # demo mode has "virtual" run button, not in layout
                if not self.demo:
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

            elif not self.demo and self.run_button_existed:
                self.run_button.setText('Analyze model')

            elif self.demo:
                # the models have default, just run
                self.run_button_clicked()

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
                self.cur_display_image = nero_utilities.tensor_to_qt_image(
                    self.cur_image_pt, self.display_image_size, revert_color=True
                )
                # additional preparation required for MNIST
                self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)

            elif self.mode == 'object_detection':
                # prepare image path
                self.image_index = self.coco_classes.index(text.split(' ')[0])
                self.image_path = self.single_images_paths[self.image_index]
                self.load_single_image()

                # convert to QImage for display purpose
                self.cur_display_image = nero_utilities.tensor_to_qt_image(
                    self.cur_image_pt, self.display_image_size
                )

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
                    self.model_1_path = glob.glob(
                        os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt')
                    )[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(
                        self.mode, 'non-eqv', self.model_1_path
                    )
                elif text == 'E2CNN model':
                    self.model_1_path = glob.glob(
                        os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt')
                    )[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(
                        self.mode, 'rot-eqv', self.model_1_path
                    )
                elif text == 'Data Aug':
                    self.model_1_path = glob.glob(
                        os.path.join(
                            os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'
                        )
                    )[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(
                        self.mode, 'aug-eqv', self.model_1_path
                    )

                print('Model 1 path:', self.model_1_path)

            elif self.mode == 'object_detection':
                if '%' in self.model_1_name:
                    self.model_1_cache_name = self.model_1_name.split('%')[0]
                else:
                    self.model_1_cache_name = self.model_1_name.split('-')[0]

                if text == '0% jittering':
                    self.model_1_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_0-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_1_path
                    )
                    print('Model 1 path:', self.model_1_path)
                elif text == '20% jittering':
                    self.model_1_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_20-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_1_path
                    )
                    print('Model 1 path:', self.model_1_path)
                elif text == '40% jittering':
                    self.model_1_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_40-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_1_path
                    )
                    print('Model 1 path:', self.model_1_path)
                elif text == '60% jittering':
                    self.model_1_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_60-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_1_path
                    )
                    print('Model 1 path:', self.model_1_path)
                elif text == '80% jittering':
                    self.model_1_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_80-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_1_path
                    )
                    print('Model 1 path:', self.model_1_path)
                elif text == '100% jittering':
                    self.model_1_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_100-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_1 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_1_path
                    )
                    print('Model 1 path:', self.model_1_path)
                elif text == 'Pre-trained':
                    self.model_1_path = None
                    self.model_1 = nero_run_model.load_model(
                        self.mode, 'pre_trained', self.model_1_path
                    )
                    print('Model 1 path: Downloaded from PyTorch')

            elif self.mode == 'piv':
                self.model_1_cache_name = self.model_1_name.split('-')[1]
                if text == 'PIV-LiteFlowNet-en':
                    self.model_1_path = glob.glob(
                        os.path.join(
                            os.getcwd(), 'example_models', self.mode, 'PIV-LiteFlowNet-en', f'*.pt'
                        )
                    )[0]
                    self.model_1 = nero_run_model.load_model(
                        self.mode, self.model_1_name, self.model_1_path
                    )
                    print('Model 1 path:', self.model_1_path)

                elif text == 'Gunnar-Farneback':
                    # Gunnar-Farneback does not need model path
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
                    self.run_model_single()
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
                    self.model_2_path = glob.glob(
                        os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pt')
                    )[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model('non_eqv', self.model_2_path)
                elif text == 'E2CNN model':
                    self.model_2_path = glob.glob(
                        os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pt')
                    )[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model('rot_eqv', self.model_2_path)
                elif text == 'Data Aug':
                    self.model_2_path = glob.glob(
                        os.path.join(
                            os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'
                        )
                    )[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model('aug_eqv', self.model_2_path)

                print('Model 2 path:', self.model_2_path)

            elif self.mode == 'object_detection':
                if '%' in self.model_2_name:
                    self.model_2_cache_name = self.model_2_name.split('%')[0]
                else:
                    self.model_2_cache_name = self.model_2_name.split('-')[0]

                if text == '0% jittering':
                    self.model_2_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_0-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_2_path
                    )
                    print('Model 2 path:', self.model_2_path)
                elif text == '20% jittering':
                    self.model_2_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_20-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_2_path
                    )
                    print('Model 2 path:', self.model_2_path)
                elif text == '40% jittering':
                    self.model_2_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_40-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_2_path
                    )
                    print('Model 2 path:', self.model_2_path)
                elif text == '60% jittering':
                    self.model_2_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_60-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_2_path
                    )
                    print('Model 2 path:', self.model_2_path)
                elif text == '80% jittering':
                    self.model_2_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_80-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_2_path
                    )
                    print('Model 2 path:', self.model_2_path)
                elif text == '100% jittering':
                    self.model_2_path = glob.glob(
                        os.path.join(
                            os.getcwd(),
                            'example_models',
                            self.mode,
                            'custom_trained',
                            f'object_100-jittered',
                            '*.pth',
                        )
                    )[0]
                    # reload model
                    self.model_2 = nero_run_model.load_model(
                        self.mode, 'custom_trained', self.model_2_path
                    )
                    print('Model 2 path:', self.model_2_path)
                elif text == 'Pre-trained':
                    self.model_2_path = None
                    self.model_2 = nero_run_model.load_model(
                        self.mode, 'pre_trained', self.model_2_path
                    )
                    print('Model 2 path: Downloaded from PyTorch')

            elif self.mode == 'piv':
                self.model_1_cache_name = self.model_1_name.split('-')[1]
                if text == 'PIV-LiteFlowNet-en':
                    self.model_1_path = glob.glob(
                        os.path.join(
                            os.getcwd(), 'example_models', self.mode, 'PIV-LiteFlowNet-en', f'*.pt'
                        )
                    )[0]
                    self.model_1 = nero_run_model.load_model(
                        self.mode, self.model_1_name, self.model_1_path
                    )
                    print('Model 1 path:', self.model_1_path)

                elif text == 'Gunnar-Farneback':
                    # Gunnar-Farneback does not need model path
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
                    self.run_model_single()
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
        if self.demo:
            self.demo_layout = QtWidgets.QGridLayout()
            if self.mode == 'piv':
                self.demo_layout.setHorizontalSpacing(10)
                self.demo_layout.setVerticalSpacing(0)
            else:
                self.demo_layout.setHorizontalSpacing(0)
                self.demo_layout.setVerticalSpacing(0)
        else:
            self.load_menu_layout = QtWidgets.QGridLayout()

        # draw text
        model_pixmap = QPixmap(350, 50)
        model_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(model_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        if self.mode == 'piv':
            painter.drawText(0, 0, 350, 50, QtGui.Qt.AlignLeft, 'Dataset:')
        else:
            painter.drawText(0, 0, 350, 50, QtGui.Qt.AlignLeft, 'Data Set:')
        painter.end()

        # create label to contain the texts
        self.model_label = QLabel(self)
        self.model_label.setFixedSize(QtCore.QSize(300, 50))
        self.model_label.setPixmap(model_pixmap)
        # add to the layout
        if self.demo:
            self.demo_layout.addWidget(self.model_label, 0, 0)
        else:
            self.load_menu_layout.addWidget(self.model_label, 0, 2)

        # aggregate images loading drop down menu
        self.aggregate_image_menu = QtWidgets.QComboBox()
        self.aggregate_image_menu.setFixedSize(QtCore.QSize(220, 50))
        self.aggregate_image_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        self.aggregate_image_menu.addItem('Input dataset')

        # data dir (sorted in a way that bigger dataset first)
        if self.mode == 'digit_recognition':
            self.aggregate_data_dirs = sorted(
                glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, f'MNIST*')),
                reverse=True,
            )
        elif self.mode == 'object_detection':
            self.aggregate_data_dirs = sorted(
                glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, f'COCO*')),
                reverse=True,
            )
        elif self.mode == 'piv':
            self.aggregate_data_dirs = sorted(
                glob.glob(os.path.join(os.getcwd(), 'example_data', self.mode, f'JHTDB*')),
                reverse=True,
            )
        # load all images in the folder
        for i in range(len(self.aggregate_data_dirs)):
            self.aggregate_image_menu.addItem(self.aggregate_data_dirs[i].split('/')[-1])

        # set default to the first test dataset
        if self.demo:
            self.aggregate_image_menu.setCurrentIndex(1)
        # set default to the prompt/description
        else:
            self.aggregate_image_menu.setCurrentIndex(0)

        # connect the drop down menu with actions
        self.aggregate_image_menu.currentTextChanged.connect(aggregate_dataset_selection_changed)
        self.aggregate_image_menu.setEditable(True)
        self.aggregate_image_menu.lineEdit().setReadOnly(True)
        self.aggregate_image_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        if self.demo:
            aggregate_image_menu_layout = QtWidgets.QHBoxLayout()
            aggregate_image_menu_layout.setContentsMargins(150, 0, 0, 0)
            aggregate_image_menu_layout.addWidget(self.aggregate_image_menu)
            self.demo_layout.addLayout(aggregate_image_menu_layout, 0, 0)
        else:
            self.load_menu_layout.addWidget(self.aggregate_image_menu, 0, 3)

        # single image loading drop down menu
        self.image_menu = QtWidgets.QComboBox()
        self.image_menu.setFixedSize(QtCore.QSize(400, 50))
        self.image_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        self.image_menu.addItem('Input image')

        if self.mode == 'digit_recognition':
            self.single_images_paths = []
            # add a image of each class
            for i in range(10):
                cur_image_path = glob.glob(
                    os.path.join(
                        os.getcwd(), 'example_data', self.mode, 'single', f'label_{i}*.png'
                    )
                )[0]
                self.single_images_paths.append(cur_image_path)
                self.image_menu.addItem(QtGui.QIcon(cur_image_path), f'Image {i}')

            self.image_menu.setCurrentIndex(0)

        elif self.mode == 'object_detection':
            self.single_images_paths = []
            self.data_config_paths = []
            self.coco_classes = ['car', 'bottle', 'cup', 'chair', 'book']
            # add a image of each class
            for i, cur_class in enumerate(self.coco_classes):
                cur_image_path = glob.glob(
                    os.path.join(
                        os.getcwd(),
                        'example_data',
                        self.mode,
                        'single',
                        'images',
                        f'{cur_class}*.jpg',
                    )
                )[0]
                self.single_images_paths.append(cur_image_path)
                self.image_menu.addItem(QtGui.QIcon(cur_image_path), f'{cur_class} image')

            self.image_menu.setCurrentIndex(0)

        elif self.mode == 'piv':
            # different flow types
            self.flow_types = ['Uniform', 'Backstep', 'Cylinder', 'SQG', 'DNS', 'Isotropic']
            self.single_images_1_paths = []
            self.single_images_2_paths = []
            self.single_labels_paths = []
            # image pairs
            for i, flow_type in enumerate(self.flow_types):
                cur_image_1_path = glob.glob(
                    os.path.join(
                        os.getcwd(), 'example_data', self.mode, 'single', f'{flow_type}*img1.tif'
                    )
                )[0]
                self.single_images_1_paths.append(cur_image_1_path)
                self.single_images_2_paths.append(cur_image_1_path.replace('img1', 'img2'))
                self.single_labels_paths.append(
                    cur_image_1_path.replace('img1', 'flow').replace('tif', 'flo')
                )

                # add flow to the menu
                self.image_menu.addItem(f'Image pair {i} - {flow_type}')

            self.image_menu.setCurrentIndex(0)

        # connect the drop down menu with actions
        self.image_menu.currentTextChanged.connect(single_image_selection_changed)
        self.image_menu.setEditable(True)
        self.image_menu.lineEdit().setReadOnly(True)
        self.image_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        # not show in the demo mode
        if not self.demo:
            self.load_menu_layout.addWidget(self.image_menu, 1, 3)

        # init flag to inidicate if an image has ever been loaded
        self.data_existed = False
        self.image_existed = False

        # load models choices
        # draw text
        # if self.mode == 'digit_recognition':
        model_selection_pixmap = QPixmap(450, 50)
        model_selection_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(model_selection_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 450, 50, QtGui.Qt.AlignLeft, 'Models in Comparisons: ')
        painter.end()
        # create label to contain the texts
        self.model_selection_label = QLabel(self)
        self.model_selection_label.setFixedSize(QtCore.QSize(500, 50))
        # self.class_selection_label.setAlignment(QtCore.Qt.AlignLeft)
        # self.class_selection_label.setWordWrap(True)
        # self.class_selection_label.setTextFormat(QtGui.Qt.AutoText)
        self.model_selection_label.setPixmap(model_selection_pixmap)
        if self.mode == 'piv':
            self.model_selection_label.setContentsMargins(0, 0, 0, 0)
        else:
            self.model_selection_label.setContentsMargins(20, 0, 0, 0)

        # model 1
        # graphic representation
        self.model_1_label = QLabel(self)
        self.model_1_label.setContentsMargins(0, 0, 0, 0)
        self.model_1_label.setAlignment(QtCore.Qt.AlignCenter)
        if self.mode == 'digit_recognition' or self.mode == 'object_detection':
            model_1_icon = QPixmap(25, 25)
            model_1_icon.fill(QtCore.Qt.white)
            # draw model representation
            painter = QtGui.QPainter(model_1_icon)
            draw_circle(painter, 12, 12, 10, 'blue')
        elif self.mode == 'piv':
            model_1_icon = QtGui.QIcon('symbols/top_row_icon.png')

        self.model_1_menu = QtWidgets.QComboBox()
        self.model_1_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        if self.mode == 'digit_recognition':
            self.model_1_menu.setFixedSize(QtCore.QSize(200, 50))
            self.model_1_menu.addItem(model_1_icon, 'Original')
            self.model_1_menu.addItem(model_1_icon, 'E2CNN')
            self.model_1_menu.addItem(model_1_icon, 'Data Aug')
            self.model_1_menu.setCurrentText('Original model')
        elif self.mode == 'object_detection':
            self.model_1_menu.setFixedSize(QtCore.QSize(250, 50))
            self.model_1_menu.addItem(model_1_icon, '0% jittering')
            self.model_1_menu.addItem(model_1_icon, '20% jittering')
            self.model_1_menu.addItem(model_1_icon, '40% jittering')
            self.model_1_menu.addItem(model_1_icon, '60% jittering')
            self.model_1_menu.addItem(model_1_icon, '80% jittering')
            self.model_1_menu.addItem(model_1_icon, '100% jittering')
            self.model_1_menu.addItem(model_1_icon, 'Pre-trained')
            self.model_1_menu.setCurrentText('0% jittering')
            self.model_1_name = '0% jittering'
        elif self.mode == 'piv':
            self.model_1_menu.setFixedSize(QtCore.QSize(400, 50))
            self.model_1_menu.setIconSize(QtCore.QSize(50, 50))
            self.model_1_menu.addItem('PIV-LiteFlowNet-en')
            self.model_1_menu.addItem('Gunnar-Farneback')
            self.model_1_menu.setItemIcon(0, model_1_icon)
            self.model_1_menu.setItemIcon(1, model_1_icon)
            self.model_1_menu.setCurrentText('PIV-LiteFlowNet-en')

        # connect the drop down menu with actions
        self.model_1_menu.currentTextChanged.connect(model_1_selection_changed)
        self.model_1_menu.setEditable(True)
        self.model_1_menu.lineEdit().setReadOnly(True)
        self.model_1_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        if self.demo:
            if self.mode == 'digit_recognition':
                model_menus_layout = QtWidgets.QHBoxLayout()
                model_menus_layout.setContentsMargins(0, 0, 0, 0)
                model_menus_layout.addWidget(self.model_1_menu)
            elif self.mode == 'object_detection':
                model_menus_layout = QtWidgets.QGridLayout()
                model_menus_layout.addWidget(self.model_1_menu, 0, 0)
            elif self.mode == 'piv':
                model_menus_layout = QtWidgets.QGridLayout()
                model_menus_layout.addWidget(self.model_1_menu, 0, 0)
        else:
            self.load_menu_layout.addWidget(self.model_1_menu, 2, 3)

        # model 2
        # graphic representation
        self.model_2_label = QLabel(self)
        self.model_2_label.setContentsMargins(0, 0, 0, 0)
        self.model_2_label.setAlignment(QtCore.Qt.AlignCenter)
        if self.mode == 'digit_recognition' or self.mode == 'object_detection':
            model_2_icon = QPixmap(25, 25)
            model_2_icon.fill(QtCore.Qt.white)
            # draw model representation
            painter = QtGui.QPainter(model_2_icon)
            draw_circle(painter, 12, 12, 10, 'magenta')
        elif self.mode == 'piv':
            model_2_icon = QtGui.QIcon('symbols/bottom_row_icon.png')

        self.model_2_menu = QtWidgets.QComboBox()
        self.model_2_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        self.model_2_menu.setEditable(True)
        self.model_2_menu.lineEdit().setReadOnly(True)
        self.model_2_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        if self.mode == 'digit_recognition':
            self.model_2_menu.setFixedSize(QtCore.QSize(200, 50))
            self.model_2_menu.addItem(model_2_icon, 'Original')
            self.model_2_menu.addItem(model_2_icon, 'E2CNN')
            self.model_2_menu.addItem(model_2_icon, 'Data Aug')
            # model_2_menu.setCurrentText('E2CNN model')
            self.model_2_menu.setCurrentText('Data Aug')
        elif self.mode == 'object_detection':
            self.model_2_menu.setFixedSize(QtCore.QSize(250, 50))
            self.model_2_menu.addItem(model_2_icon, '0% jittering')
            self.model_2_menu.addItem(model_2_icon, '20% jittering')
            self.model_2_menu.addItem(model_2_icon, '40% jittering')
            self.model_2_menu.addItem(model_2_icon, '60% jittering')
            self.model_2_menu.addItem(model_2_icon, '80% jittering')
            self.model_2_menu.addItem(model_2_icon, '100% jittering')
            self.model_2_menu.addItem(model_2_icon, 'Pre-trained')
            self.model_2_menu.setCurrentText('Pre-trained')
            self.model_2_name = 'Pre-trained'
        elif self.mode == 'piv':
            self.model_2_menu.setFixedSize(QtCore.QSize(400, 50))
            self.model_2_menu.setIconSize(QtCore.QSize(50, 50))
            self.model_2_menu.addItem(model_2_icon, 'PIV-LiteFlowNet-en')
            self.model_2_menu.addItem(model_2_icon, 'Gunnar-Farneback')
            self.model_2_menu.setItemIcon(0, model_2_icon)
            self.model_2_menu.setItemIcon(1, model_2_icon)
            self.model_2_menu.setCurrentText('Gunnar-Farneback')

        # connect the drop down menu with actions
        self.model_2_menu.currentTextChanged.connect(model_2_selection_changed)
        if self.demo:
            if self.mode == 'digit_recognition':
                model_menus_layout.addWidget(self.model_2_menu)
                self.demo_layout.addWidget(self.model_selection_label, 1, 2)
                self.demo_layout.addLayout(model_menus_layout, 2, 2)
            elif self.mode == 'object_detection':
                model_menus_layout.addWidget(self.model_2_menu, 0, 1)
                self.demo_layout.addWidget(self.model_selection_label, 1, 2)
                self.demo_layout.addLayout(model_menus_layout, 2, 2)
            elif self.mode == 'piv':
                model_menus_layout.addWidget(self.model_2_menu, 0, 1)
                self.demo_layout.addWidget(self.model_selection_label, 1, 2)
                self.demo_layout.addLayout(model_menus_layout, 2, 2, 1, 2)
                # model_2_menu_layout = QtWidgets.QHBoxLayout()
                # model_2_menu_layout.addWidget(self.model_2_menu)
                # model_2_menu_layout.setContentsMargins(0, 0, 0, 0)
                # self.demo_layout.addLayout(model_2_menu_layout, 2, 3)
        else:
            self.load_menu_layout.addWidget(self.model_2_menu, 3, 3)

        # add this layout to the general layout
        if self.demo:
            self.layout.addLayout(self.demo_layout, 0, 0)
        else:
            self.layout.addLayout(self.load_menu_layout, 0, 1)

        # demo mode selects the aggregate dataset to start three-level view
        if self.demo:
            aggregate_dataset_selection_changed(self.aggregate_image_menu.currentText())
            self.use_cache_checkbox.setChecked(True)
            # the models have default, just run
            self.run_button_clicked()

    def init_aggregate_result_layout(self):

        # loaded images and model result layout
        self.aggregate_result_layout = QtWidgets.QGridLayout()
        # self.aggregate_result_layout.setContentsMargins(30, 50, 30, 50)

        # if model result ever existed
        self.aggregate_result_existed = False

        # batch size when running in aggregate mode
        if self.mode == 'digit_recognition':
            # add to general layout
            self.layout.addLayout(self.aggregate_result_layout, 1, 0, 3, 3)
        elif self.mode == 'object_detection':
            # add to general layout
            self.layout.addLayout(self.aggregate_result_layout, 1, 0, 3, 3)
        elif self.mode == 'piv':
            self.layout.addLayout(self.aggregate_result_layout, 1, 0, 3, 3)

    def init_single_result_layout(self):
        # loaded images and model result layout
        self.single_result_layout = QtWidgets.QGridLayout()
        self.single_result_layout.setContentsMargins(30, 50, 30, 50)

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
            self.run_model_single()
            self.single_result_existed = True

    # initialize digit selection control drop down menu
    def init_aggregate_plot_control(self):

        self.aggregate_plot_control_existed = True
        # mark if dimension reduction algorithm has been run
        self.dr_result_existed = False

        # aggregate class selection drop-down menu
        @QtCore.Slot()
        def aggregate_class_selection_changed(text):
            # re-initialize the scatter plot
            self.dr_result_existed = False
            # update the current digit selection
            # for digit recognition (MNIST)
            if self.mode == 'digit_recognition':
                if text.split(' ')[0] == 'All':
                    self.class_selection = 'all'
                elif text.split(' ')[0] == 'Digit':
                    self.class_selection = int(text.split(' ')[-1])

                # display the plot
                self.display_mnist_aggregate_result()

            # for object detection (COCO)
            elif self.mode == 'object_detection':
                if text.split(' ')[0] == 'All':
                    self.class_selection = 'all'
                else:
                    self.class_selection = text

                # display the plot
                self.display_coco_aggregate_result()

            # for piv
            elif self.mode == 'piv':
                # select different flows
                if text.split(' ')[0] == 'All':
                    self.class_selection = 'all'
                else:
                    self.class_selection = text

                # display the plot
                self.display_piv_aggregate_result()

            # after change class, run new dimension reduction if previously run
            if self.demo or self.dr_result_existed:
                self.run_dimension_reduction()

        @QtCore.Slot()
        def use_consensus_checkbox_clicked():
            if self.use_consensus_checkbox.isChecked():
                self.use_consensus = True
                print(f'Plotting {self.quantity_name} with respect to consensus')
            else:
                self.use_consensus = False
                print(f'Plotting {self.quantity_name} with respect to ground truth')

            # update plotting quantities
            for y in range(len(self.y_translation)):
                for x in range(len(self.x_translation)):
                    # model 1
                    all_samples_conf_sum_1 = []
                    all_sampels_conf_correctness_sum_1 = []
                    all_samples_iou_sum_1 = []
                    all_sampels_iou_correctness_sum_1 = []
                    all_samples_precision_sum_1 = []
                    all_samples_recall_sum_1 = []
                    all_samples_F_measure_sum_1 = []

                    # model 2
                    all_samples_conf_sum_2 = []
                    all_sampels_conf_correctness_sum_2 = []
                    all_samples_iou_sum_2 = []
                    all_sampels_iou_correctness_sum_2 = []
                    all_samples_precision_sum_2 = []
                    all_samples_recall_sum_2 = []
                    all_samples_F_measure_sum_2 = []

                    if self.use_consensus:
                        for i in range(len(self.aggregate_consensus_outputs_1[y, x])):
                            # model 1
                            all_samples_conf_sum_1.append(
                                self.aggregate_consensus_outputs_1[y, x][i][0, 4]
                            )
                            all_sampels_conf_correctness_sum_1.append(
                                self.aggregate_consensus_outputs_1[y, x][i][0, 4]
                                * self.aggregate_consensus_outputs_1[y, x][i][0, 7]
                            )
                            all_samples_iou_sum_1.append(
                                self.aggregate_consensus_outputs_1[y, x][i][0, 6]
                            )
                            all_sampels_iou_correctness_sum_1.append(
                                self.aggregate_consensus_outputs_1[y, x][i][0, 6]
                                * self.aggregate_consensus_outputs_1[y, x][i][0, 7]
                            )
                            all_samples_precision_sum_1.append(
                                self.aggregate_consensus_precision_1[y, x][i]
                            )
                            all_samples_recall_sum_1.append(
                                self.aggregate_consensus_recall_1[y, x][i]
                            )
                            all_samples_F_measure_sum_1.append(
                                self.aggregate_consensus_F_measure_1[y, x][i]
                            )

                            # model 2
                            all_samples_conf_sum_2.append(
                                self.aggregate_consensus_outputs_2[y, x][i][0, 4]
                            )
                            all_sampels_conf_correctness_sum_2.append(
                                self.aggregate_consensus_outputs_2[y, x][i][0, 4]
                                * self.aggregate_consensus_outputs_2[y, x][i][0, 7]
                            )
                            all_samples_iou_sum_2.append(
                                self.aggregate_consensus_outputs_2[y, x][i][0, 6]
                            )
                            all_sampels_iou_correctness_sum_2.append(
                                self.aggregate_consensus_outputs_2[y, x][i][0, 6]
                                * self.aggregate_consensus_outputs_2[y, x][i][0, 7]
                            )
                            all_samples_precision_sum_2.append(
                                self.aggregate_consensus_precision_2[y, x][i]
                            )
                            all_samples_recall_sum_2.append(
                                self.aggregate_consensus_recall_2[y, x][i]
                            )
                            all_samples_F_measure_sum_2.append(
                                self.aggregate_consensus_F_measure_2[y, x][i]
                            )
                    else:
                        for i in range(len(self.aggregate_outputs_1[y, x])):
                            # model 1
                            all_samples_conf_sum_1.append(self.aggregate_outputs_1[y, x][i][0, 4])
                            all_sampels_conf_correctness_sum_1.append(
                                self.aggregate_outputs_1[y, x][i][0, 4]
                                * self.aggregate_outputs_1[y, x][i][0, 7]
                            )
                            all_samples_iou_sum_1.append(self.aggregate_outputs_1[y, x][i][0, 6])
                            all_sampels_iou_correctness_sum_1.append(
                                self.aggregate_outputs_1[y, x][i][0, 6]
                                * self.aggregate_outputs_1[y, x][i][0, 7]
                            )
                            all_samples_precision_sum_1.append(self.aggregate_precision_1[y, x][i])
                            all_samples_recall_sum_1.append(self.aggregate_recall_1[y, x][i])
                            all_samples_F_measure_sum_1.append(self.aggregate_F_measure_1[y, x][i])

                            # model 2
                            all_samples_conf_sum_2.append(self.aggregate_outputs_2[y, x][i][0, 4])
                            all_sampels_conf_correctness_sum_2.append(
                                self.aggregate_outputs_2[y, x][i][0, 4]
                                * self.aggregate_outputs_2[y, x][i][0, 7]
                            )
                            all_samples_iou_sum_2.append(self.aggregate_outputs_2[y, x][i][0, 6])
                            all_sampels_iou_correctness_sum_2.append(
                                self.aggregate_outputs_2[y, x][i][0, 6]
                                * self.aggregate_outputs_2[y, x][i][0, 7]
                            )
                            all_samples_precision_sum_2.append(self.aggregate_precision_2[y, x][i])
                            all_samples_recall_sum_2.append(self.aggregate_recall_2[y, x][i])
                            all_samples_F_measure_sum_2.append(self.aggregate_F_measure_2[y, x][i])

                    # take the average result
                    # model 1
                    self.aggregate_avg_conf_1[y, x] = np.mean(all_samples_conf_sum_1)
                    self.aggregate_avg_conf_correctness_1[y, x] = np.mean(
                        all_sampels_conf_correctness_sum_1
                    )
                    self.aggregate_avg_iou_1[y, x] = np.mean(all_samples_iou_sum_1)
                    self.aggregate_avg_iou_correctness_1[y, x] = np.mean(
                        all_sampels_iou_correctness_sum_1
                    )
                    self.aggregate_avg_precision_1[y, x] = np.mean(all_samples_precision_sum_1)
                    self.aggregate_avg_recall_1[y, x] = np.mean(all_samples_recall_sum_1)
                    self.aggregate_avg_F_measure_1[y, x] = np.mean(all_samples_F_measure_sum_1)

                    # model 2
                    self.aggregate_avg_conf_2[y, x] = np.mean(all_samples_conf_sum_2)
                    self.aggregate_avg_conf_correctness_2[y, x] = np.mean(
                        all_sampels_conf_correctness_sum_2
                    )
                    self.aggregate_avg_iou_2[y, x] = np.mean(all_samples_iou_sum_2)
                    self.aggregate_avg_iou_correctness_2[y, x] = np.mean(
                        all_sampels_iou_correctness_sum_2
                    )
                    self.aggregate_avg_precision_2[y, x] = np.mean(all_samples_precision_sum_2)
                    self.aggregate_avg_recall_2[y, x] = np.mean(all_samples_recall_sum_2)
                    self.aggregate_avg_F_measure_2[y, x] = np.mean(all_samples_F_measure_sum_2)

            # keep existing plot quantity selection
            if self.quantity_name == 'Confidence':
                self.cur_aggregate_plot_quantity_1 = self.aggregate_avg_conf_1
                self.cur_aggregate_plot_quantity_2 = self.aggregate_avg_conf_2
            elif self.quantity_name == 'IOU':
                self.cur_aggregate_plot_quantity_1 = self.aggregate_avg_iou_correctness_1
                self.cur_aggregate_plot_quantity_2 = self.aggregate_avg_iou_correctness_2
            elif self.quantity_name == 'Conf*IOU':
                self.cur_aggregate_plot_quantity_1 = (
                    self.aggregate_avg_conf_1 * self.aggregate_avg_iou_correctness_1
                )
                self.cur_aggregate_plot_quantity_2 = (
                    self.aggregate_avg_conf_2 * self.aggregate_avg_iou_correctness_2
                )
            else:
                raise Exception(f'Unknown quantity {self.quantity_name}')

            # re-display the aggregate heatmap
            self.draw_coco_nero(mode='aggregate')

            # re-run dimension reduction and show result
            if self.dr_result_existed:
                self.run_dimension_reduction()

            # update single NERO plot as well
            for y in range(len(self.y_translation)):
                for x in range(len(self.x_translation)):
                    if self.quantity_name == 'Conf*IOU':
                        if self.use_consensus:
                            self.cur_single_plot_quantity_1[y, x] = (
                                self.aggregate_consensus_outputs_1[y, x][self.image_index][0, 4]
                                * self.aggregate_consensus_outputs_1[y, x][self.image_index][0, 6]
                                * self.aggregate_consensus_outputs_1[y, x][self.image_index][0, 7]
                            )
                            self.cur_single_plot_quantity_2[y, x] = (
                                self.aggregate_consensus_outputs_2[y, x][self.image_index][0, 4]
                                * self.aggregate_consensus_outputs_2[y, x][self.image_index][0, 6]
                                * self.aggregate_consensus_outputs_1[y, x][self.image_index][0, 7]
                            )
                        else:
                            self.cur_single_plot_quantity_1[y, x] = (
                                self.aggregate_outputs_1[y, x][self.image_index][0, 4]
                                * self.aggregate_outputs_1[y, x][self.image_index][0, 6]
                                * self.aggregate_outputs_1[y, x][self.image_index][0, 7]
                            )
                            self.cur_single_plot_quantity_2[y, x] = (
                                self.aggregate_outputs_2[y, x][self.image_index][0, 4]
                                * self.aggregate_outputs_2[y, x][self.image_index][0, 6]
                                * self.aggregate_outputs_1[y, x][self.image_index][0, 7]
                            )
                    elif self.quantity_name == 'Confidence':
                        if self.use_consensus:
                            self.cur_single_plot_quantity_1[
                                y, x
                            ] = self.aggregate_consensus_outputs_1[y, x][self.image_index][0, 4]
                            self.cur_single_plot_quantity_2[
                                y, x
                            ] = self.aggregate_consensus_outputs_2[y, x][self.image_index][0, 4]
                        else:
                            self.cur_single_plot_quantity_1[y, x] = self.aggregate_outputs_1[y, x][
                                self.image_index
                            ][0, 4]
                            self.cur_single_plot_quantity_2[y, x] = self.aggregate_outputs_2[y, x][
                                self.image_index
                            ][0, 4]
                    elif self.quantity_name == 'IOU':
                        if self.use_consensus:
                            self.cur_single_plot_quantity_1[y, x] = (
                                self.aggregate_consensus_outputs_1[y, x][self.image_index][0, 6]
                                * self.aggregate_consensus_outputs_1[y, x][self.image_index][0, 7]
                            )
                            self.cur_single_plot_quantity_2[y, x] = (
                                self.aggregate_consensus_outputs_2[y, x][self.image_index][0, 6]
                                * self.aggregate_consensus_outputs_1[y, x][self.image_index][0, 7]
                            )
                        else:
                            self.cur_single_plot_quantity_1[y, x] = (
                                self.aggregate_outputs_1[y, x][self.image_index][0, 6]
                                * self.aggregate_outputs_1[y, x][self.image_index][0, 7]
                            )
                            self.cur_single_plot_quantity_2[y, x] = (
                                self.aggregate_outputs_2[y, x][self.image_index][0, 6]
                                * self.aggregate_outputs_1[y, x][self.image_index][0, 7]
                            )

            # re-display the heatmap
            self.draw_coco_nero(mode='single')

        @QtCore.Slot()
        def average_nero_checkbox_clicked():
            if self.average_nero_checkbox.isChecked():
                self.show_average = True
            else:
                self.show_average = False

            self.draw_piv_nero('aggregate')

            if self.single_result_existed:
                # show the previously selected detail area when in detail mode
                if self.show_average == False:
                    self.piv_heatmap_click_enable = True
                self.draw_piv_nero('single')

        # change different dimension reduction algorithms
        def dr_selection_changed(text):
            self.dr_selection = text

            # reset plot flag
            self.dr_result_existed = False

            # re-run dimension reduction and show result
            if self.dr_result_existed:
                self.run_dimension_reduction()

        # layout that controls the plotting items
        self.aggregate_plot_control_layout = QtWidgets.QGridLayout()
        # add plot control layout to general layout
        if not self.demo:
            self.aggregate_result_layout.addLayout(self.aggregate_plot_control_layout, 0, 0, 3, 1)

        # drop down menu on choosing the display class
        # draw text
        class_selection_pixmap = QPixmap(300, 50)
        class_selection_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(class_selection_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 350, 50, QtGui.Qt.AlignLeft, 'Subset: ')
        painter.end()
        # create label to contain the texts
        self.class_selection_label = QLabel(self)
        # self.class_selection_label.setContentsMargins(0, 50, 0, 0)
        self.class_selection_label.setFixedSize(QtCore.QSize(400, 50))
        # self.class_selection_label.setAlignment(QtCore.Qt.AlignLeft)
        # self.class_selection_label.setWordWrap(True)
        # self.class_selection_label.setTextFormat(QtGui.Qt.AutoText)
        self.class_selection_label.setPixmap(class_selection_pixmap)
        # add to the layout
        if self.demo:
            self.demo_layout.addWidget(self.class_selection_label, 1, 0)
        else:
            self.aggregate_plot_control_layout.addWidget(self.class_selection_label, 0, 0)

        self.class_selection_menu = QtWidgets.QComboBox()
        self.class_selection_menu.setFixedSize(QtCore.QSize(220, 50))
        self.class_selection_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        if self.mode == 'digit_recognition':
            self.class_selection_menu.addItem(f'All digits')
            # add all digits as items
            for i in range(10):
                self.class_selection_menu.addItem(f'Digit {i}')
        elif self.mode == 'object_detection':
            self.class_selection_menu.addItem(f'All objects')
            # add all classes as items
            for cur_class in self.coco_classes:
                self.class_selection_menu.addItem(f'{cur_class}')
        elif self.mode == 'piv':
            self.class_selection_menu.addItem(f'All types')
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
        self.class_selection_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        # add to local layout
        if self.demo:
            class_selection_menu_layout = QtWidgets.QHBoxLayout()
            class_selection_menu_layout.setContentsMargins(150, 0, 0, 0)
            class_selection_menu_layout.addWidget(self.class_selection_menu)
            self.demo_layout.addLayout(class_selection_menu_layout, 1, 0)
        else:
            self.aggregate_plot_control_layout.addWidget(self.class_selection_menu, 1, 0)

        # drop down menu on choosing the dimension reduction method
        # draw text
        dr_selection_pixmap = QPixmap(330, 60)
        dr_selection_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(dr_selection_pixmap)
        if self.mode == 'digit_recognition' or self.mode == 'object_detection':
            painter.setFont(QFont('Helvetica', 30))
            painter.drawText(0, 0, 330, 60, QtGui.Qt.AlignLeft, 'DR Plot Layout: ')
        elif self.mode == 'piv':
            painter.setFont(QFont('Helvetica', 30))
            painter.drawText(0, 0, 330, 60, QtGui.Qt.AlignLeft, 'DR Layout: ')
        painter.end()
        # create label to contain the texts
        self.dr_selection_label = QLabel(self)
        self.dr_selection_label.setFixedSize(QtCore.QSize(330, 60))
        self.dr_selection_label.setPixmap(dr_selection_pixmap)

        # add to the layout
        if self.demo:
            scatterplot_layout = QtWidgets.QHBoxLayout()
            scatterplot_layout.addWidget(self.dr_selection_label)
        else:
            self.load_menu_layout.addWidget(self.dr_selection_label, 1, 1)

        self.dr_selection_menu = QtWidgets.QComboBox()
        self.dr_selection_menu.setFixedSize(QtCore.QSize(150, 50))
        self.dr_selection_menu.setContentsMargins(0, 0, 0, 0)
        self.dr_selection_menu.setStyleSheet(
            'color: black; font-family: Helvetica; font-style: normal; font-size: 34px'
        )
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
        if self.demo:
            scatterplot_layout.addWidget(self.dr_selection_menu)
            self.demo_layout.addLayout(scatterplot_layout, 0, 1, 1, 1)
        else:
            self.aggregate_plot_control_layout.addWidget(self.dr_selection_menu, 2, 0)

        # push button on running PCA
        self.run_dr_button = QtWidgets.QPushButton('Run Dimension Reduction')
        self.run_dr_button.setStyleSheet('font-size: 18px')
        self.run_dr_button.setFixedSize(QtCore.QSize(250, 50))
        self.run_dr_button.clicked.connect(self.run_dimension_reduction)
        if not self.demo:
            self.aggregate_plot_control_layout.addWidget(self.run_dr_button, 3, 0)

        # add checkbox options for COCO and PIV
        if self.mode == 'object_detection' or self.mode == 'piv':
            self.checkbox_layout = QtWidgets.QVBoxLayout()
            self.checkbox_layout.setAlignment(QtGui.Qt.AlignTop)
            self.checkbox_layout.setContentsMargins(0, 0, 0, 0)

            # consensus
            self.use_consensus_checkbox = QtWidgets.QCheckBox('Use Consensus')
            self.use_consensus_checkbox.setStyleSheet(
                'color: black; font-style: normal; font-family: Helvetica; font-size: 24px;'
            )
            self.use_consensus_checkbox.setFixedSize(QtCore.QSize(300, 50))
            self.use_consensus_checkbox.stateChanged.connect(use_consensus_checkbox_clicked)
            # set conensus to default when we don't have ground truth labels
            if self.all_labels_paths == []:
                self.use_consensus = True
                self.use_consensus_checkbox.setChecked(self.use_consensus)
            else:
                self.use_consensus = False
                self.use_consensus_checkbox.setChecked(self.use_consensus)

            # add to layout
            self.checkbox_layout.addWidget(self.use_consensus_checkbox)

            # piv only averaged NERO
            if self.mode == 'piv':
                self.average_nero_checkbox = QtWidgets.QCheckBox('Show averaged NERO')
                self.average_nero_checkbox.setStyleSheet(
                    'color: black; font-style: normal; font-family: Helvetica; font-size: 24px;'
                )
                self.average_nero_checkbox.setFixedSize(QtCore.QSize(300, 50))
                self.average_nero_checkbox.stateChanged.connect(average_nero_checkbox_clicked)
                self.average_nero_checkbox.setChecked(False)
                if self.average_nero_checkbox.checkState() == QtCore.Qt.Checked:
                    self.show_average = True
                else:
                    self.show_average = False

                self.checkbox_layout.addWidget(self.average_nero_checkbox)

            if self.demo:
                self.demo_layout.addLayout(self.checkbox_layout, 2, 0, 1, 1)
            else:
                self.aggregate_plot_control_layout.addLayout(self.checkbox_layout, 6, 0)

    # run PCA on demand
    @QtCore.Slot()
    def run_dimension_reduction(self):

        # update the slider 1's text
        def update_slider_1_text():
            slider_1_text_pixmap = QPixmap(150, 50)
            slider_1_text_pixmap.fill(QtCore.Qt.white)
            painter = QtGui.QPainter(slider_1_text_pixmap)
            painter.setFont(QFont('Helvetica', 12))
            painter.drawText(
                0,
                0,
                150,
                50,
                QtGui.Qt.AlignCenter,
                f'{self.dr_result_selection_slider_1.value()+1}/{len(self.cur_class_indices)}',
            )
            painter.end()
            self.slider_1_text_label.setPixmap(slider_1_text_pixmap)

        # update the slider 2's text
        def update_slider_2_text():
            slider_2_text_pixmap = QPixmap(150, 50)
            slider_2_text_pixmap.fill(QtCore.Qt.white)
            painter = QtGui.QPainter(slider_2_text_pixmap)
            painter.setFont(QFont('Helvetica', 12))
            painter.drawText(
                0,
                0,
                150,
                50,
                QtGui.Qt.AlignCenter,
                f'{self.dr_result_selection_slider_2.value()+1}/{len(self.cur_class_indices)}',
            )
            painter.end()
            self.slider_2_text_label.setPixmap(slider_2_text_pixmap)

        # helper function on computing dimension reductions
        def dimension_reduce(high_dim, target_dim):

            if self.dr_selection == 'PCA':
                pca = PCA(n_components=target_dim, svd_solver='full')
                low_dim = pca.fit_transform(high_dim)
            elif self.dr_selection == 'ICA':
                ica = FastICA(n_components=target_dim, random_state=12)
                low_dim = ica.fit_transform(high_dim)
            elif self.dr_selection == 'ISOMAP':
                low_dim = manifold.Isomap(
                    n_neighbors=5, n_components=target_dim, n_jobs=-1
                ).fit_transform(high_dim)
            elif self.dr_selection == 't-SNE':
                low_dim = TSNE(n_components=target_dim, n_iter=250).fit_transform(high_dim)
            elif self.dr_selection == 'UMAP':
                low_dim = umap.UMAP(
                    n_neighbors=5, min_dist=0.3, n_components=target_dim
                ).fit_transform(high_dim)

            return low_dim

        # when clicked on the scatter plot item
        def low_dim_scatter_clicked(item=None, points=None):
            @QtCore.Slot()
            def slider_1_left_button_clicked():
                self.dr_result_selection_slider_1.setValue(
                    self.dr_result_selection_slider_1.value() - 1
                )
                # update the text
                update_slider_1_text()

            @QtCore.Slot()
            def slider_1_right_button_clicked():
                self.dr_result_selection_slider_1.setValue(
                    self.dr_result_selection_slider_1.value() + 1
                )
                # update the text
                update_slider_1_text()

            @QtCore.Slot()
            def slider_2_left_button_clicked():
                self.dr_result_selection_slider_2.setValue(
                    self.dr_result_selection_slider_2.value() - 1
                )
                # update the text
                update_slider_2_text()

            @QtCore.Slot()
            def slider_2_right_button_clicked():
                self.dr_result_selection_slider_2.setValue(
                    self.dr_result_selection_slider_2.value() + 1
                )
                # update the text
                update_slider_2_text()

            # initialize sliders if first time clicked
            if not self.dr_result_sliders_existed:

                # sliders that rank the dimension reduction result and can select one of them
                # slider 1
                self.slider_1_layout = QtWidgets.QGridLayout()
                self.slider_1_layout.setVerticalSpacing(0)
                self.dr_result_selection_slider_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                self.dr_result_selection_slider_1.setFixedSize(self.plot_size, 50)
                self.dr_result_selection_slider_1.setMinimum(0)
                self.dr_result_selection_slider_1.setMaximum(len(self.all_high_dim_points_1) - 1)
                self.dr_result_selection_slider_1.setValue(0)
                self.dr_result_selection_slider_1.setTickPosition(QtWidgets.QSlider.NoTicks)
                self.dr_result_selection_slider_1.setTickInterval(1)
                self.dr_result_selection_slider_1.valueChanged.connect(
                    dr_result_selection_slider_1_changed
                )
                self.slider_1_layout.addWidget(self.dr_result_selection_slider_1, 0, 0, 1, 3)
                # left and right buttons to move the slider around, with number in the middle
                # left button
                self.slider_1_left_button = QtWidgets.QToolButton()
                self.slider_1_left_button.setArrowType(QtCore.Qt.LeftArrow)
                self.slider_1_left_button.clicked.connect(slider_1_left_button_clicked)
                self.slider_1_left_button.setFixedSize(30, 30)
                self.slider_1_left_button.setStyleSheet('color: black')
                self.slider_1_layout.addWidget(self.slider_1_left_button, 1, 0, 1, 1)

                # middle text
                slider_1_text_pixmap = QPixmap(300, 50)
                slider_1_text_pixmap.fill(QtCore.Qt.white)
                painter = QtGui.QPainter(slider_1_text_pixmap)
                painter.drawText(
                    0,
                    0,
                    300,
                    50,
                    QtGui.Qt.AlignCenter,
                    f'{self.dr_result_selection_slider_1.value()+1}/{len(self.cur_class_indices)}',
                )
                painter.setFont(QFont('Helvetica', 30))
                painter.end()
                # create label to contain the texts
                self.slider_1_text_label = QLabel(self)
                self.slider_1_text_label.setContentsMargins(0, 0, 0, 0)
                self.slider_1_text_label.setFixedSize(QtCore.QSize(150, 50))
                self.slider_1_text_label.setAlignment(QtCore.Qt.AlignCenter)
                self.slider_1_text_label.setPixmap(slider_1_text_pixmap)
                self.slider_1_layout.addWidget(self.slider_1_text_label, 1, 1, 1, 1)

                # right button
                self.slider_1_right_button = QtWidgets.QToolButton()
                self.slider_1_right_button.setArrowType(QtCore.Qt.RightArrow)
                self.slider_1_right_button.setFixedSize(30, 30)
                self.slider_1_right_button.setStyleSheet('color: black')
                self.slider_1_right_button.clicked.connect(slider_1_right_button_clicked)
                self.slider_1_layout.addWidget(self.slider_1_right_button, 1, 2, 1, 1)

                # add slider 1 layout to the general layout
                if self.demo:
                    if self.mode == 'digit_recognition':
                        self.demo_layout.addLayout(self.slider_1_layout, 4, 1, 1, 1)
                    elif self.mode == 'object_detection':
                        self.demo_layout.addLayout(self.slider_1_layout, 4, 1, 1, 1)
                        self.slider_1_layout.setContentsMargins(40, 0, 0, 0)
                    elif self.mode == 'piv':
                        self.demo_layout.addLayout(self.slider_1_layout, 4, 1, 1, 1)
                else:
                    self.aggregate_result_layout.addLayout(self.slider_1_layout, 3, 1)

                # slider 2
                self.slider_2_layout = QtWidgets.QGridLayout()
                self.slider_2_layout.setVerticalSpacing(0)
                self.dr_result_selection_slider_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                self.dr_result_selection_slider_2.setFixedSize(self.plot_size, 50)
                self.dr_result_selection_slider_2.setMinimum(0)
                self.dr_result_selection_slider_2.setMaximum(len(self.all_high_dim_points_2) - 1)
                self.dr_result_selection_slider_2.setValue(0)
                self.dr_result_selection_slider_2.setTickPosition(QtWidgets.QSlider.NoTicks)
                self.dr_result_selection_slider_2.setTickInterval(1)
                self.dr_result_selection_slider_2.valueChanged.connect(
                    dr_result_selection_slider_2_changed
                )
                self.slider_2_layout.addWidget(self.dr_result_selection_slider_2, 0, 0, 1, 3)
                # left and right buttons to move the slider around, with number in the middle
                # left button
                self.slider_2_left_button = QtWidgets.QToolButton()
                self.slider_2_left_button.setArrowType(QtCore.Qt.LeftArrow)
                self.slider_2_left_button.setFixedSize(30, 30)
                self.slider_2_left_button.setStyleSheet('color: black')
                self.slider_2_left_button.clicked.connect(slider_2_left_button_clicked)
                self.slider_2_layout.addWidget(self.slider_2_left_button, 1, 0, 1, 1)

                # middle text
                slider_2_text_pixmap = QPixmap(150, 30)
                slider_2_text_pixmap.fill(QtCore.Qt.white)
                painter = QtGui.QPainter(slider_2_text_pixmap)
                painter.setFont(QFont('Helvetica', 12))
                painter.drawText(
                    0,
                    0,
                    150,
                    30,
                    QtGui.Qt.AlignCenter,
                    f'{self.dr_result_selection_slider_2.value()+1}/{len(self.cur_class_indices)}',
                )
                painter.end()
                # create label to contain the texts
                self.slider_2_text_label = QLabel(self)
                self.slider_2_text_label.setContentsMargins(0, 0, 0, 0)
                self.slider_2_text_label.setFixedSize(QtCore.QSize(150, 50))
                self.slider_2_text_label.setAlignment(QtCore.Qt.AlignCenter)
                self.slider_2_text_label.setPixmap(slider_2_text_pixmap)
                self.slider_2_layout.addWidget(self.slider_2_text_label, 1, 1, 1, 1)

                # right button
                self.slider_2_right_button = QtWidgets.QToolButton()
                self.slider_2_right_button.setArrowType(QtCore.Qt.RightArrow)
                self.slider_2_right_button.setFixedSize(30, 30)
                self.slider_2_right_button.setStyleSheet('color: black')
                self.slider_2_right_button.clicked.connect(slider_2_right_button_clicked)
                self.slider_2_layout.addWidget(self.slider_2_right_button, 1, 2, 1, 1)

                # add slider 2 layout to the general layout
                if self.demo:
                    if self.mode == 'digit_recognition':
                        self.demo_layout.addLayout(self.slider_2_layout, 6, 1, 1, 1)
                    elif self.mode == 'object_detection':
                        self.demo_layout.addLayout(self.slider_2_layout, 6, 1, 1, 1)
                        self.slider_2_layout.setContentsMargins(40, 0, 0, 0)
                    elif self.mode == 'piv':
                        self.demo_layout.addLayout(self.slider_2_layout, 6, 1, 1, 1)
                else:
                    self.aggregate_result_layout.addLayout(self.slider_2_layout, 3, 2)

                self.dr_result_sliders_existed = True

            # get the clicked scatter item's information
            # when item is not none, it is from real click
            if item != None:
                self.image_index = int(item.opts['name'])
                print(f'clicked image index {self.image_index}')
            # when the input is empty, it is called automatically
            else:
                # image index should be defined
                if self.image_index == None:
                    raise Exception(
                        'image_index should be defined prior to calling run_dimension_reduction'
                    )

            # get the ranking in each colorbar and change its value while locking both sliders
            # slider 1
            self.slider_1_locked = True
            self.slider_2_locked = True
            self.slider_1_selected_index = self.sorted_class_indices_1.index(self.image_index)
            self.dr_result_selection_slider_1.setValue(self.slider_1_selected_index)
            # update the text
            update_slider_1_text()
            # slider 2
            self.slider_2_selected_index = self.sorted_class_indices_2.index(self.image_index)
            self.dr_result_selection_slider_2.setValue(self.slider_2_selected_index)
            # update the text
            update_slider_2_text()
            # update the indicator of current selected item
            display_dimension_reduction(compute_dr=False)
            # unlock after changing the values
            self.slider_1_locked = False
            self.slider_2_locked = False

            # get the corresponding image path
            if self.mode == 'digit_recognition' or self.mode == 'object_detection':
                self.image_path = self.all_images_paths[self.image_index]
                print(f'Selected image at {self.image_path}')
            elif self.mode == 'piv':
                # single case images paths
                self.image_1_path = self.all_images_1_paths[self.image_index]
                self.image_2_path = self.all_images_2_paths[self.image_index]
                print(f'Selected image 1 at {self.image_1_path}')
                print(f'Selected image 2 at {self.image_2_path}')

                # single case model outputs
                self.all_quantities_1 = self.aggregate_outputs_1[:, self.image_index]
                self.all_quantities_2 = self.aggregate_outputs_2[:, self.image_index]
                self.all_ground_truths = self.aggregate_ground_truths[:, self.image_index]

            # load the image
            self.load_single_image()

            # display individual view
            if self.mode == 'digit_recognition':
                # convert to QImage for display purpose
                self.cur_display_image = nero_utilities.tensor_to_qt_image(
                    self.cur_image_pt, self.display_image_size, revert_color=True
                )
                # prepare image tensor for model purpose
                self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)

            elif self.mode == 'object_detection':
                # convert to QImage for display purpose
                self.cur_display_image = nero_utilities.tensor_to_qt_image(
                    self.cur_image_pt, self.display_image_size
                )

            elif self.mode == 'piv':
                # create new GIF
                display_image_1_pil = Image.fromarray(self.cur_image_1_pt.numpy(), 'RGB')
                display_image_2_pil = Image.fromarray(self.cur_image_2_pt.numpy(), 'RGB')
                other_images_pil = [
                    display_image_1_pil,
                    display_image_2_pil,
                    display_image_2_pil,
                    self.blank_image_pil,
                ]
                self.gif_path = os.path.join(
                    self.cache_dir, self.mode, self.loaded_image_1_name.split('.')[0] + '.gif'
                )
                display_image_1_pil.save(
                    fp=self.gif_path,
                    format='GIF',
                    append_images=other_images_pil,
                    save_all=True,
                    duration=300,
                    loop=0,
                )

            # run model all and display results (Individual NERO plot)
            self.run_model_single()

        # when hovered on the scatter plot item
        def low_dim_scatter_hovered(item, points):
            item.setToolTip(item.opts['hover_text'])

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

        # plot all the scatter items with brush color reflecting the intensity
        def plot_dr_scatter(
            low_dim_scatter_plot,
            low_dim,
            sorted_intensity,
            sorted_class_indices,
            slider_selected_index,
        ):
            # same colorbar as used in aggregate NERO plot, to be used in color encode scatter points
            # if self.mode == 'digit_recognition':
            # digit recognition does not have color defined elsewhere like others since it never uses heatmaps
            self.color_map = pg.colormap.get('viridis')
            if self.mode == 'object_detection':
                self.cm_range = [0, 1.0]
                scatter_lut = self.color_map.getLookupTable(
                    start=self.cm_range[0], stop=self.cm_range[1], nPts=500, alpha=False
                )
            elif self.mode == 'piv':
                self.cm_range = (self.loss_high_bound, self.loss_low_bound)
                scatter_lut = self.color_map.getLookupTable(
                    start=self.cm_range[1], stop=self.cm_range[0], nPts=500, alpha=False
                )

            # digit recognition has color bar only for scatter plot
            if self.demo:
                self.color_bar = pg.ColorBarItem(
                    values=self.cm_range,
                    colorMap=self.color_map,
                    interactive=False,
                    orientation='horizontal',
                    width=30,
                )
                # add colorbar to a specific place if in demo mode
                dummy_view = pg.GraphicsLayoutWidget()
                dummy_plot = pg.PlotItem()
                if self.mode == 'digit_recognition':
                    dummy_plot.layout.setContentsMargins(0, 50, 10, 0)
                else:
                    dummy_plot.layout.setContentsMargins(0, 0, 0, 0)
                dummy_plot.setFixedHeight(0)
                dummy_plot.setFixedWidth(self.plot_size * 1.2)
                dummy_plot.hideAxis('bottom')
                dummy_plot.hideAxis('left')
                dummy_view.addItem(dummy_plot)
                dummy_image = pg.ImageItem()
                self.color_bar.setImageItem(dummy_image, insert_in=dummy_plot)
                if self.mode == 'digit_recognition':
                    self.scatterplot_sorting_layout.addWidget(dummy_view, 3, 0, 1, 2)
                elif self.mode == 'object_detection':
                    self.scatterplot_sorting_layout.addWidget(dummy_view, 3, 0, 1, 2)
                    dummy_plot.layout.setContentsMargins(50, 0, 0, 0)
                elif self.mode == 'piv':
                    self.scatterplot_sorting_layout.addWidget(dummy_view, 3, 0, 1, 2)
                    dummy_plot.layout.setContentsMargins(50, 0, 0, 0)

            # quantize all the intensity into color
            color_indices = []
            for i in range(len(sorted_intensity)):
                lut_index = nero_utilities.lerp(
                    sorted_intensity[i], self.cm_range[0], self.cm_range[1], 0, 499
                )
                if lut_index > 499:
                    lut_index = 499
                elif lut_index < 0:
                    lut_index = 0

                color_indices.append(scatter_lut[int(lut_index)])

            # image index position in the current sorted class indices
            if self.image_index != None:
                sorted_selected_index = sorted_class_indices.index(self.image_index)
            else:
                sorted_selected_index = len(sorted_class_indices) - 1

            for i, index in enumerate(sorted_class_indices):
                # add the selected item's color at last to make sure that the current selected item is always on top (rendered last)
                if i == sorted_selected_index:
                    continue
                # add individual items for getting the item's name later when clicking
                # Set pxMode=True to have scatter items stay at the same screen size
                low_dim_scatter_item = pg.ScatterPlotItem(pxMode=True, hoverable=True)
                low_dim_scatter_item.opts[
                    'hover_text'
                ] = f'{self.intensity_method}: {round(sorted_intensity[i], 3)}'
                low_dim_scatter_item.setSymbol('o')
                low_dim_point = [
                    {
                        'pos': (low_dim[i, 0], low_dim[i, 1]),
                        'size': self.scatter_item_size,
                        'pen': QtGui.QColor(
                            color_indices[i][0], color_indices[i][1], color_indices[i][2]
                        ),
                        'brush': QtGui.QColor(
                            color_indices[i][0], color_indices[i][1], color_indices[i][2]
                        ),
                    }
                ]

                # add points to the item, the name are its original index within the ENTIRE dataset
                low_dim_scatter_item.setData(low_dim_point, name=str(index))
                # connect click events on scatter items
                low_dim_scatter_item.sigClicked.connect(low_dim_scatter_clicked)
                low_dim_scatter_item.sigHovered.connect(low_dim_scatter_hovered)
                # add points to the plot
                low_dim_scatter_plot.addItem(low_dim_scatter_item)

            # add the current selected one
            low_dim_scatter_item = pg.ScatterPlotItem(pxMode=True, hoverable=True)
            low_dim_scatter_item.opts[
                'hover_text'
            ] = f'{self.intensity_method}: {round(sorted_intensity[sorted_selected_index], 3)}'
            low_dim_scatter_item.setSymbol('o')
            # set red pen indicator if slider selects
            if slider_selected_index != None:
                # smaller circles in accounting for the red ring
                low_dim_point = [
                    {
                        'pos': (
                            low_dim[sorted_selected_index, 0],
                            low_dim[sorted_selected_index, 1],
                        ),
                        'size': self.scatter_item_size - 2.0001,
                        'pen': {'color': 'red', 'width': 2},
                        'brush': QtGui.QColor(
                            color_indices[sorted_selected_index][0],
                            color_indices[sorted_selected_index][1],
                            color_indices[sorted_selected_index][2],
                        ),
                    }
                ]
            else:
                low_dim_point = [
                    {
                        'pos': (
                            low_dim[sorted_selected_index, 0],
                            low_dim[sorted_selected_index, 1],
                        ),
                        'size': self.scatter_item_size,
                        'pen': QtGui.QColor(
                            color_indices[sorted_selected_index][0],
                            color_indices[sorted_selected_index][1],
                            color_indices[sorted_selected_index][2],
                        ),
                        'brush': QtGui.QColor(
                            color_indices[sorted_selected_index][0],
                            color_indices[sorted_selected_index][1],
                            color_indices[sorted_selected_index][2],
                        ),
                    }
                ]
            # add points to the item
            low_dim_scatter_item.setData(
                low_dim_point, name=str(sorted_class_indices[sorted_selected_index])
            )
            # connect click events on scatter items
            low_dim_scatter_item.sigClicked.connect(low_dim_scatter_clicked)
            low_dim_scatter_item.sigHovered.connect(low_dim_scatter_hovered)
            # add points to the plot
            low_dim_scatter_plot.addItem(low_dim_scatter_item)

        # helper function on displaying the 2D scatter plot
        # when compute_dr is true, dimension reduction is computed
        def display_dimension_reduction(compute_dr=True):

            # initialize all the views when the first time
            if not self.dr_result_existed:
                # scatter plot on low-dim points
                self.low_dim_scatter_view_1 = pg.GraphicsLayoutWidget()
                self.low_dim_scatter_view_1.setBackground('white')
                if self.mode == 'digit_recognition':
                    self.low_dim_scatter_view_1.setFixedSize(
                        self.plot_size * 1.1, self.plot_size * 1.1
                    )
                    self.low_dim_scatter_view_1.ci.setContentsMargins(20, 100, 0, 0)
                elif self.mode == 'object_detection':
                    self.low_dim_scatter_view_1.setFixedSize(
                        self.plot_size * 1.3, self.plot_size * 1.3
                    )
                    self.low_dim_scatter_view_1.ci.setContentsMargins(20, 0, 0, 0)
                elif self.mode == 'piv':
                    self.low_dim_scatter_view_1.setFixedSize(
                        self.plot_size * 1.3, self.plot_size * 1.3
                    )
                    self.low_dim_scatter_view_1.ci.setContentsMargins(20, 0, 0, 0)

                # add plot
                self.low_dim_scatter_plot_1 = self.low_dim_scatter_view_1.addPlot()
                if self.mode == 'object_detection':
                    self.low_dim_scatter_plot_1.setContentsMargins(0, 0, 0, 150)
                self.low_dim_scatter_plot_1.hideAxis('left')
                self.low_dim_scatter_plot_1.hideAxis('bottom')

                # set axis range
                self.low_dim_scatter_plot_1.setXRange(-1.2, 1.2, padding=0)
                self.low_dim_scatter_plot_1.setYRange(-1.2, 1.2, padding=0)
                # Not letting user zoom out past axis limit
                self.low_dim_scatter_plot_1.vb.setLimits(xMin=-1.2, xMax=1.2, yMin=-1.2, yMax=1.2)
                # No auto range when adding new item (red indicator)
                self.low_dim_scatter_plot_1.vb.disableAutoRange(axis=pg.ViewBox.XYAxes)

                self.low_dim_scatter_view_2 = pg.GraphicsLayoutWidget()
                self.low_dim_scatter_view_2.setBackground('white')
                if self.mode == 'digit_recognition':
                    self.low_dim_scatter_view_2.setFixedSize(
                        self.plot_size * 1.1, self.plot_size * 1.1
                    )
                    self.low_dim_scatter_view_2.ci.setContentsMargins(20, 0, 0, 0)
                elif self.mode == 'object_detection':
                    self.low_dim_scatter_view_2.setFixedSize(
                        self.plot_size * 1.25, self.plot_size * 1.25
                    )
                    self.low_dim_scatter_view_2.ci.setContentsMargins(20, 0, 0, 0)
                elif self.mode == 'piv':
                    self.low_dim_scatter_view_2.setFixedSize(
                        self.plot_size * 1.25, self.plot_size * 1.25
                    )
                    self.low_dim_scatter_view_2.ci.setContentsMargins(20, 0, 0, 100)

                # add plot
                self.low_dim_scatter_plot_2 = self.low_dim_scatter_view_2.addPlot()
                self.low_dim_scatter_plot_2.hideAxis('left')
                self.low_dim_scatter_plot_2.hideAxis('bottom')

                # set axis range
                self.low_dim_scatter_plot_2.setXRange(-1.2, 1.2, padding=0)
                self.low_dim_scatter_plot_2.setYRange(-1.2, 1.2, padding=0)
                # Not letting user zoom out past axis limit
                self.low_dim_scatter_plot_2.vb.setLimits(xMin=-1.2, xMax=1.2, yMin=-1.2, yMax=1.2)
                # scatter item size
                self.scatter_item_size = 12

                self.dr_result_existed = True

            # run dimension reduction algorithm
            if compute_dr:
                # try to load from cache
                low_dim_1_name = f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_name}_{self.class_selection}_{self.quantity_name}_{self.dr_selection}'
                self.low_dim_1 = self.load_from_cache(low_dim_1_name)

                # when no cache available
                if not self.load_successfully:
                    self.low_dim_1 = dimension_reduce(self.all_high_dim_points_1, target_dim=2)
                    self.low_dim_1 = normalize_low_dim_result(self.low_dim_1)
                    self.save_to_cache(low_dim_1_name, self.low_dim_1)

                # try to load from cache
                low_dim_2_name = f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_name}_{self.class_selection}_{self.quantity_name}_{self.dr_selection}'
                self.low_dim_2 = self.load_from_cache(low_dim_2_name)

                # when no cache available
                if not self.load_successfully:
                    self.low_dim_2 = dimension_reduce(self.all_high_dim_points_2, target_dim=2)
                    self.low_dim_2 = normalize_low_dim_result(self.low_dim_2)
                    self.save_to_cache(low_dim_2_name, self.low_dim_2)

                # plot both scatter plots
                # rank the intensity values (small to large)
                self.sorted_intensity_indices_1 = np.argsort(self.all_intensity_1)
                self.sorted_intensity_1 = sorted(self.all_intensity_1)
                self.sorted_class_indices_1 = [
                    self.cur_class_indices[idx] for idx in self.sorted_intensity_indices_1
                ]
                self.sorted_intensity_indices_2 = np.argsort(self.all_intensity_2)
                self.sorted_intensity_2 = sorted(self.all_intensity_2)
                self.sorted_class_indices_2 = [
                    self.cur_class_indices[idx] for idx in self.sorted_intensity_indices_2
                ]

                # sort the low dim points accordingly
                self.low_dim_1 = self.low_dim_1[self.sorted_intensity_indices_1]
                self.low_dim_2 = self.low_dim_2[self.sorted_intensity_indices_2]

            # plot the dimension reduction scatter plot
            plot_dr_scatter(
                self.low_dim_scatter_plot_1,
                self.low_dim_1,
                self.sorted_intensity_1,
                self.sorted_class_indices_1,
                self.slider_1_selected_index,
            )

            plot_dr_scatter(
                self.low_dim_scatter_plot_2,
                self.low_dim_2,
                self.sorted_intensity_2,
                self.sorted_class_indices_2,
                self.slider_2_selected_index,
            )

            if self.mode == 'digit_recognition':
                if self.demo:
                    self.demo_layout.addWidget(self.low_dim_scatter_view_1, 2, 1, 2, 1)
                    self.demo_layout.addWidget(self.low_dim_scatter_view_2, 4, 1, 2, 1)
                else:
                    self.aggregate_result_layout.addWidget(self.low_dim_scatter_view_1, 1, 3)
                    self.aggregate_result_layout.addWidget(self.low_dim_scatter_view_2, 2, 3)
            elif self.mode == 'object_detection':
                if self.demo:
                    self.demo_layout.addWidget(self.low_dim_scatter_view_1, 2, 1, 3, 1)
                    self.demo_layout.addWidget(self.low_dim_scatter_view_2, 4, 1, 3, 1)
                else:
                    self.aggregate_result_layout.addWidget(self.low_dim_scatter_view_1, 2, 1)
                    self.aggregate_result_layout.addWidget(self.low_dim_scatter_view_2, 2, 2)
            # arguebly the layout for PIV is the same as object detection, but separated them for future expandibility
            elif self.mode == 'piv':
                if self.demo:
                    self.demo_layout.addWidget(self.low_dim_scatter_view_1, 2, 1, 3, 1)
                    self.demo_layout.addWidget(self.low_dim_scatter_view_2, 4, 1, 3, 1)
                else:
                    self.aggregate_result_layout.addWidget(self.low_dim_scatter_view_1, 2, 1)
                    self.aggregate_result_layout.addWidget(self.low_dim_scatter_view_2, 2, 2)

        # radio buttons on choosing quantity used to compute intensity
        @QtCore.Slot()
        def mean_intensity_button_clicked():
            self.intensity_method = 'mean'
            self.all_intensity_1 = np.mean(self.all_high_dim_points_1, axis=1)
            self.all_intensity_2 = np.mean(self.all_high_dim_points_2, axis=1)

            # re-display the scatter plot
            display_dimension_reduction(compute_dr=False)

        @QtCore.Slot()
        def variance_intensity_button_clicked():
            self.intensity_method = 'variance'
            self.all_intensity_1 = np.var(self.all_high_dim_points_1, axis=1)
            self.all_intensity_2 = np.var(self.all_high_dim_points_2, axis=1)

            # normalize to colormap range
            intensity_min = min(np.min(self.all_intensity_1), np.min(self.all_intensity_2))
            intensity_max = max(np.max(self.all_intensity_1), np.max(self.all_intensity_2))
            self.all_intensity_1 = nero_utilities.lerp(
                self.all_intensity_1,
                intensity_min,
                intensity_max,
                self.cm_range[0],
                self.cm_range[1],
            )
            self.all_intensity_2 = nero_utilities.lerp(
                self.all_intensity_2,
                intensity_min,
                intensity_max,
                self.cm_range[0],
                self.cm_range[1],
            )

            # re-display the scatter plot
            display_dimension_reduction(compute_dr=False)

        @QtCore.Slot()
        def dr_result_selection_slider_1_changed():

            # when the slider bar is changed directly by user, it is unlocked
            # mimics that a point has been clicked
            if not self.slider_1_locked:
                # change the ranking in the other colorbar
                self.slider_1_selected_index = self.dr_result_selection_slider_1.value()

                # get the clicked scatter item's information
                self.image_index = self.sorted_class_indices_1[self.slider_1_selected_index]
                print(
                    f'slider 1 image index {self.image_index}, ranked position {self.slider_1_selected_index}'
                )
                # update the text
                update_slider_1_text()

                # change the other slider's value
                self.slider_2_locked = True
                self.slider_2_selected_index = self.sorted_class_indices_2.index(self.image_index)
                self.dr_result_selection_slider_2.setValue(self.slider_2_selected_index)
                # update the text
                update_slider_2_text()
                self.slider_2_locked = False

                # update the scatter plot without re-computing dimension reduction algorithm
                display_dimension_reduction(compute_dr=False)

                # get the corresponding image path
                if self.mode == 'digit_recognition' or self.mode == 'object_detection':
                    self.image_path = self.all_images_paths[self.image_index]
                    print(f'Selected image at {self.image_path}')
                elif self.mode == 'piv':
                    # single case images paths
                    self.image_1_path = self.all_images_1_paths[self.image_index]
                    self.image_2_path = self.all_images_2_paths[self.image_index]
                    print(f'Selected image 1 at {self.image_1_path}')
                    print(f'Selected image 2 at {self.image_2_path}')

                    # single case model outputs
                    self.all_quantities_1 = self.aggregate_outputs_1[:, self.image_index]
                    self.all_quantities_2 = self.aggregate_outputs_2[:, self.image_index]
                    self.all_ground_truths = self.aggregate_ground_truths[:, self.image_index]

                # load the image
                self.load_single_image()

                # display individual view
                if self.mode == 'digit_recognition':
                    # convert to QImage for display purpose
                    self.cur_display_image = nero_utilities.tensor_to_qt_image(
                        self.cur_image_pt, self.display_image_size, revert_color=True
                    )
                    # prepare image tensor for model purpose
                    self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)

                elif self.mode == 'object_detection':
                    # convert to QImage for display purpose
                    self.cur_display_image = nero_utilities.tensor_to_qt_image(
                        self.cur_image_pt, self.display_image_size
                    )

                elif self.mode == 'piv':
                    # create new GIF
                    display_image_1_pil = Image.fromarray(self.cur_image_1_pt.numpy(), 'RGB')
                    display_image_2_pil = Image.fromarray(self.cur_image_2_pt.numpy(), 'RGB')
                    other_images_pil = [
                        display_image_1_pil,
                        display_image_2_pil,
                        display_image_2_pil,
                        self.blank_image_pil,
                    ]
                    self.gif_path = os.path.join(
                        self.cache_dir, self.loaded_image_1_name.split('.')[0] + '.gif'
                    )
                    display_image_1_pil.save(
                        fp=self.gif_path,
                        format='GIF',
                        append_images=other_images_pil,
                        save_all=True,
                        duration=300,
                        loop=0,
                    )

                # run model all and display results (Individual NERO plot and detailed plot)
                self.run_model_single()

        @QtCore.Slot()
        def dr_result_selection_slider_2_changed():

            # when the slider bar is changed directly by user, it is unlocked
            # mimics that a point has been clicked
            if not self.slider_2_locked:
                # change the ranking in the other colorbar
                self.slider_2_selected_index = self.dr_result_selection_slider_2.value()

                # get the clicked scatter item's information
                self.image_index = self.sorted_class_indices_2[self.slider_2_selected_index]
                print(
                    f'slider 2 image index {self.image_index}, ranked position {self.slider_2_selected_index}'
                )
                # update the text
                update_slider_2_text()

                # change the other slider's value
                self.slider_1_locked = True
                self.slider_1_selected_index = self.sorted_class_indices_1.index(self.image_index)
                self.dr_result_selection_slider_1.setValue(self.slider_1_selected_index)
                # update the text
                update_slider_1_text()
                self.slider_1_locked = False

                # update the scatter plot
                display_dimension_reduction(compute_dr=False)

                # get the corresponding image path
                if self.mode == 'digit_recognition' or self.mode == 'object_detection':
                    self.image_path = self.all_images_paths[self.image_index]
                    print(f'Selected image at {self.image_path}')
                elif self.mode == 'piv':
                    # single case images paths
                    self.image_1_path = self.all_images_1_paths[self.image_index]
                    self.image_2_path = self.all_images_2_paths[self.image_index]
                    print(f'Selected image 1 at {self.image_1_path}')
                    print(f'Selected image 2 at {self.image_2_path}')

                    # single case model outputs
                    self.all_quantities_1 = self.aggregate_outputs_1[:, self.image_index]
                    self.all_quantities_2 = self.aggregate_outputs_2[:, self.image_index]
                    self.all_ground_truths = self.aggregate_ground_truths[:, self.image_index]

                # load the image
                self.load_single_image()

                # display individual view
                if self.mode == 'digit_recognition':
                    # convert to QImage for display purpose
                    self.cur_display_image = nero_utilities.tensor_to_qt_image(
                        self.cur_image_pt, self.display_image_size, revert_color=True
                    )
                    # prepare image tensor for model purpose
                    self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)

                elif self.mode == 'object_detection':
                    # convert to QImage for display purpose
                    self.cur_display_image = nero_utilities.tensor_to_qt_image(
                        self.cur_image_pt, self.display_image_size
                    )

                elif self.mode == 'piv':
                    # create new GIF
                    display_image_1_pil = Image.fromarray(self.cur_image_1_pt.numpy(), 'RGB')
                    display_image_2_pil = Image.fromarray(self.cur_image_2_pt.numpy(), 'RGB')
                    other_images_pil = [
                        display_image_1_pil,
                        display_image_2_pil,
                        display_image_2_pil,
                        self.blank_image_pil,
                    ]
                    self.gif_path = os.path.join(
                        self.cache_dir, self.loaded_image_1_name.split('.')[0] + '.gif'
                    )
                    display_image_1_pil.save(
                        fp=self.gif_path,
                        format='GIF',
                        append_images=other_images_pil,
                        save_all=True,
                        duration=300,
                        loop=0,
                    )

                # run model all and display results (Individual NERO plot)
                self.run_model_single()

        # run dimension reduction of all images on the selected digit
        # each image has tensor with length being the number of translations
        self.cur_class_indices = []
        if self.class_selection == 'all':
            # all the indices
            self.cur_class_indices = list(range(len(self.loaded_images_labels)))
        else:
            for i in range(len(self.loaded_images_labels)):
                if self.class_selection == self.loaded_images_labels[i]:
                    self.cur_class_indices.append(i)

        if self.mode == 'digit_recognition':
            num_transformations = len(self.all_aggregate_angles)
        elif self.mode == 'object_detection':
            num_transformations = len(self.x_translation) * len(self.y_translation)
        elif self.mode == 'piv':
            num_transformations = 16

        self.all_high_dim_points_1 = np.zeros((len(self.cur_class_indices), num_transformations))
        self.all_high_dim_points_2 = np.zeros((len(self.cur_class_indices), num_transformations))

        for i, index in enumerate(self.cur_class_indices):
            # go through all the transfomations
            for j in range(num_transformations):
                if self.mode == 'digit_recognition':

                    self.all_high_dim_points_1[i, j] = int(
                        self.all_outputs_1[j][index].argmax() == self.loaded_images_labels[index]
                    )
                    self.all_high_dim_points_2[i, j] = int(
                        self.all_outputs_2[j][index].argmax() == self.loaded_images_labels[index]
                    )

                elif self.mode == 'object_detection':

                    y = int(j // len(self.x_translation))
                    x = int(j % len(self.x_translation))

                    # aggregate_outputs_1 has shape (num_y_translations, num_x_translations, num_samples, 8)
                    if self.use_consensus:
                        cur_conf_1 = self.aggregate_consensus_outputs_1[y, x][index][0, 4]
                        cur_iou_1 = self.aggregate_consensus_outputs_1[y, x][index][0, 6]
                        cur_correctness_1 = self.aggregate_consensus_outputs_1[y, x][index][0, 7]

                        cur_conf_2 = self.aggregate_consensus_outputs_2[y, x][index][0, 4]
                        cur_iou_2 = self.aggregate_consensus_outputs_2[y, x][index][0, 6]
                        cur_correctness_2 = self.aggregate_consensus_outputs_2[y, x][index][0, 7]
                    else:
                        cur_conf_1 = self.aggregate_outputs_1[y, x][index][0, 4]
                        cur_iou_1 = self.aggregate_outputs_1[y, x][index][0, 6]
                        cur_correctness_1 = self.aggregate_outputs_1[y, x][index][0, 7]

                        cur_conf_2 = self.aggregate_outputs_2[y, x][index][0, 4]
                        cur_iou_2 = self.aggregate_outputs_2[y, x][index][0, 6]
                        cur_correctness_2 = self.aggregate_outputs_2[y, x][index][0, 7]

                        if j >= 490 and j <= 510:
                            print(cur_conf_1, cur_iou_1, cur_correctness_1)
                            print(cur_conf_2, cur_iou_2, cur_correctness_2)

                    # always have the correctness involved
                    if self.quantity_name == 'Conf*IOU':
                        cur_value_1 = cur_conf_1 * cur_iou_1 * cur_correctness_1
                        cur_value_2 = cur_conf_2 * cur_iou_2 * cur_correctness_2
                    elif self.quantity_name == 'Confidence':
                        cur_value_1 = cur_conf_1
                        cur_value_2 = cur_conf_2
                    elif self.quantity_name == 'IOU':
                        cur_value_1 = cur_iou_1 * cur_correctness_1
                        cur_value_2 = cur_iou_2 * cur_correctness_2

                    # below values exist in non-demo mode
                    elif self.quantity_name == 'Precision':
                        if self.use_consensus:
                            cur_value_1 = self.aggregate_consensus_precision_1[y, x][index]
                            cur_value_2 = self.aggregate_consensus_precision_2[y, x][index]
                        else:
                            cur_value_1 = self.aggregate_precision_1[y, x][index]
                            cur_value_2 = self.aggregate_precision_2[y, x][index]
                    elif self.quantity_name == 'Recall':
                        if self.use_consensus:
                            cur_value_1 = self.aggregate_consensus_recall_1[y, x][index]
                            cur_value_2 = self.aggregate_consensus_recall_2[y, x][index]
                        else:
                            cur_value_1 = self.aggregate_recall_1[y, x][index]
                            cur_value_2 = self.aggregate_recall_2[y, x][index]
                    elif self.quantity_name == 'F1 Score':
                        if self.use_consensus:
                            cur_value_1 = self.aggregate_consensus_F_measure_1[y, x][index]
                            cur_value_2 = self.aggregate_consensus_F_measure_2[y, x][index]
                        else:
                            cur_value_1 = self.aggregate_F_measure_1[y, x][index]
                            cur_value_2 = self.aggregate_F_measure_2[y, x][index]
                    elif self.quantity_name == 'mAP':
                        cur_value_1 = 0
                        cur_value_2 = 0

                    self.all_high_dim_points_1[i, j] = cur_value_1
                    self.all_high_dim_points_2[i, j] = cur_value_2

                elif self.mode == 'piv':
                    self.all_high_dim_points_1[i, j] = self.loss_module(
                        self.aggregate_outputs_1[j, i],
                        self.aggregate_ground_truths[j, i],
                        reduction='mean',
                    )
                    self.all_high_dim_points_2[i, j] = self.loss_module(
                        self.aggregate_outputs_2[j, i],
                        self.aggregate_ground_truths[j, i],
                        reduction='mean',
                    )

        # radio buittons on choosing the intensity quantity
        if not self.dr_result_existed:
            # Title on the two radio buttons
            intensity_button_pixmap = QPixmap(300, 60)
            intensity_button_pixmap.fill(QtCore.Qt.white)
            painter = QtGui.QPainter(intensity_button_pixmap)
            if self.mode == 'digit_recognition' or self.mode == 'object_detection':
                painter.setFont(QFont('Helvetica', 30))
                painter.drawText(0, 0, 300, 60, QtGui.Qt.AlignLeft, 'DR Plot Sorting:')
            elif self.mode == 'piv':
                painter.setFont(QFont('Helvetica', 30))
                painter.drawText(0, 0, 300, 60, QtGui.Qt.AlignLeft, 'DR Sorting:')
            painter.end()

            # create label to contain the texts
            intensity_button_label = QLabel(self)
            intensity_button_label.setContentsMargins(0, 0, 0, 0)
            intensity_button_label.setFixedSize(QtCore.QSize(350, 115))
            intensity_button_label.setAlignment(QtCore.Qt.AlignLeft)
            intensity_button_label.setWordWrap(True)
            intensity_button_label.setTextFormat(QtGui.Qt.AutoText)
            intensity_button_label.setPixmap(intensity_button_pixmap)
            intensity_button_label.setContentsMargins(5, 60, 0, 0)
            # add to the layout
            if self.demo:
                self.scatterplot_sorting_layout = QtWidgets.QGridLayout()
                # the title occupies two rows because we have two selections (mean and variance)
                self.scatterplot_sorting_layout.addWidget(intensity_button_label, 0, 0, 2, 1)
            else:
                self.aggregate_plot_control_layout.addWidget(intensity_button_label, 7, 0)

            self.mean_intensity_button = QRadioButton('Mean')
            self.mean_intensity_button.setFixedSize(QtCore.QSize(160, 50))
            self.mean_intensity_button.setContentsMargins(0, 0, 0, 0)
            self.mean_intensity_button.setStyleSheet(
                'color: black; font-style: normal; font-family: Helvetica; font-size: 28px;'
            )
            self.mean_intensity_button.pressed.connect(mean_intensity_button_clicked)
            if self.demo:
                self.scatterplot_sorting_layout.addWidget(self.mean_intensity_button, 1, 1, 1, 1)
            else:
                self.aggregate_plot_control_layout.addWidget(self.mean_intensity_button, 8, 0)

            self.variance_intensity_button = QRadioButton('Variance')
            self.variance_intensity_button.setFixedSize(QtCore.QSize(160, 50))
            self.variance_intensity_button.setContentsMargins(0, 0, 0, 0)
            self.variance_intensity_button.setStyleSheet(
                'color: black; font-style: normal; font-family: Helvetica; font-size: 28px;'
            )
            self.variance_intensity_button.pressed.connect(variance_intensity_button_clicked)
            if self.demo:
                self.scatterplot_sorting_layout.addWidget(
                    self.variance_intensity_button, 2, 1, 1, 1
                )
                # self.scatterplot_sorting_layout.setContentsMargins(0, 0, 30, 0)
                self.demo_layout.addLayout(self.scatterplot_sorting_layout, 1, 1, 2, 1)
            else:
                self.aggregate_plot_control_layout.addWidget(self.variance_intensity_button, 9, 0)

            # by default the intensities are computed via mean
            self.mean_intensity_button.setChecked(True)
            self.intensity_method = 'mean'
            # compute each sample's metric average (e.g., avg confidence for mnist) across all
            # transformations as intensity
            self.all_intensity_1 = np.mean(self.all_high_dim_points_1, axis=1)
            self.all_intensity_2 = np.mean(self.all_high_dim_points_2, axis=1)
            self.intensity_min = min(np.min(self.all_intensity_1), np.min(self.all_intensity_2))
            self.intensity_max = max(np.max(self.all_intensity_1), np.max(self.all_intensity_2))

        # show the scatter plot of dimension reduction result
        self.image_index = None
        self.slider_1_selected_index = None
        self.slider_2_selected_index = None
        if self.dr_result_sliders_existed:
            self.clear_layout(self.slider_1_layout)
            self.clear_layout(self.slider_2_layout)
            self.dr_result_sliders_existed = False
        # draw the scatter plot
        display_dimension_reduction()

        # demo mode automatically selects an image and trigger individual NERO
        if self.demo:
            # preselected the digit we want to use in paper
            # Note: this will create error when changing subset
            if self.mode == 'digit_recognition':
                # digit 4, 6 and 9
                selected_image = '/home/zhuokai/Desktop/UChicago/Research/nero_vis/qt_app/example_data/digit_recognition/MNIST_500/label_4_sample_6924.png'
                selected_image_index = self.all_images_paths.index(selected_image)
                # selected_image_index = 0
            elif self.mode == 'object_detection':
                # selected_image = '/home/zhuokai/Desktop/UChicago/Research/nero_vis/qt_app/example_data/object_detection/COCO_500/images/car_797_0.jpg'
                selected_image = '/home/zhuokai/Desktop/UChicago/Research/nero_vis/qt_app/example_data/object_detection/COCO_500/images/car_3572_3.jpg'
                selected_image_index = self.all_images_paths.index(selected_image)
                # selected_image_index = 0
            elif self.mode == 'piv':
                # take the worst-performing one by default
                selected_image_index = 0

            self.image_index = self.cur_class_indices[selected_image_index]

            print(f'Preselected image {self.image_index} from scatter plot')
            low_dim_scatter_clicked()

    # run model on the aggregate dataset
    def run_model_aggregated(self):

        if not self.aggregate_plot_control_existed:
            # initialize digit selection control
            self.init_aggregate_plot_control()

        if self.mode == 'digit_recognition':
            # all the rotation angles applied to the aggregated dataset
            self.all_aggregate_angles = list(range(0, 365, 5))

            # load from cache if available
            self.all_avg_accuracy_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_accuracy'
            )
            self.all_avg_accuracy_per_digit_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_accuracy_per_digit'
            )
            self.all_outputs_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_outputs'
            )

            self.all_avg_accuracy_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_accuracy'
            )
            self.all_avg_accuracy_per_digit_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_accuracy_per_digit'
            )
            self.all_outputs_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_outputs'
            )

            if not self.load_successfully:
                # average accuracies over all digits under all rotations, has shape (num_rotations, 1)
                self.all_avg_accuracy_1 = np.zeros(len(self.all_aggregate_angles))
                self.all_avg_accuracy_2 = np.zeros(len(self.all_aggregate_angles))
                # average accuracies of each digit under all rotations, has shape (num_rotations, 10)
                self.all_avg_accuracy_per_digit_1 = np.zeros((len(self.all_aggregate_angles), 10))
                self.all_avg_accuracy_per_digit_2 = np.zeros((len(self.all_aggregate_angles), 10))
                # output of each class's probablity of all samples, has shape (num_rotations, num_samples, 10)
                self.all_outputs_1 = np.zeros(
                    (len(self.all_aggregate_angles), len(self.loaded_images_pt), 10)
                )
                self.all_outputs_2 = np.zeros(
                    (len(self.all_aggregate_angles), len(self.loaded_images_pt), 10)
                )

                # for all the loaded images
                # for i, self.cur_rotation_angle in enumerate(range(0, 365, 5)):
                for i, self.cur_rotation_angle in enumerate(self.all_aggregate_angles):
                    print(f'\nAggregate mode: Rotated {self.cur_rotation_angle} degrees')
                    # self.all_angles.append(self.cur_rotation_angle)

                    (
                        avg_accuracy_1,
                        avg_accuracy_per_digit_1,
                        output_1,
                    ) = nero_run_model.run_mnist_once(
                        self.model_1,
                        self.loaded_images_pt,
                        self.loaded_images_labels,
                        batch_size=self.batch_size,
                        rotate_angle=self.cur_rotation_angle,
                    )

                    (
                        avg_accuracy_2,
                        avg_accuracy_per_digit_2,
                        output_2,
                    ) = nero_run_model.run_mnist_once(
                        self.model_2,
                        self.loaded_images_pt,
                        self.loaded_images_labels,
                        batch_size=self.batch_size,
                        rotate_angle=self.cur_rotation_angle,
                    )

                    # append to results
                    self.all_avg_accuracy_1[i] = avg_accuracy_1
                    self.all_avg_accuracy_2[i] = avg_accuracy_2
                    self.all_avg_accuracy_per_digit_1[i] = avg_accuracy_per_digit_1
                    self.all_avg_accuracy_per_digit_2[i] = avg_accuracy_per_digit_2
                    self.all_outputs_1[i] = output_1
                    self.all_outputs_2[i] = output_2

                # save to cache
                self.save_to_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_accuracy',
                    self.all_avg_accuracy_1,
                )
                self.save_to_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_accuracy_per_digit',
                    self.all_avg_accuracy_per_digit_1,
                )
                self.save_to_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_outputs',
                    self.all_outputs_1,
                )

                self.save_to_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_accuracy',
                    self.all_avg_accuracy_2,
                )
                self.save_to_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_accuracy_per_digit',
                    self.all_avg_accuracy_per_digit_2,
                )
                self.save_to_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_outputs',
                    self.all_outputs_2,
                )

            # display the result
            self.display_mnist_aggregate_result()

        elif self.mode == 'object_detection':
            # all the translations in x and y applied to the aggregated dataset
            self.x_translation = list(
                range(-self.image_size // 2, self.image_size // 2, self.translation_step_aggregate)
            )
            self.y_translation = list(
                range(-self.image_size // 2, self.image_size // 2, self.translation_step_aggregate)
            )

            # always try loading from cache
            self.aggregate_outputs_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_outputs'
            )
            self.aggregate_precision_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_precision'
            )
            self.aggregate_recall_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_recall'
            )
            self.aggregate_mAP_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_mAP'
            )
            self.aggregate_F_measure_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_F_measure'
            )

            self.aggregate_outputs_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_outputs'
            )
            self.aggregate_precision_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_precision'
            )
            self.aggregate_recall_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_recall'
            )
            self.aggregate_mAP_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_mAP'
            )
            self.aggregate_F_measure_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_F_measure'
            )

            if not self.load_successfully:
                # output of each sample for all translations, has shape (num_y_trans, num_x_trans, num_samples, num_samples, 7)
                self.aggregate_outputs_1 = np.zeros(
                    (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
                )
                self.aggregate_outputs_2 = np.zeros(
                    (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
                )

                # individual precision, recall, F measure and AP
                self.aggregate_precision_1 = np.zeros(
                    (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
                )
                self.aggregate_recall_1 = np.zeros(
                    (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
                )
                self.aggregate_F_measure_1 = np.zeros(
                    (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
                )
                # mAP does not have individuals
                self.aggregate_mAP_1 = np.zeros((len(self.y_translation), len(self.x_translation)))

                self.aggregate_precision_2 = np.zeros(
                    (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
                )
                self.aggregate_recall_2 = np.zeros(
                    (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
                )
                self.aggregate_F_measure_2 = np.zeros(
                    (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
                )
                # mAP does not have individuals
                self.aggregate_mAP_2 = np.zeros((len(self.y_translation), len(self.x_translation)))

                # for all the loaded images
                for y, y_tran in enumerate(self.y_translation):
                    for x, x_tran in enumerate(self.x_translation):
                        print(f'y_tran = {y_tran}, x_tran = {x_tran}')
                        # model 1 output
                        (
                            cur_qualified_output_1,
                            cur_precision_1,
                            cur_recall_1,
                            cur_F_measure_1,
                        ) = nero_run_model.run_coco_once(
                            'aggregate',
                            self.model_1_name,
                            self.model_1,
                            self.all_images_paths,
                            self.custom_coco_names,
                            self.pytorch_coco_names,
                            batch_size=self.batch_size,
                            x_tran=x_tran,
                            y_tran=y_tran,
                            coco_names=self.original_coco_names,
                        )

                        # save to result arrays
                        self.aggregate_outputs_1[y, x] = cur_qualified_output_1
                        self.aggregate_precision_1[y, x] = cur_precision_1
                        self.aggregate_recall_1[y, x] = cur_recall_1
                        self.aggregate_F_measure_1[y, x] = cur_F_measure_1
                        self.aggregate_mAP_1[y, x] = nero_utilities.compute_ap(
                            cur_recall_1, cur_precision_1
                        )

                        # model 2 output
                        (
                            cur_qualified_output_2,
                            cur_precision_2,
                            cur_recall_2,
                            cur_F_measure_2,
                        ) = nero_run_model.run_coco_once(
                            'aggregate',
                            self.model_2_name,
                            self.model_2,
                            self.all_images_paths,
                            self.custom_coco_names,
                            self.pytorch_coco_names,
                            batch_size=self.batch_size,
                            x_tran=x_tran,
                            y_tran=y_tran,
                            coco_names=self.original_coco_names,
                        )

                        # save to result arrays
                        self.aggregate_outputs_2[y, x] = cur_qualified_output_2
                        self.aggregate_precision_2[y, x] = cur_precision_2
                        self.aggregate_recall_2[y, x] = cur_recall_2
                        self.aggregate_F_measure_2[y, x] = cur_F_measure_2
                        self.aggregate_mAP_2[y, x] = nero_utilities.compute_ap(
                            cur_recall_2, cur_precision_2
                        )

                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_outputs',
                    content=self.aggregate_outputs_1,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_precision',
                    content=self.aggregate_precision_1,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_recall',
                    content=self.aggregate_recall_1,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_mAP',
                    content=self.aggregate_mAP_1,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_F_measure',
                    content=self.aggregate_F_measure_1,
                )

                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_outputs',
                    content=self.aggregate_outputs_2,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_precision',
                    content=self.aggregate_precision_2,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_recall',
                    content=self.aggregate_recall_2,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_mAP',
                    content=self.aggregate_mAP_2,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_F_measure',
                    content=self.aggregate_F_measure_2,
                )

            # load consensus, an alternative to ground truths, and associate losses
            # model 1
            self.aggregate_consensus_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus'
            )
            self.aggregate_consensus_outputs_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus_outputs'
            )
            self.aggregate_consensus_precision_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus_precision'
            )
            self.aggregate_consensus_recall_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus_recall'
            )
            self.aggregate_consensus_mAP_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus_mAP'
            )
            self.aggregate_consensus_F_measure_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus_F_measure'
            )

            # model 2
            self.aggregate_consensus_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus'
            )
            self.aggregate_consensus_outputs_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus_outputs'
            )
            self.aggregate_consensus_precision_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus_precision'
            )
            self.aggregate_consensus_recall_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus_recall'
            )
            self.aggregate_consensus_mAP_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus_mAP'
            )
            self.aggregate_consensus_F_measure_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus_F_measure'
            )

            # print(self.aggregate_consensus_1[0])
            # print(self.aggregate_consensus_1.shape)
            # print(self.aggregate_consensus_outputs_1.shape)
            # aggregate_consensus_outputs = np.zeros(
            #     (
            #         self.aggregate_consensus_outputs_1.shape[0]
            #         * self.aggregate_consensus_outputs_1.shape[1],
            #         8,
            #     )
            # )
            # for y in range(len(self.y_translation)):
            #     for x in range(len(self.x_translation)):
            #         aggregate_consensus_outputs[
            #             y * len(self.y_translation) + x
            #         ] = self.aggregate_consensus_outputs_1[y, x][0][0]
            # print(aggregate_consensus_outputs.shape)
            # np.save('aggregate_consensus.npy', self.aggregate_consensus_1)
            # np.save('aggregate_consensus_outputs.npy', aggregate_consensus_outputs)
            # exit()

            if not self.load_successfully:
                # if True:
                print(f'Computing consensus from model outputs')

                # compute consensus for two models
                self.aggregate_consensus_1 = self.compute_consensus(self.aggregate_outputs_1)
                self.aggregate_consensus_2 = self.compute_consensus(self.aggregate_outputs_2)

                # compute losses computed based on consensus (instead of ground truth)
                # model 1
                (
                    self.aggregate_consensus_outputs_1,
                    self.aggregate_consensus_precision_1,
                    self.aggregate_consensus_recall_1,
                    self.aggregate_consensus_mAP_1,
                    self.aggregate_consensus_F_measure_1,
                ) = self.compute_consensus_losses(
                    self.aggregate_outputs_1, self.aggregate_consensus_1
                )

                # model 2
                (
                    self.aggregate_consensus_outputs_2,
                    self.aggregate_consensus_precision_2,
                    self.aggregate_consensus_recall_2,
                    self.aggregate_consensus_mAP_2,
                    self.aggregate_consensus_F_measure_2,
                ) = self.compute_consensus_losses(
                    self.aggregate_outputs_2, self.aggregate_consensus_2
                )

                # save to cache
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus',
                    content=self.aggregate_consensus_1,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus_outputs',
                    content=self.aggregate_consensus_outputs_1,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus_precision',
                    content=self.aggregate_consensus_precision_1,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus_recall',
                    content=self.aggregate_consensus_recall_1,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus_mAP',
                    content=self.aggregate_consensus_mAP_1,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_consensus_F_measure',
                    content=self.aggregate_consensus_F_measure_1,
                )

                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus',
                    content=self.aggregate_consensus_2,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus_outputs',
                    content=self.aggregate_consensus_outputs_2,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus_precision',
                    content=self.aggregate_consensus_precision_2,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus_recall',
                    content=self.aggregate_consensus_recall_2,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus_mAP',
                    content=self.aggregate_consensus_mAP_2,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_consensus_F_measure',
                    content=self.aggregate_consensus_F_measure_2,
                )

            # display the result
            self.display_coco_aggregate_result()

        elif self.mode == 'piv':
            # Dihedral group4 transformations
            time_reverses = [0, 1]
            self.num_transformations = 16

            # always try loading from cache
            self.aggregate_outputs_1 = torch.from_numpy(
                self.load_from_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_outputs'
                )
            )
            self.aggregate_outputs_2 = torch.from_numpy(
                self.load_from_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_outputs'
                )
            )
            self.aggregate_ground_truths = torch.from_numpy(
                self.load_from_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_ground_truths'
                )
            )

            if not self.load_successfully:
                # output are dense 2D velocity field of the input image pairs
                # output for all transformation, has shape (num_transformations, num_samples, image_size, image_size, 2)
                self.aggregate_outputs_1 = torch.zeros(
                    (
                        self.num_transformations,
                        len(self.all_images_1_paths),
                        self.image_size,
                        self.image_size,
                        2,
                    )
                )
                self.aggregate_outputs_2 = torch.zeros(
                    (
                        self.num_transformations,
                        len(self.all_images_2_paths),
                        self.image_size,
                        self.image_size,
                        2,
                    )
                )
                self.aggregate_ground_truths = torch.zeros(
                    (
                        self.num_transformations,
                        len(self.all_labels_paths),
                        self.image_size,
                        self.image_size,
                        2,
                    )
                )

                # take a batch of images
                num_batches = int(len(self.all_images_1_paths) / self.batch_size)
                if len(self.all_images_1_paths) % self.batch_size != 0:
                    num_batches += 1
                # construct sample indices for each batch
                batch_indices = []
                for i in range(num_batches):
                    batch_indices.append(
                        (
                            i * self.batch_size,
                            min((i + 1) * self.batch_size, len(self.all_images_1_paths)),
                        )
                    )

                # go through each transformation type
                for is_time_reversed in time_reverses:
                    for transformation_index in range(8):
                        print(f'Transformation {is_time_reversed*8+transformation_index}')
                        # modify all current batch samples to one kind of transformation and run model with it
                        for index_range in batch_indices:
                            cur_images_1_paths = self.all_images_1_paths[
                                index_range[0] : index_range[1]
                            ]
                            cur_images_2_paths = self.all_images_2_paths[
                                index_range[0] : index_range[1]
                            ]
                            cur_labels_paths = self.all_labels_paths[
                                index_range[0] : index_range[1]
                            ]

                            batch_d4_images_1_pt = torch.zeros(
                                (len(cur_images_1_paths), self.image_size, self.image_size, 3)
                            )
                            batch_d4_images_2_pt = torch.zeros(
                                (len(cur_images_2_paths), self.image_size, self.image_size, 3)
                            )
                            batch_ground_truth = torch.zeros(
                                (len(cur_images_1_paths), self.image_size, self.image_size, 2)
                            )

                            # load and modify data of the current batch
                            for i in range(len(cur_images_1_paths)):
                                # load the data
                                cur_image_1_pil = Image.open(cur_images_1_paths[i]).convert('RGB')
                                cur_image_2_pil = Image.open(cur_images_2_paths[i]).convert('RGB')
                                # convert to torch tensor
                                cur_d4_image_1_pt = torch.from_numpy(np.asarray(cur_image_1_pil))
                                cur_d4_image_2_pt = torch.from_numpy(np.asarray(cur_image_2_pil))
                                # load the ground truth flow field
                                cur_ground_truth = torch.from_numpy(
                                    fz.read_flow(cur_labels_paths[i])
                                )

                                # modify the data
                                if is_time_reversed:
                                    (
                                        cur_d4_image_1_pt,
                                        cur_d4_image_2_pt,
                                        cur_ground_truth,
                                    ) = nero_transform.time_reverse_piv_data(
                                        cur_d4_image_1_pt, cur_d4_image_2_pt, cur_ground_truth
                                    )

                                # 0: no transformation (original)
                                if transformation_index == 0:
                                    batch_d4_images_1_pt[i] = cur_d4_image_1_pt.clone()
                                    batch_d4_images_2_pt[i] = cur_d4_image_2_pt.clone()
                                    batch_ground_truth[i] = cur_ground_truth.clone()

                                # 1: right diagonal flip (/)
                                elif transformation_index == 1:
                                    (
                                        batch_d4_images_1_pt[i],
                                        batch_d4_images_2_pt[i],
                                        batch_ground_truth[i],
                                    ) = nero_transform.flip_piv_data(
                                        cur_d4_image_1_pt,
                                        cur_d4_image_2_pt,
                                        cur_ground_truth,
                                        flip_type='right-diagonal',
                                    )
                                # 2: counter-clockwise 90 rotation
                                elif transformation_index == 2:
                                    (
                                        batch_d4_images_1_pt[i],
                                        batch_d4_images_2_pt[i],
                                        batch_ground_truth[i],
                                    ) = nero_transform.rotate_piv_data(
                                        cur_d4_image_1_pt, cur_d4_image_2_pt, cur_ground_truth, 90
                                    )
                                # 3: horizontal flip (by y axis)
                                elif transformation_index == 3:
                                    (
                                        batch_d4_images_1_pt[i],
                                        batch_d4_images_2_pt[i],
                                        batch_ground_truth[i],
                                    ) = nero_transform.flip_piv_data(
                                        cur_d4_image_1_pt,
                                        cur_d4_image_2_pt,
                                        cur_ground_truth,
                                        flip_type='horizontal',
                                    )
                                # 4: counter-clockwise 180 rotation
                                elif transformation_index == 4:
                                    (
                                        batch_d4_images_1_pt[i],
                                        batch_d4_images_2_pt[i],
                                        batch_ground_truth[i],
                                    ) = nero_transform.rotate_piv_data(
                                        cur_d4_image_1_pt, cur_d4_image_2_pt, cur_ground_truth, 180
                                    )
                                # 5: \ diagnal flip
                                elif transformation_index == 5:
                                    (
                                        batch_d4_images_1_pt[i],
                                        batch_d4_images_2_pt[i],
                                        batch_ground_truth[i],
                                    ) = nero_transform.flip_piv_data(
                                        cur_d4_image_1_pt,
                                        cur_d4_image_2_pt,
                                        cur_ground_truth,
                                        flip_type='left-diagonal',
                                    )
                                # 6: counter-clockwise 270 rotation
                                elif transformation_index == 6:
                                    (
                                        batch_d4_images_1_pt[i],
                                        batch_d4_images_2_pt[i],
                                        batch_ground_truth[i],
                                    ) = nero_transform.rotate_piv_data(
                                        cur_d4_image_1_pt, cur_d4_image_2_pt, cur_ground_truth, 270
                                    )
                                # 7: vertical flip (by x axis)
                                elif transformation_index == 7:
                                    (
                                        batch_d4_images_1_pt[i],
                                        batch_d4_images_2_pt[i],
                                        batch_ground_truth[i],
                                    ) = nero_transform.flip_piv_data(
                                        cur_d4_image_1_pt,
                                        cur_d4_image_2_pt,
                                        cur_ground_truth,
                                        flip_type='vertical',
                                    )

                            # run models on the current batch
                            cur_outputs_1 = nero_run_model.run_piv_once(
                                'aggregate',
                                self.model_1_name,
                                self.model_1,
                                batch_d4_images_1_pt,
                                batch_d4_images_2_pt,
                            )

                            cur_outputs_2 = nero_run_model.run_piv_once(
                                'aggregate',
                                self.model_2_name,
                                self.model_2,
                                batch_d4_images_1_pt,
                                batch_d4_images_2_pt,
                            )

                            # add to all outputs
                            # HS does not need further pixel normalization
                            if self.model_1_name == 'Gunnar-Farneback':
                                self.aggregate_outputs_1[
                                    is_time_reversed * 8 + transformation_index,
                                    index_range[0] : index_range[1],
                                ] = cur_outputs_1
                            else:
                                self.aggregate_outputs_1[
                                    is_time_reversed * 8 + transformation_index,
                                    index_range[0] : index_range[1],
                                ] = (
                                    cur_outputs_1 / self.image_size
                                )

                            if self.model_2_name == 'Gunnar-Farneback':
                                self.aggregate_outputs_2[
                                    is_time_reversed * 8 + transformation_index,
                                    index_range[0] : index_range[1],
                                ] = cur_outputs_2
                            else:
                                self.aggregate_outputs_2[
                                    is_time_reversed * 8 + transformation_index,
                                    index_range[0] : index_range[1],
                                ] = (
                                    cur_outputs_2 / self.image_size
                                )

                            self.aggregate_ground_truths[
                                is_time_reversed * 8 + transformation_index,
                                index_range[0] : index_range[1],
                            ] = batch_ground_truth

                # save to cache
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_outputs',
                    content=self.aggregate_outputs_1,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_outputs',
                    content=self.aggregate_outputs_2,
                )
                self.save_to_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_ground_truths',
                    content=self.aggregate_ground_truths,
                )

            # display the result
            self.display_piv_aggregate_result()

        # automatically run dimensin reduction in demo mode
        if self.demo:
            self.run_dimension_reduction()

    # run model on a single test sample
    def run_model_once(self):
        if self.mode == 'digit_recognition':
            self.output_1 = nero_run_model.run_mnist_once(self.model_1, self.cur_image_pt)
            self.output_2 = nero_run_model.run_mnist_once(self.model_2, self.cur_image_pt)

            # display result
            self.display_mnist_single_result(type='bar')

        elif self.mode == 'object_detection':

            self.output_1 = nero_run_model.run_coco_once(
                'single',
                self.model_1_name,
                self.model_1,
                self.cropped_image_pt,
                self.custom_coco_names,
                self.pytorch_coco_names,
                test_label=self.cur_image_label,
            )

            self.output_2 = nero_run_model.run_coco_once(
                'single',
                self.model_2_name,
                self.model_2,
                self.cropped_image_pt,
                self.custom_coco_names,
                self.pytorch_coco_names,
                test_label=self.cur_image_label,
            )

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
        bb_min_x = max(0, object_center_x - original_bb_width / 2)
        bb_max_x = min(image_size[1] - 1, object_center_x + original_bb_width / 2)
        bb_min_y = max(0, object_center_y - original_bb_height / 2)
        bb_max_y = min(image_size[0] - 1, object_center_y + original_bb_height / 2)

        return bb_min_x, bb_min_y, bb_max_x, bb_max_y

    # helper function that clamps consensus so that it could be plotted
    def update_consensus(self, cur_bounding_box, image_size, x_dist=0, y_dist=0):
        # convert key object bounding box to be based on extracted image
        x_min_center_bb = cur_bounding_box[0] - x_dist
        y_min_center_bb = cur_bounding_box[1] - y_dist
        x_max_center_bb = cur_bounding_box[2] - x_dist
        y_max_center_bb = cur_bounding_box[3] - y_dist

        # compute the center of the object in the extracted image
        object_center_x = (x_min_center_bb + x_max_center_bb) / 2
        object_center_y = (y_min_center_bb + y_max_center_bb) / 2

        # compute the width and height of the real bounding box of this object
        original_bb_width = cur_bounding_box[2] - cur_bounding_box[0]
        original_bb_height = cur_bounding_box[3] - cur_bounding_box[1]

        # compute the range of the bounding box, do the clamping if go out of extracted image
        bb_min_x = max(0, object_center_x - original_bb_width / 2)
        bb_max_x = min(image_size[1] - 1, object_center_x + original_bb_width / 2)
        bb_min_y = max(0, object_center_y - original_bb_height / 2)
        bb_max_y = min(image_size[0] - 1, object_center_y + original_bb_height / 2)

        return bb_min_x, bb_min_y, bb_max_x, bb_max_y

    # helper function on redisplaying COCO input image with FOV mask and ground truth labelling
    def display_coco_image(self, consensus_index=None):
        # display the whole image
        self.display_image()

        self.x_tran = self.cur_x_tran + self.x_translation[0]
        self.y_tran = self.cur_y_tran + self.y_translation[0]

        display_rect_width = self.display_image_size / 2
        display_rect_height = self.display_image_size / 2
        # since the translation measures on the movement of object instead of the point of view, the sign is reversed
        rect_center_x = self.display_image_size / 2 - self.x_tran * (
            self.display_image_size / self.uncropped_image_size
        )
        rect_center_y = self.display_image_size / 2 - self.y_tran * (
            self.display_image_size / self.uncropped_image_size
        )

        # draw rectangles on the displayed image to indicate scanning process
        painter = QtGui.QPainter(self.image_pixmap)
        # draw the rectangles
        cover_color = QtGui.QColor(65, 65, 65, 225)
        self.draw_fov_mask(
            painter,
            rect_center_x,
            rect_center_y,
            display_rect_width,
            display_rect_height,
            cover_color,
        )

        # end the painter
        painter.end()

        # draw ground truth or consensus label on the display image
        painter = QtGui.QPainter(self.image_pixmap)
        if self.use_consensus:
            # draw the consensus label
            self.cur_consensus_1 = self.update_consensus(
                self.aggregate_consensus_1[self.image_index][:4],
                (self.image_size, self.image_size),
                x_dist=-self.x_tran,
                y_dist=-self.y_tran,
            )
            self.cur_consensus_2 = self.update_consensus(
                self.aggregate_consensus_2[self.image_index][:4],
                (self.image_size, self.image_size),
                x_dist=-self.x_tran,
                y_dist=-self.y_tran,
            )
            # we have two sets of consensus from two sets of models outputs
            consensus_display_center_x_1 = (
                self.cur_consensus_1[0] + self.cur_consensus_1[2]
            ) / 2 * (display_rect_width / self.image_size) + (
                rect_center_x - display_rect_width / 2
            )
            consensus_display_center_y_1 = (
                self.cur_consensus_1[1] + self.cur_consensus_1[3]
            ) / 2 * (display_rect_height / self.image_size) + (
                rect_center_y - display_rect_height / 2
            )
            consensus_display_rect_width_1 = (
                self.cur_consensus_1[2] - self.cur_consensus_1[0]
            ) * (display_rect_width / self.image_size / 1.21)
            consensus_display_rect_height_1 = (
                self.cur_consensus_1[3] - self.cur_consensus_1[1]
            ) * (display_rect_height / self.image_size / 1.21)

            self.draw_rectangle(
                painter,
                consensus_display_center_x_1,
                consensus_display_center_y_1,
                consensus_display_rect_width_1,
                consensus_display_rect_height_1,
                color='yellow',
                label='Consensus_1',
            )

            consensus_display_center_x_2 = (
                self.cur_consensus_2[0] + self.cur_consensus_2[2]
            ) / 2 * (display_rect_width / self.image_size) + (
                rect_center_x - display_rect_width / 2
            )
            consensus_display_center_y_2 = (
                self.cur_consensus_2[1] + self.cur_consensus_2[3]
            ) / 2 * (display_rect_height / self.image_size) + (
                rect_center_y - display_rect_height / 2
            )
            consensus_display_rect_width_2 = (
                self.cur_consensus_2[2] - self.cur_consensus_2[0]
            ) * (display_rect_width / self.image_size / 1.21)
            consensus_display_rect_height_2 = (
                self.cur_consensus_2[3] - self.cur_consensus_2[1]
            ) * (display_rect_height / self.image_size / 1.21)

            self.draw_rectangle(
                painter,
                consensus_display_center_x_2,
                consensus_display_center_y_2,
                consensus_display_rect_width_2,
                consensus_display_rect_height_2,
                color='orange',
                label='Consensus_2',
            )
        else:
            # draw the ground truth label
            gt_display_center_x = (self.cur_image_label[0, 1] + self.cur_image_label[0, 3]) / 2 * (
                display_rect_width / self.image_size
            ) + (rect_center_x - display_rect_width / 2)
            gt_display_center_y = (self.cur_image_label[0, 2] + self.cur_image_label[0, 4]) / 2 * (
                display_rect_height / self.image_size
            ) + (rect_center_y - display_rect_height / 2)
            gt_display_rect_width = (self.cur_image_label[0, 3] - self.cur_image_label[0, 1]) * (
                display_rect_width / self.image_size / 1.21
            )
            gt_display_rect_height = (self.cur_image_label[0, 4] - self.cur_image_label[0, 2]) * (
                display_rect_height / self.image_size / 1.21
            )

            self.draw_rectangle(
                painter,
                gt_display_center_x,
                gt_display_center_y,
                gt_display_rect_width,
                gt_display_rect_height,
                color='yellow',
                label='Ground Truth',
            )
        painter.end()

        # update pixmap with the label
        self.image_label.setPixmap(self.image_pixmap)

        # force repaint
        self.image_label.repaint()

    # helper function on update the correct label when coco input is changed by user
    def update_coco_label(self):
        self.x_tran = self.cur_x_tran + self.x_translation[0]
        self.y_tran = self.cur_y_tran + self.y_translation[0]
        # modify the underlying image tensor accordingly
        # take the cropped part of the entire input image
        cur_center_x = self.center_x - self.x_tran
        cur_center_y = self.center_y - self.y_tran
        self.x_min = cur_center_x - self.image_size // 2
        self.x_max = cur_center_x + self.image_size // 2
        self.y_min = cur_center_y - self.image_size // 2
        self.y_max = cur_center_y + self.image_size // 2
        # model takes image between [0, 1]
        self.cropped_image_pt = (
            self.loaded_image_pt[self.y_min : self.y_max, self.x_min : self.x_max, :] / 255
        )

        self.cur_image_label = np.zeros((len(self.loaded_image_label), 6))
        for i in range(len(self.cur_image_label)):
            # object index
            self.cur_image_label[i, 0] = i
            # since PyTorch FRCNN has 0 as background
            self.cur_image_label[i, 5] = self.loaded_image_label[i, 4] + 1
            # modify the label accordingly
            self.cur_image_label[i, 1:5] = self.compute_label(
                self.loaded_image_label[i, :4],
                self.x_min,
                self.y_min,
                (self.image_size, self.image_size),
            )

    # modify the image tensor and the associated GIF as user rotates, flips or time-reverses
    def modify_display_gif(self):

        # torch rot treats ccw as positive
        if self.rotate_ccw:
            self.cur_image_1_pt = torch.rot90(self.cur_image_1_pt, 1)
            self.cur_image_2_pt = torch.rot90(self.cur_image_2_pt, 1)
            self.rotate_ccw = False

        elif self.rotate_cw:
            self.cur_image_1_pt = torch.rot90(self.cur_image_1_pt, -1)
            self.cur_image_2_pt = torch.rot90(self.cur_image_2_pt, -1)
            self.rotate_cw = False

        # vertical flip is by x axis
        elif self.vertical_flip:
            self.cur_image_1_pt = torch.flip(self.cur_image_1_pt, [0])
            self.cur_image_2_pt = torch.flip(self.cur_image_2_pt, [0])
            self.vertical_flip = False

        # horizontal flip is by y axis
        elif self.horizontal_flip:
            self.cur_image_1_pt = torch.flip(self.cur_image_1_pt, [1])
            self.cur_image_2_pt = torch.flip(self.cur_image_2_pt, [1])
            self.horizontal_flip = False

        # reverse the order in pair
        elif self.time_reverse:
            self.cur_image_2_pt, self.cur_image_1_pt = self.cur_image_1_pt, self.cur_image_2_pt

        # create new GIF
        display_image_1_pil = Image.fromarray(self.cur_image_1_pt.numpy(), 'RGB')
        display_image_2_pil = Image.fromarray(self.cur_image_2_pt.numpy(), 'RGB')
        other_images_pil = [
            display_image_1_pil,
            display_image_2_pil,
            display_image_2_pil,
            self.blank_image_pil,
        ]
        self.gif_path = os.path.join(
            self.cache_dir, self.loaded_image_1_name.split('.')[0] + '.gif'
        )
        display_image_1_pil.save(
            fp=self.gif_path,
            format='GIF',
            append_images=other_images_pil,
            save_all=True,
            duration=300,
            loop=0,
        )

        # get the rectangle index from cayley graph
        if self.time_reverse:
            if self.rectangle_index + 8 <= 15:
                self.rectangle_index += 8
            else:
                self.rectangle_index -= 8

            self.time_reverse = False
        else:
            self.rectangle_index = self.cayley_table[self.transform_index, self.rectangle_index]

    # run model on all the available transformations on a single sample
    def run_model_single(self):

        if self.mode == 'digit_recognition':
            # display the image
            self.display_image()

            # quantity that is displayed in the individual NERO plot
            self.all_angles = []
            self.all_quantities_1 = []
            self.all_quantities_2 = []

            # run all rotation test with 5 degree increment
            for i, self.cur_rotation_angle in enumerate(
                range(0, 360 + self.rotation_step, self.rotation_step)
            ):
                # print(f"\nRotated {self.cur_rotation_angle} degrees")
                self.all_angles.append(self.cur_rotation_angle)

                # take single result from aggregated result
                # all_outputs_1 has shape (num_rotations, num_samples, 10)
                self.output_1 = self.all_outputs_1[i, self.image_index, :]
                self.output_2 = self.all_outputs_2[i, self.image_index, :]

                # plotting the quantity regarding the correct label
                quantity_1 = self.output_1[self.loaded_image_label]
                quantity_2 = self.output_2[self.loaded_image_label]

                self.all_quantities_1.append(quantity_1)
                self.all_quantities_2.append(quantity_2)

                # print(f"Loaded quantity_1: {quantity_1}")

                # # compute single result in real-time (less efficient)
                # # rotate the image tensor
                # self.cur_image_pt = nero_transform.rotate_mnist_image(
                #     self.loaded_image_pt, self.cur_rotation_angle
                # )
                # # prepare image tensor for model purpose
                # self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)
                # # run model
                # self.output_1 = nero_run_model.run_mnist_once(self.model_1, self.cur_image_pt)
                # self.output_2 = nero_run_model.run_mnist_once(self.model_2, self.cur_image_pt)

                # # plotting the quantity regarding the correct label
                # quantity_1 = self.output_1[self.loaded_image_label]
                # quantity_2 = self.output_2[self.loaded_image_label]
                # print(f"New computed quantity_1: {quantity_1}")

                # self.all_quantities_1.append(quantity_1)
                # self.all_quantities_2.append(quantity_2)

            # display result
            # individual NERO plot
            self.display_mnist_single_result(type='polar')
            # detailed bar plot
            self.display_mnist_single_result(type='bar')

        elif self.mode == 'object_detection':
            # when this is called in the single case
            if self.data_mode == 'single':
                # all the x and y translations
                # x translates on columns, y translates on rows
                self.x_translation = list(
                    range(
                        -self.image_size // 2, self.image_size // 2, self.translation_step_single
                    )
                )
                self.y_translation = list(
                    range(
                        -self.image_size // 2, self.image_size // 2, self.translation_step_single
                    )
                )
                num_x_translations = len(self.x_translation)
                num_y_translations = len(self.y_translation)
                self.all_translations = np.zeros((num_y_translations, num_x_translations, 2))

                # always try loading from cache
                self.all_quantities_1 = self.load_from_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.image_index}'
                )
                self.all_quantities_2 = self.load_from_cache(
                    name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_{self.image_index}'
                )

                if not self.load_successfully:
                    self.all_quantities_1 = np.zeros((num_y_translations, num_x_translations, 8))
                    self.all_quantities_2 = np.zeros((num_y_translations, num_x_translations, 8))

                    for y, y_tran in enumerate(self.y_translation):
                        for x, x_tran in enumerate(self.x_translation):

                            # translation amout
                            # cur_x_tran and cur_y_tran are used to draw points on the heatmap to indicate translation amount
                            self.cur_x_tran = x_tran - self.x_translation[0]
                            self.cur_y_tran = y_tran - self.y_translation[0]
                            # all_translations are for book keeping
                            self.all_translations[y, x] = [x_tran, y_tran]

                            # udpate the correct coco label
                            self.update_coco_label()

                            # skip running model if using cache
                            if self.use_cache:
                                continue

                            # re-display image for each rectangle drawn every 8 steps
                            if (x_tran) % 2 == 0 and (y_tran) % 2 == 0:
                                self.display_coco_image()

                            # run the model
                            # update the model output
                            self.output_1 = nero_run_model.run_coco_once(
                                'single',
                                self.model_1_name,
                                self.model_1,
                                self.cropped_image_pt,
                                self.custom_coco_names,
                                self.pytorch_coco_names,
                                test_label=self.cur_image_label,
                            )

                            self.output_2 = nero_run_model.run_coco_once(
                                'single',
                                self.model_2_name,
                                self.model_2,
                                self.cropped_image_pt,
                                self.custom_coco_names,
                                self.pytorch_coco_names,
                                test_label=self.cur_image_label,
                            )

                            # plotting the quantity regarding the correct label
                            quantity_1 = self.output_1[0][0][0]
                            quantity_2 = self.output_2[0][0][0]
                            self.all_quantities_1[y, x] = quantity_1
                            self.all_quantities_2[y, x] = quantity_2

                    # save to cache
                    self.save_to_cache(
                        name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.image_index}',
                        content=self.all_quantities_1,
                    )
                    self.save_to_cache(
                        name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_{self.image_index}',
                        content=self.all_quantities_2,
                    )

                # display as the final x_tran, y_tran
                self.display_coco_image()

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
                    # since PyTorch FRCNN has 0 as background
                    self.cur_image_label[i, 5] = (
                        self.custom_coco_names.index(
                            self.original_coco_names[int(self.loaded_image_label[i, -1])]
                        )
                        + 1
                    )
                    # modify the label accordingly
                    self.cur_image_label[i, 1:5] = self.loaded_image_label[i, :4]

                # information needed for interactive display
                # bounding box center of the key object
                self.center_x = int((self.cur_image_label[0, 1] + self.cur_image_label[0, 3]) // 2)
                self.center_y = int((self.cur_image_label[0, 2] + self.cur_image_label[0, 4]) // 2)
                # just need information for interactive display
                self.x_min = self.center_x - self.image_size // 2
                self.x_max = self.center_x + self.image_size // 2
                self.y_min = self.center_y - self.image_size // 2
                self.y_max = self.center_y + self.image_size // 2
                # no transformation to start
                self.cur_x_tran = self.image_size // 2
                self.cur_y_tran = self.image_size // 2

                # initialiate the image object
                self.display_image()

                display_rect_width = self.display_image_size / 2
                display_rect_height = self.display_image_size / 2
                # since the translation measures on the movement of object instead of the point of view, the sign is reversed
                rect_center_x = self.display_image_size / 2
                rect_center_y = self.display_image_size / 2
                # draw rectangles on the displayed image to indicate scanning process
                painter = QtGui.QPainter(self.image_pixmap)
                # draw the rectangles
                cover_color = QtGui.QColor(65, 65, 65, 225)
                self.draw_fov_mask(
                    painter,
                    rect_center_x,
                    rect_center_y,
                    display_rect_width,
                    display_rect_height,
                    cover_color,
                )

                # re-compute the ground truth or consensus bounding boxes of the cropped image
                if self.use_consensus:
                    self.cur_consensus_1 = self.update_consensus(
                        self.aggregate_consensus_1[self.image_index][:4],
                        (self.image_size, self.image_size),
                    )
                    self.cur_consensus_2 = self.update_consensus(
                        self.aggregate_consensus_2[self.image_index][:4],
                        (self.image_size, self.image_size),
                    )

                    # draw the consensus label
                    gt_display_center_x_1 = (
                        self.cur_consensus_1[0] + self.cur_consensus_1[2]
                    ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                        rect_center_x - display_rect_width / 2
                    )
                    gt_display_center_y_1 = (
                        self.cur_consensus_1[1] + self.cur_consensus_1[3]
                    ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                        rect_center_y - display_rect_height / 2
                    )
                    gt_display_rect_width_1 = (
                        self.cur_consensus_1[2] - self.cur_consensus_1[0]
                    ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                    gt_display_rect_height_1 = (
                        self.cur_consensus_1[3] - self.cur_consensus_1[1]
                    ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                    self.draw_rectangle(
                        painter,
                        gt_display_center_x_1,
                        gt_display_center_y_1,
                        gt_display_rect_width_1,
                        gt_display_rect_height_1,
                        color='yellow',
                        label='Consensus_1',
                    )

                    gt_display_center_x_2 = (
                        self.cur_consensus_2[0] + self.cur_consensus_2[2]
                    ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                        rect_center_x - display_rect_width / 2
                    )
                    gt_display_center_y_2 = (
                        self.cur_consensus_2[1] + self.cur_consensus_2[3]
                    ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                        rect_center_y - display_rect_height / 2
                    )
                    gt_display_rect_width_2 = (
                        self.cur_consensus_2[2] - self.cur_consensus_2[0]
                    ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                    gt_display_rect_height_2 = (
                        self.cur_consensus_2[3] - self.cur_consensus_2[1]
                    ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                    self.draw_rectangle(
                        painter,
                        gt_display_center_x_2,
                        gt_display_center_y_2,
                        gt_display_rect_width_2,
                        gt_display_rect_height_2,
                        color='orange',
                        label='Consensus_2',
                    )
                else:
                    for i in range(len(self.cur_image_label)):
                        self.cur_image_label[i, 1:5] = self.compute_label(
                            self.loaded_image_label[i, :4],
                            self.x_min,
                            self.y_min,
                            (self.image_size, self.image_size),
                        )

                    # draw the ground truth label
                    gt_display_center_x = (
                        self.cur_image_label[0, 1] + self.cur_image_label[0, 3]
                    ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                        rect_center_x - display_rect_width / 2
                    )
                    gt_display_center_y = (
                        self.cur_image_label[0, 4] + self.cur_image_label[0, 2]
                    ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                        rect_center_y - display_rect_height / 2
                    )
                    gt_display_rect_width = (
                        self.cur_image_label[0, 3] - self.cur_image_label[0, 1]
                    ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                    gt_display_rect_height = (
                        self.cur_image_label[0, 4] - self.cur_image_label[0, 2]
                    ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                    self.draw_rectangle(
                        painter,
                        gt_display_center_x,
                        gt_display_center_y,
                        gt_display_rect_width,
                        gt_display_rect_height,
                        color='yellow',
                        label='Ground Truth',
                    )
                painter.end()

                # update pixmap with the label
                self.image_label.setPixmap(self.image_pixmap)

                # force repaint
                self.image_label.repaint()

            # display the individual NERO plot
            self.display_coco_single_result()

        elif self.mode == 'piv':

            if self.average_nero_checkbox.checkState() == QtCore.Qt.Checked:
                self.show_average = True
            else:
                self.show_average = False

            # flags on controlling current image tensor
            self.rotate_ccw = False
            self.rotate_cw = False
            self.vertical_flip = False
            self.horizontal_flip = False
            self.time_reverse = False

            # prepare input image transformations
            def init_input_control():
                # add buttons for controlling the single GIF
                self.gif_control_layout = QtWidgets.QVBoxLayout()
                self.gif_control_layout.setAlignment(QtGui.Qt.AlignTop)
                self.gif_control_layout.setContentsMargins(0, 0, 0, 0)
                if self.data_mode == 'single':
                    self.single_result_layout.addLayout(self.gif_control_layout, 2, 0)
                elif self.data_mode == 'aggregate':
                    if self.demo:
                        self.demo_layout.addLayout(self.gif_control_layout, 0, 5, 2, 1)
                    else:
                        self.aggregate_result_layout.addLayout(self.gif_control_layout, 2, 3)

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

            @QtCore.Slot()
            def rotate_90_ccw():
                self.rotate_ccw = True
                self.transform_index = 2
                print(f'Rotate 90 degrees counter clockwise')

                # modify the image, display and current triangle index
                self.modify_display_gif()

                # display the image
                self.display_image()

                # redraw the nero plot with new triangle display
                self.draw_piv_nero('single')
                # update detailed plot of PIV
                self.draw_piv_details()

            @QtCore.Slot()
            def rotate_90_cw():
                self.rotate_cw = True
                self.transform_index = 6
                print(f'Rotate 90 degrees clockwise')

                # modify the image, display and current triangle index
                self.modify_display_gif()

                # display the image
                self.display_image()

                # redraw the nero plot with new triangle display
                self.draw_piv_nero('single')
                # update detailed plot of PIV
                self.draw_piv_details()

            @QtCore.Slot()
            def vertical_flip():
                self.transform_index = 7
                self.vertical_flip = True
                print(f'Flip vertically')
                # modify the image, display and current triangle index
                self.modify_display_gif()

                # display the image
                self.display_image()

                # redraw the nero plot with new triangle display
                self.draw_piv_nero('single')
                # update detailed plot of PIV
                self.draw_piv_details()

            @QtCore.Slot()
            def horizontal_flip():
                self.horizontal_flip = True
                self.transform_index = 3
                print(f'Flip horizontally')
                # modify the image, display and current triangle index
                self.modify_display_gif()

                # display the image
                self.display_image()

                # redraw the nero plot with new triangle display
                self.draw_piv_nero('single')
                # update detailed plot of PIV
                self.draw_piv_details()

            @QtCore.Slot()
            def time_reverse():
                self.time_reverse = True
                print(f'Time reverse')
                # modify the image, display and current triangle index
                self.modify_display_gif()

                # display the image
                self.display_image()

                # redraw the nero plot with new triangle display
                self.draw_piv_nero('single')
                # update detailed plot of PIV
                self.draw_piv_details()

            # Dihedral group4 transformations plus time-reverse
            self.num_transformations = 16
            time_reverses = [0, 1]
            # keep track for all D4 transformation
            self.all_d4_images_1_pt = torch.zeros(
                (self.num_transformations, self.image_size, self.image_size, 3)
            )
            self.all_d4_images_2_pt = torch.zeros(
                (self.num_transformations, self.image_size, self.image_size, 3)
            )
            self.all_ground_truths = torch.zeros(
                (self.num_transformations, self.image_size, self.image_size, 2)
            )

            # input after transformation
            for is_time_reversed in time_reverses:
                if is_time_reversed:
                    (
                        cur_d4_image_1_pt,
                        cur_d4_image_2_pt,
                        cur_ground_truth,
                    ) = nero_transform.time_reverse_piv_data(
                        self.loaded_image_1_pt, self.loaded_image_2_pt, self.loaded_image_label_pt
                    )
                else:
                    cur_d4_image_1_pt = self.loaded_image_1_pt.clone()
                    cur_d4_image_2_pt = self.loaded_image_2_pt.clone()
                    cur_ground_truth = self.loaded_image_label_pt.clone()

                # 0: no transformation (original)
                self.all_d4_images_1_pt[is_time_reversed * 8 + 0] = cur_d4_image_1_pt.clone()
                self.all_d4_images_2_pt[is_time_reversed * 8 + 0] = cur_d4_image_2_pt.clone()
                self.all_ground_truths[is_time_reversed * 8 + 0] = cur_ground_truth.clone()

                # 1: right diagonal flip (/)
                (
                    self.all_d4_images_1_pt[is_time_reversed * 8 + 1],
                    self.all_d4_images_2_pt[is_time_reversed * 8 + 1],
                    self.all_ground_truths[is_time_reversed * 8 + 1],
                ) = nero_transform.flip_piv_data(
                    cur_d4_image_1_pt,
                    cur_d4_image_2_pt,
                    cur_ground_truth,
                    flip_type='right-diagonal',
                )
                # 2: counter-clockwise 90 rotation
                (
                    self.all_d4_images_1_pt[is_time_reversed * 8 + 2],
                    self.all_d4_images_2_pt[is_time_reversed * 8 + 2],
                    self.all_ground_truths[is_time_reversed * 8 + 2],
                ) = nero_transform.rotate_piv_data(
                    cur_d4_image_1_pt, cur_d4_image_2_pt, cur_ground_truth, 90
                )
                # 3: horizontal flip (by y axis)
                (
                    self.all_d4_images_1_pt[is_time_reversed * 8 + 3],
                    self.all_d4_images_2_pt[is_time_reversed * 8 + 3],
                    self.all_ground_truths[is_time_reversed * 8 + 3],
                ) = nero_transform.flip_piv_data(
                    cur_d4_image_1_pt, cur_d4_image_2_pt, cur_ground_truth, flip_type='horizontal'
                )
                # 4: counter-clockwise 180 rotation
                (
                    self.all_d4_images_1_pt[is_time_reversed * 8 + 4],
                    self.all_d4_images_2_pt[is_time_reversed * 8 + 4],
                    self.all_ground_truths[is_time_reversed * 8 + 4],
                ) = nero_transform.rotate_piv_data(
                    cur_d4_image_1_pt, cur_d4_image_2_pt, cur_ground_truth, 180
                )
                # 5: \ diagnal flip
                (
                    self.all_d4_images_1_pt[is_time_reversed * 8 + 5],
                    self.all_d4_images_2_pt[is_time_reversed * 8 + 5],
                    self.all_ground_truths[is_time_reversed * 8 + 5],
                ) = nero_transform.flip_piv_data(
                    cur_d4_image_1_pt,
                    cur_d4_image_2_pt,
                    cur_ground_truth,
                    flip_type='left-diagonal',
                )
                # 6: counter-clockwise 270 rotation
                (
                    self.all_d4_images_1_pt[is_time_reversed * 8 + 6],
                    self.all_d4_images_2_pt[is_time_reversed * 8 + 6],
                    self.all_ground_truths[is_time_reversed * 8 + 6],
                ) = nero_transform.rotate_piv_data(
                    cur_d4_image_1_pt, cur_d4_image_2_pt, cur_ground_truth, 270
                )
                # 7: vertical flip (by x axis)
                (
                    self.all_d4_images_1_pt[is_time_reversed * 8 + 7],
                    self.all_d4_images_2_pt[is_time_reversed * 8 + 7],
                    self.all_ground_truths[is_time_reversed * 8 + 7],
                ) = nero_transform.flip_piv_data(
                    cur_d4_image_1_pt, cur_d4_image_2_pt, cur_ground_truth, flip_type='vertical'
                )

            # when in single mode
            if self.data_mode == 'single':
                # initialize input image control
                init_input_control()

                # all_quantities has shape (16, 256, 256, 2)
                # always try loading from cache
                self.all_quantities_1 = torch.from_numpy(
                    self.load_from_cache(
                        name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.image_index}'
                    )
                )
                self.all_quantities_2 = torch.from_numpy(
                    self.load_from_cache(
                        name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_{self.image_index}'
                    )
                )

                if not self.load_successfully:
                    # each model output are dense 2D velocity field of the input image
                    self.all_quantities_1 = torch.zeros(
                        (self.num_transformations, self.image_size, self.image_size, 2)
                    )
                    self.all_quantities_2 = torch.zeros(
                        (self.num_transformations, self.image_size, self.image_size, 2)
                    )

                    # compute the result
                    for i in range(self.num_transformations):
                        image_1_pt = self.all_d4_images_1_pt[i]
                        image_2_pt = self.all_d4_images_2_pt[i]

                        print(f'Compute model outputs for D4 transformation {i}')

                        # run the model
                        quantity_1 = nero_run_model.run_piv_once(
                            'single', self.model_1_name, self.model_1, image_1_pt, image_2_pt
                        )

                        quantity_2 = nero_run_model.run_piv_once(
                            'single', self.model_2_name, self.model_2, image_1_pt, image_2_pt
                        )

                        # HS does not need further pixel normalization
                        if self.model_1_name == 'Gunnar-Farneback':
                            self.all_quantities_1[i] = quantity_1
                        else:
                            self.all_quantities_1[i] = quantity_1 / self.image_size

                        if self.model_2_name == 'Gunnar-Farneback':
                            self.all_quantities_2[i] = quantity_2
                        else:
                            self.all_quantities_2[i] = quantity_2 / self.image_size

                        self.all_quantities_1[i] = quantity_1 / self.image_size
                        self.all_quantities_2[i] = quantity_2 / self.image_size

                    # save to cache
                    self.save_to_cache(
                        name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.image_index}',
                        content=self.all_quantities_1.numpy(),
                    )
                    self.save_to_cache(
                        name=f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_2_cache_name}_{self.image_index}',
                        content=self.all_quantities_2.numpy(),
                    )

                # display the piv single case result
                self.rectangle_index = 0
                # default detail view starts at the center of the original rectangle
                # if self.show_average:
                #     self.piv_heatmap_click_enable = False
                # else:
                #     self.piv_heatmap_click_enable = True
                self.detail_rect_x = self.image_size // 2
                self.detail_rect_y = self.image_size // 2
                self.display_piv_single_result()

            # when in aggregate mode but a certain sample has been selected
            elif self.data_mode == 'aggregate':
                # display the GIF
                self.display_image()

                # initialize input image control
                init_input_control()

                # display the piv single case result
                self.rectangle_index = 0
                # if self.show_average:
                #     self.piv_heatmap_click_enable = False
                # else:
                #     self.piv_heatmap_click_enable = True
                self.detail_rect_x = self.image_size // 2
                self.detail_rect_y = self.image_size // 2
                self.display_piv_single_result()

    # draw a rectangle
    def draw_rectangle(
        self,
        painter,
        center_x,
        center_y,
        width,
        height,
        color=None,
        alpha=255,
        fill=None,
        boundary_width=5,
        label=None,
        image_size=None,
    ):
        if center_x == 0 and center_y == 0 and width == 0 and height == 0:
            return

        # left, top, width, height for QRect
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        if image_size != None:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x1 = min(image_size[0], x1)
            y1 = min(image_size[1], y1)

        rectangle = QtCore.QRect(x1, y1, width, height)

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
            text_rect = QtCore.QRect(center_x - width // 2, center_y - height // 2 - 20, 100, 20)
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

        # Add horizontal and vertical grid lines
        plot.addLine(x=0, pen=pg.mkPen('black', width=2, length=2))
        plot.addLine(y=0, pen=pg.mkPen('black', width=2, length=2))
        labels = ['0', '0.2', '0.4', '0.6', '0.8']

        for i, r in enumerate(np.arange(0, 1.2, 0.2)):
            # add circles
            circle = pg.QtWidgets.QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            circle.setPen(pg.mkPen('black', width=2))
            plot.addItem(circle)

            # add horizontal axis labels
            if i < len(labels):
                text = pg.TextItem(labels[i], color='black')
                text.setPos(r, 0)
                text.setFont(QFont('Helvetica', 16))
                plot.addItem(text)

        return plot

    # helper function on drawing the little circle on polar plot
    def draw_circle_on_polar(self):
        r = 0.06
        # transform to x and y coordinate
        cur_quantity_1_x = self.output_1[self.loaded_image_label] * np.cos(
            self.cur_rotation_angle / 180 * np.pi
        )
        cur_quantity_1_y = self.output_1[self.loaded_image_label] * np.sin(
            self.cur_rotation_angle / 180 * np.pi
        )
        # plot a circle item
        self.circle_1 = pg.QtWidgets.QGraphicsEllipseItem(
            cur_quantity_1_x - r / 2, cur_quantity_1_y - r / 2, r, r
        )
        self.circle_1.setPen(pg.mkPen('blue', width=7))
        self.polar_plot.addItem(self.circle_1)

        # transform to x and y coordinate
        cur_quantity_2_x = self.output_2[self.loaded_image_label] * np.cos(
            self.cur_rotation_angle / 180 * np.pi
        )
        cur_quantity_2_y = self.output_2[self.loaded_image_label] * np.sin(
            self.cur_rotation_angle / 180 * np.pi
        )
        # plot a circle item
        self.circle_2 = pg.QtWidgets.QGraphicsEllipseItem(
            cur_quantity_2_x - r / 2, cur_quantity_2_y - r / 2, r, r
        )
        self.circle_2.setPen(pg.mkPen('magenta', width=7))
        self.polar_plot.addItem(self.circle_2)

    # draw detailed look of COCO models output on cropped regions
    def draw_model_output(self, take_from_aggregate_output=False):
        def draw_detailed_plot(
            detailed_display_image, model_output, model_name, color, index=None
        ):

            # prepare a pixmap for the image
            detailed_image_pixmap = QPixmap(detailed_display_image)

            # add a new label for loaded image
            detailed_image_label = QLabel(self)

            # draw bounding boxes on the enlarged view
            # draw ground truth
            painter = QtGui.QPainter(detailed_image_pixmap)
            # draw the ground truth label
            if self.use_consensus:
                if index == 1:
                    consensus_display_center_x = (
                        (self.cur_consensus_1[0] + self.cur_consensus_1[2])
                        / 2
                        * (self.plot_size * 1.21 / self.image_size)
                    )
                    consensus_display_center_y = (
                        (self.cur_consensus_1[1] + self.cur_consensus_1[3])
                        / 2
                        * (self.plot_size * 1.21 / self.image_size)
                    )
                    consensus_display_rect_width = (
                        self.cur_consensus_1[2] - self.cur_consensus_1[0]
                    ) * (self.plot_size * 1.21 / self.image_size)
                    consensus_display_rect_height = (
                        self.cur_consensus_1[3] - self.cur_consensus_1[1]
                    ) * (self.plot_size * 1.21 / self.image_size)
                elif index == 2:
                    consensus_display_center_x = (
                        (self.cur_consensus_2[0] + self.cur_consensus_2[2])
                        / 2
                        * (self.plot_size * 1.21 / self.image_size)
                    )
                    consensus_display_center_y = (
                        (self.cur_consensus_2[1] + self.cur_consensus_2[3])
                        / 2
                        * (self.plot_size * 1.21 / self.image_size)
                    )
                    consensus_display_rect_width = (
                        self.cur_consensus_2[2] - self.cur_consensus_2[0]
                    ) * (self.plot_size * 1.21 / self.image_size)
                    consensus_display_rect_height = (
                        self.cur_consensus_2[3] - self.cur_consensus_2[1]
                    ) * (self.plot_size * 1.21 / self.image_size)

                self.draw_rectangle(
                    painter,
                    consensus_display_center_x,
                    consensus_display_center_y,
                    consensus_display_rect_width,
                    consensus_display_rect_height,
                    color='yellow',
                    alpha=166,
                    label=f'Consensus_{index}',
                )
            else:
                gt_display_center_x = (
                    (self.cur_image_label[0, 1] + self.cur_image_label[0, 3])
                    / 2
                    * (self.plot_size * 1.21 / self.image_size)
                )
                gt_display_center_y = (
                    (self.cur_image_label[0, 2] + self.cur_image_label[0, 4])
                    / 2
                    * (self.plot_size * 1.21 / self.image_size)
                )
                gt_display_rect_width = (
                    self.cur_image_label[0, 3] - self.cur_image_label[0, 1]
                ) * (self.plot_size * 1.21 / self.image_size)
                gt_display_rect_height = (
                    self.cur_image_label[0, 4] - self.cur_image_label[0, 2]
                ) * (self.plot_size * 1.21 / self.image_size)
                self.draw_rectangle(
                    painter,
                    gt_display_center_x,
                    gt_display_center_y,
                    gt_display_rect_width,
                    gt_display_rect_height,
                    color='yellow',
                    alpha=166,
                    label='Ground Truth',
                )

            # box from model
            bounding_boxes = model_output[0][0][:, :4]
            confidences = model_output[0][0][:, 4]
            ious = model_output[0][0][:, 6]
            # showing a maximum of 3 bounding boxes
            num_boxes_1 = min(3, len(bounding_boxes))
            for i in range(num_boxes_1):
                center_x = (
                    (bounding_boxes[i, 0] + bounding_boxes[i, 2])
                    / 2
                    * (self.plot_size * 1.21 / self.image_size)
                )
                center_y = (
                    (bounding_boxes[i, 1] + bounding_boxes[i, 3])
                    / 2
                    * (self.plot_size * 1.21 / self.image_size)
                )
                model_display_rect_width = (bounding_boxes[i, 2] - bounding_boxes[i, 0]) * (
                    self.plot_size * 1.21 / self.image_size
                )
                model_display_rect_height = (bounding_boxes[i, 3] - bounding_boxes[i, 1]) * (
                    self.plot_size * 1.21 / self.image_size
                )

                # compute alpha value based on confidence
                cur_alpha = nero_utilities.lerp(confidences[i], 0, 1, 255 / 4, 255)
                # compute boundary width based on IOU
                cur_boundary_width = nero_utilities.lerp(ious[i], 0, 1, 2, 5)

                self.draw_rectangle(
                    painter,
                    center_x,
                    center_y,
                    model_display_rect_width,
                    model_display_rect_height,
                    color,
                    alpha=cur_alpha,
                    boundary_width=cur_boundary_width,
                    label=f'Prediction {i+1}',
                    image_size=(
                        self.plot_size * 1.21,
                        self.plot_size * 1.21,
                    ),
                )

            painter.end()

            # put pixmap in the label
            detailed_image_label.setPixmap(detailed_image_pixmap)

            # force repaint
            detailed_image_label.repaint()

            # detailed information showed next to the image
            class TableModel(QtCore.QAbstractTableModel):
                def __init__(self, data):
                    super(TableModel, self).__init__()
                    self._data = data

                # When subclassing QAbstractTableModel, you must implement rowCount(), columnCount(), and data()
                def data(self, index, role):
                    if role == PySide6.QtCore.Qt.DisplayRole:
                        # See below for the nested-list data structure.
                        # .row() indexes into the outer list,
                        # .column() indexes into the sub-list
                        value = self._data[index.row()][index.column()]

                        if isinstance(value, float):
                            # Render float to 3 dp
                            return '%.3f' % value

                        if isinstance(value, str):
                            # Render strings without quotes
                            return '%s' % value

                        # Default (anything not captured above: e.g. int)
                        return value

                def rowCount(self, index):
                    # The length of the outer list.
                    return len(self._data)

                def columnCount(self, index):
                    # The following takes the first sub-list, and returns
                    # the length (only works if all rows are an equal length)
                    return len(self._data[0])

            detailed_image_table = QtWidgets.QTableView()
            detailed_image_table.setShowGrid(False)
            detailed_image_table.horizontalHeader().hide()
            detailed_image_table.verticalHeader().hide()
            detailed_image_table.horizontalScrollBar().hide()
            detailed_image_table.verticalScrollBar().hide()
            detailed_image_table.setFrameStyle(QtWidgets.QFrame.NoFrame)
            detailed_image_table.setColumnWidth(0, 20)
            detailed_image_table.setStyleSheet(
                'color: black; font-family: Helvetica; font-style: normal; font-size: 24px'
            )

            data = [[' ', ' ', ' ', ' '], ['Pred #', 'Class', 'Conf', 'IOU']]
            for i in range(num_boxes_1):
                data.append(
                    [
                        i + 1,
                        self.custom_coco_names[int(model_output[0][0][i, 5] - 1)],
                        model_output[0][0][i, 4],
                        model_output[0][0][i, 6],
                    ]
                )

            model = TableModel(data)
            detailed_image_table.setModel(model)

            return detailed_image_label, detailed_image_table

        # size of the enlarged image
        # convert and resize current selected FOV to QImage for display purpose
        if take_from_aggregate_output:
            # still needs the new cropped image for detail model readout vis
            self.detailed_display_image = nero_utilities.tensor_to_qt_image(
                self.loaded_image_pt[self.y_min : self.y_max, self.x_min : self.x_max, :],
                self.display_image_size * 1.21,
            )
            if self.use_consensus:
                self.output_1 = [
                    [
                        self.aggregate_consensus_outputs_1[self.block_y, self.block_x][
                            self.image_index
                        ]
                    ]
                ]
                self.output_2 = [
                    [
                        self.aggregate_consensus_outputs_2[self.block_y, self.block_x][
                            self.image_index
                        ]
                    ]
                ]
            else:
                self.output_1 = [
                    [self.aggregate_outputs_1[self.block_y, self.block_x][self.image_index]]
                ]
                self.output_2 = [
                    [self.aggregate_outputs_2[self.block_y, self.block_x][self.image_index]]
                ]
        else:
            self.detailed_display_image = nero_utilities.tensor_to_qt_image(
                self.loaded_image_pt[self.y_min : self.y_max, self.x_min : self.x_max, :],
                self.display_image_size * 1.21,
            )
            # run model with the cropped view
            self.cropped_image_pt = (
                self.loaded_image_pt[self.y_min : self.y_max, self.x_min : self.x_max, :] / 255
            )
            self.run_model_once()

        # display for model 1
        self.detailed_image_label_1, self.detailed_text_label_1 = draw_detailed_plot(
            self.detailed_display_image, self.output_1, self.model_1_name, 'blue', 1
        )
        # display for model 2
        self.detailed_image_label_2, self.detailed_text_label_2 = draw_detailed_plot(
            self.detailed_display_image, self.output_2, self.model_2_name, 'magenta', 2
        )
        # spacer item between image and text
        image_text_spacer = QtWidgets.QSpacerItem(
            self.plot_size, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        if self.data_mode == 'single':
            self.single_result_layout.addWidget(self.detailed_image_label_1, 3, 1)
            self.single_result_layout.addItem(image_text_spacer, 4, 2)
            self.single_result_layout.addWidget(self.detailed_text_label_1, 4, 1)
            self.single_result_layout.addWidget(self.detailed_image_label_2, 3, 2)
            self.single_result_layout.addWidget(self.detailed_text_label_2, 4, 2)
        elif self.data_mode == 'aggregate':
            if self.demo:
                self.demo_layout.addWidget(self.detailed_image_label_1, 2, 3, 3, 1)
                self.demo_layout.addWidget(self.detailed_text_label_1, 3, 4, 4, 1)
                self.demo_layout.addWidget(self.detailed_image_label_2, 4, 3, 3, 1)
                self.demo_layout.addWidget(self.detailed_text_label_2, 5, 4, 4, 1)
            else:
                self.aggregate_result_layout.addWidget(self.detailed_image_label_1, 2, 4)
                self.aggregate_result_layout.addItem(image_text_spacer, 3, 5)
                self.aggregate_result_layout.addWidget(self.detailed_text_label_1, 3, 4)
                self.aggregate_result_layout.addWidget(self.detailed_image_label_2, 2, 5)
                self.aggregate_result_layout.addWidget(self.detailed_text_label_2, 3, 5)

    def display_image(self):
        # add a new label for loaded image if no image has existed
        if not self.image_existed:
            self.image_label = QLabel(self)
            self.image_label.setAlignment(QtCore.Qt.AlignTop)
            # self.image_label.setFixedSize(1000, 1000)
            self.image_existed = True
            # no additional content margin to prevent cutoff on images

            # add the image label to the layout
            if self.data_mode == 'single':
                self.single_result_layout.addWidget(self.image_label, 1, 0)

            elif self.data_mode == 'aggregate':
                if self.mode == 'digit_recognition':
                    if self.demo:
                        # self.image_layout = QtWidgets.QHBoxLayout()
                        # self.image_layout.addWidget(self.image_label)
                        # self.image_layout.setContentsMargins(0, 0, 0, 0)
                        # self.demo_layout.addLayout(self.image_layout, 0, 3, 3, 1)
                        self.demo_layout.addWidget(self.image_label, 0, 3, 3, 1)
                    else:
                        self.aggregate_result_layout.addWidget(self.image_label, 1, 4, 2, 1)
                elif self.mode == 'object_detection':
                    if self.demo:
                        self.demo_layout.addWidget(self.image_label, 1, 4, 2, 1)
                    else:
                        self.aggregate_result_layout.addWidget(self.image_label, 1, 3, 3, 1)
                elif self.mode == 'piv':
                    if self.demo:
                        self.demo_layout.addWidget(self.image_label, 0, 4, 3, 1)
                    else:
                        self.aggregate_result_layout.addWidget(self.image_label, 1, 3, 3, 1)

        if self.mode == 'digit_recognition' or self.mode == 'object_detection':

            # prepare a pixmap for the image
            self.image_pixmap = QPixmap(self.cur_display_image)

            # single pixmap in the label
            self.image_label.setPixmap(self.image_pixmap)

            if self.mode == 'digit_recognition':
                # plot_size should be bigger than the display_size, so that some margins exist
                self.image_label.setFixedSize(
                    self.display_image_size + 100, self.display_image_size + 50
                )
                self.image_label.setContentsMargins(100, 50, 0, 0)  # left, top, right, bottom
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
                            display_rect_width = self.display_image_size / 2
                            display_rect_height = self.display_image_size / 2

                            # restrict x and y value
                            if rect_center_x + display_rect_width / 2 >= self.display_image_size:
                                rect_center_x = self.display_image_size - display_rect_width / 2
                            elif rect_center_x - display_rect_width / 2 < 0:
                                rect_center_x = display_rect_width / 2

                            if rect_center_y + display_rect_height / 2 >= self.display_image_size:
                                rect_center_y = self.display_image_size - display_rect_height / 2
                            elif rect_center_y - display_rect_height / 2 < 0:
                                rect_center_y = display_rect_height / 2

                            # draw rectangle on the displayed image to indicate scanning process
                            painter = QtGui.QPainter(self.image_pixmap)
                            # draw the rectangles that cover the non field of view
                            cover_color = QtGui.QColor(65, 65, 65, 225)
                            self.draw_fov_mask(
                                painter,
                                rect_center_x,
                                rect_center_y,
                                display_rect_width,
                                display_rect_height,
                                cover_color,
                            )

                            # how much the fov center is away from the image center
                            x_dist = (rect_center_x - self.display_image_size / 2) / (
                                self.display_image_size / self.uncropped_image_size
                            )
                            y_dist = (rect_center_y - self.display_image_size / 2) / (
                                self.display_image_size / self.uncropped_image_size
                            )
                            # compute rectangle center wrt to the original image
                            cur_center_x = self.center_x + x_dist
                            cur_center_y = self.center_y + y_dist
                            self.x_min = int(cur_center_x - self.image_size / 2)
                            self.x_max = int(cur_center_x + self.image_size / 2)
                            self.y_min = int(cur_center_y - self.image_size / 2)
                            self.y_max = int(cur_center_y + self.image_size / 2)

                            # ground truth/consensus label bounding boxes of the cropped image
                            if self.use_consensus:
                                self.cur_consensus_1 = self.update_consensus(
                                    self.aggregate_consensus_1[self.image_index][:4],
                                    (self.image_size, self.image_size),
                                    x_dist,
                                    y_dist,
                                )
                                self.cur_consensus_2 = self.update_consensus(
                                    self.aggregate_consensus_2[self.image_index][:4],
                                    (self.image_size, self.image_size),
                                    x_dist,
                                    y_dist,
                                )

                                # draw the consensus label
                                gt_display_center_x_1 = (
                                    self.cur_consensus_1[0] + self.cur_consensus_1[2]
                                ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                                    rect_center_x - display_rect_width / 2
                                )
                                gt_display_center_y_1 = (
                                    self.cur_consensus_1[1] + self.cur_consensus_1[3]
                                ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                                    rect_center_y - display_rect_height / 2
                                )
                                gt_display_rect_width_1 = (
                                    self.cur_consensus_1[2] - self.cur_consensus_1[0]
                                ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                                gt_display_rect_height_1 = (
                                    self.cur_consensus_1[3] - self.cur_consensus_1[1]
                                ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                                self.draw_rectangle(
                                    painter,
                                    gt_display_center_x_1,
                                    gt_display_center_y_1,
                                    gt_display_rect_width_1,
                                    gt_display_rect_height_1,
                                    color='yellow',
                                    label='Consensus_1',
                                )

                                gt_display_center_x_2 = (
                                    self.cur_consensus_2[0] + self.cur_consensus_2[2]
                                ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                                    rect_center_x - display_rect_width / 2
                                )
                                gt_display_center_y_2 = (
                                    self.cur_consensus_2[1] + self.cur_consensus_2[3]
                                ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                                    rect_center_y - display_rect_height / 2
                                )
                                gt_display_rect_width_2 = (
                                    self.cur_consensus_2[2] - self.cur_consensus_2[0]
                                ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                                gt_display_rect_height_2 = (
                                    self.cur_consensus_2[3] - self.cur_consensus_2[1]
                                ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                                self.draw_rectangle(
                                    painter,
                                    gt_display_center_x_2,
                                    gt_display_center_y_2,
                                    gt_display_rect_width_2,
                                    gt_display_rect_height_2,
                                    color='orange',
                                    label='Consensus_2',
                                )
                            else:
                                for i in range(len(self.cur_image_label)):
                                    self.cur_image_label[i, 1:5] = self.compute_label(
                                        self.loaded_image_label[i, :4],
                                        self.x_min,
                                        self.y_min,
                                        (self.image_size, self.image_size),
                                    )

                                # draw the ground truth label
                                gt_display_center_x = (
                                    self.cur_image_label[0, 1] + self.cur_image_label[0, 3]
                                ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                                    rect_center_x - display_rect_width / 2
                                )
                                gt_display_center_y = (
                                    self.cur_image_label[0, 4] + self.cur_image_label[0, 2]
                                ) / 2 * (self.display_image_size / self.uncropped_image_size) + (
                                    rect_center_y - display_rect_height / 2
                                )
                                gt_display_rect_width = (
                                    self.cur_image_label[0, 3] - self.cur_image_label[0, 1]
                                ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                                gt_display_rect_height = (
                                    self.cur_image_label[0, 4] - self.cur_image_label[0, 2]
                                ) * (self.display_image_size / self.uncropped_image_size / 1.21)
                                self.draw_rectangle(
                                    painter,
                                    gt_display_center_x,
                                    gt_display_center_y,
                                    gt_display_rect_width,
                                    gt_display_rect_height,
                                    color='yellow',
                                    label='Ground Truth',
                                )
                            painter.end()

                            # update pixmap with the label
                            self.image_label.setPixmap(self.image_pixmap)

                            # force repaint
                            self.image_label.repaint()

                            # show corresponding translation amount on the heatmap
                            # translation amout for plotting in heatmap
                            self.cur_x_tran = (
                                self.image_size - 1 - (x_dist - self.x_translation[0] + 1)
                            )
                            self.cur_y_tran = (
                                self.image_size - 1 - (y_dist - self.y_translation[0] + 1)
                            )
                            self.draw_coco_nero(mode='single')

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
            self.image_label.setFixedSize(self.plot_size * 1.3, self.plot_size * 1.3)
            image_gif = QtGui.QMovie(self.gif_path)
            gif_size = QtCore.QSize(self.plot_size * 1.2, self.plot_size * 1.2)
            image_gif.setScaledSize(gif_size)
            # add to the label
            self.image_label.setMovie(image_gif)
            image_gif.start()

    # helper function on drawing mask on input COCO image (to highlight the current FOV)
    def draw_fov_mask(
        self,
        painter,
        rect_center_x,
        rect_center_y,
        display_rect_width,
        display_rect_height,
        cover_color,
    ):
        # draw the rectangles
        # top
        top_rect_center_x = (0 + self.display_image_size) / 2
        top_rect_center_y = (0 + rect_center_y - display_rect_height / 2) / 2
        top_display_rect_width = self.display_image_size
        top_display_rect_height = top_rect_center_y * 2
        self.draw_rectangle(
            painter,
            top_rect_center_x,
            top_rect_center_y,
            top_display_rect_width,
            top_display_rect_height,
            fill=cover_color,
        )
        # bottom
        bottom_rect_center_x = (0 + self.display_image_size) / 2
        bottom_rect_center_y = (
            rect_center_y + display_rect_height / 2 + self.display_image_size
        ) / 2
        bottom_display_rect_width = self.display_image_size
        bottom_display_rect_height = (self.display_image_size - bottom_rect_center_y) * 2
        self.draw_rectangle(
            painter,
            bottom_rect_center_x,
            bottom_rect_center_y,
            bottom_display_rect_width,
            bottom_display_rect_height,
            fill=cover_color,
        )
        # left
        left_rect_center_x = (0 + rect_center_x - display_rect_width / 2) / 2
        left_rect_center_y = rect_center_y
        left_display_rect_width = rect_center_x - display_rect_width / 2
        left_display_rect_height = (
            self.display_image_size - top_display_rect_height - bottom_display_rect_height
        )
        self.draw_rectangle(
            painter,
            left_rect_center_x,
            left_rect_center_y,
            left_display_rect_width,
            left_display_rect_height,
            fill=cover_color,
        )
        # right
        right_rect_center_x = (
            rect_center_x + display_rect_width / 2 + self.display_image_size
        ) / 2
        right_rect_center_y = rect_center_y
        right_display_rect_width = self.display_image_size - (
            rect_center_x + display_rect_width / 2
        )
        right_display_rect_height = (
            self.display_image_size - top_display_rect_height - bottom_display_rect_height
        )
        self.draw_rectangle(
            painter,
            right_rect_center_x,
            right_rect_center_y,
            right_display_rect_width,
            right_display_rect_height,
            fill=cover_color,
        )

    # helper function on drawing individual heatmap (called by both individual and aggregate cases)
    def draw_individual_heatmap(self, mode, data, heatmap=None, scatter_item=None, title=None):

        # color map
        self.color_map = pg.colormap.get('viridis')
        self.color_bar = pg.ColorBarItem(
            values=self.cm_range,
            colorMap=self.color_map,
            interactive=False,
            orientation='horizontal',
            width=30,
        )
        # # add colorbar to a specific place if in demo mode
        # if self.demo:
        #     # add colorbar to a specific place if in demo mode
        #     dummy_view = pg.GraphicsLayoutWidget()
        #     dummy_plot = pg.PlotItem()
        #     dummy_plot.layout.setContentsMargins(0, 50, 10, 0)
        #     dummy_plot.setFixedHeight(0)
        #     dummy_plot.setFixedWidth(self.plot_size * 1.2)
        #     dummy_plot.hideAxis('bottom')
        #     dummy_plot.hideAxis('left')
        #     dummy_view.addItem(dummy_plot)
        #     dummy_image = pg.ImageItem()
        #     self.color_bar.setImageItem(dummy_image, insert_in=dummy_plot)
        #     # self.demo_layout.addWidget(dummy_view, 1, 2, 1, 2)
        #     self.scatterplot_sorting_layout.addWidget(dummy_view, 3, 0, 1, 2)

        if self.mode == 'object_detection':
            # viewbox that contains the heatmap
            view_box = pg.ViewBox(invertY=True)
            view_box.setAspectLocked(lock=True)

            # single mode needs to have input view_box, heatmap and scatter_item for interactively handling
            if mode == 'single':
                heatmap_plot = pg.PlotItem(viewBox=view_box, title=title)
                heatmap.setOpts(axisOrder='row-major')
                heatmap.setImage(data)
                # add image to the viewbox
                view_box.addItem(heatmap)
                # so that showing indicator at the boundary does not jitter the plot
                view_box.disableAutoRange()

                # small indicator on where the translation is at
                scatter_point = [
                    {
                        'pos': (
                            self.cur_x_tran + self.translation_step_single // 2,
                            self.cur_y_tran + self.translation_step_single // 2,
                        ),
                        'size': self.translation_step_single,
                        'pen': {'color': 'red', 'width': 3},
                        'brush': (0, 0, 0, 0),
                    }
                ]

                # add points to the item
                scatter_item.setData(scatter_point)
                heatmap_plot.addItem(scatter_item)

            elif mode == 'aggregate':
                # heatmap = pg.ImageItem()
                heatmap.setImage(data)
                view_box.addItem(heatmap)
                heatmap_plot = pg.PlotItem(viewBox=view_box, title=title)

            x_label_style = {'color': 'black', 'font-size': '16pt', 'text': 'Translation in x'}
            heatmap_plot.getAxis('bottom').setLabel(**x_label_style)
            heatmap_plot.getAxis('bottom').setStyle(tickLength=0, showValues=False)

            y_label_style = {'color': 'black', 'font-size': '16pt', 'text': 'Translation in y'}
            heatmap_plot.getAxis('left').setLabel(**y_label_style)
            heatmap_plot.getAxis('left').setStyle(tickLength=0, showValues=False)

            # disable being able to move plot around
            heatmap_plot.setMouseEnabled(x=False, y=False)

        elif self.mode == 'piv':

            # when we are not showing the detail NERO
            if self.show_average:
                for y in range(4):
                    for x in range(4):
                        data_mean = np.mean(data[y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256])
                        data[y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256] = data_mean

            # single mode needs to have input view_box, heatmap and scatter_item for interactively handling
            if mode == 'single':
                view_box = pg.ViewBox(invertY=True)
                view_box.setAspectLocked(lock=True)
                heatmap_plot = pg.PlotItem(viewBox=view_box, title=title)
                heatmap.setOpts(axisOrder='row-major')
                heatmap.setImage(data)
                # add image to the viewbox
                view_box.addItem(heatmap)
                # so that showing indicator at the boundary does not jitter the plot
                view_box.disableAutoRange()

                # small indicator on where the translation is at
                # take the corresponding one from rectangle index
                self.rect_index_y, self.rect_index_x = np.where(
                    self.piv_nero_layout == self.rectangle_index
                )
                # np.where returns ndarray, but we know there is only one
                self.rect_index_x = self.rect_index_x[0]
                self.rect_index_y = self.rect_index_y[0]
                # rect_x is the column, rect_y is the row (image coordinate)
                rect_x = self.rect_index_x * self.image_size + self.image_size // 2
                rect_y = self.rect_index_y * self.image_size + self.image_size // 2

                # draw lines that distinguish between different transformations
                # line color is the average of all plot color
                scatter_lut = self.color_map.getLookupTable(
                    start=self.cm_range[1], stop=self.cm_range[0], nPts=500, alpha=False
                )
                line_color = QtGui.QColor(
                    scatter_lut[249][0], scatter_lut[249][1], scatter_lut[249][2]
                )
                for i in range(1, 4):
                    # horizontal
                    heatmap_plot.plot(
                        [0, self.image_size * 4],
                        [self.image_size * i, self.image_size * i],
                        pen=QtGui.QPen(line_color, 4),
                    )
                    # vertical
                    heatmap_plot.plot(
                        [self.image_size * i, self.image_size * i],
                        [0, self.image_size * 4],
                        pen=QtGui.QPen(line_color, 4),
                    )

                # when clicked, display the orbit position selection rectangle
                self.scatter_point = [
                    {
                        'pos': (rect_x, rect_y),
                        'size': self.image_size,
                        'pen': {'color': 'red', 'width': 4},
                        'brush': (0, 0, 0, 0),
                    }
                ]
                # add points to the item
                scatter_item.setData(self.scatter_point)
                heatmap_plot.addItem(scatter_item)

            elif mode == 'aggregate':
                view_box = pg.ViewBox(invertY=True)
                view_box.setAspectLocked(lock=True)
                heatmap = pg.ImageItem()
                heatmap.setImage(data)
                view_box.addItem(heatmap)
                view_box.disableAutoRange()
                heatmap_plot = pg.PlotItem(viewBox=view_box, title=title)

                # draw lines that distinguish between different transformations
                # line color is the average of all plot color
                scatter_lut = self.color_map.getLookupTable(
                    start=self.cm_range[1], stop=self.cm_range[0], nPts=500, alpha=False
                )
                line_color = QtGui.QColor(
                    scatter_lut[249][0], scatter_lut[249][1], scatter_lut[249][2]
                )
                for i in range(1, 4):
                    # horizontal
                    heatmap_plot.plot(
                        [0, self.image_size * 4],
                        [self.image_size * i, self.image_size * i],
                        pen=QtGui.QPen(line_color, 4),
                    )
                    # vertical
                    heatmap_plot.plot(
                        [self.image_size * i, self.image_size * i],
                        [0, self.image_size * 4],
                        pen=QtGui.QPen(line_color, 4),
                    )

            heatmap_plot.getAxis('bottom').setStyle(tickLength=0, showValues=False)
            heatmap_plot.getAxis('left').setStyle(tickLength=0, showValues=False)

            # in show_average mode, also show the orbit indicator
            if self.show_average:

                # for each rectangle
                time_reverses = [0, 1]
                for is_time_reversed in time_reverses:
                    # original
                    if not is_time_reversed:
                        original_F_pil = Image.open('symbols/F.png').convert('RGBA')
                        # convert to torch tensor
                        original_F_np = np.array(original_F_pil)
                        original_F_np = np.transpose(original_F_np, axes=(1, 0, 2))

                        """
                        2'  2(Rot90)            1(right diag flip)   1'
                        3'  3(hori flip)        0(original)          0'
                        4'  4(Rot180)           7(vert flip)         7'
                        5'  5(left diag flip)   6(Rot270)            6'
                        """
                        # 0
                        pos_x = 2 * self.image_size + 127
                        pos_y = 1 * self.image_size + 127
                        cur_F_np = original_F_np.copy()
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 1
                        pos_x = 2 * self.image_size + 127
                        pos_y = 0 * self.image_size + 127
                        # right diag = rot90 ccw + horizontal flip
                        cur_F_np = np.flipud(np.rot90(original_F_np, k=-1))
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 2
                        pos_x = 1 * self.image_size + 127
                        pos_y = 0 * self.image_size + 127
                        # rot90
                        cur_F_np = np.rot90(original_F_np, k=-1)
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 3
                        pos_x = 1 * self.image_size + 127
                        pos_y = 1 * self.image_size + 127
                        # horizontal flip (flip image x == flip array y)
                        cur_F_np = np.flipud(original_F_np)
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 4
                        pos_x = 1 * self.image_size + 127
                        pos_y = 2 * self.image_size + 127
                        # rot180
                        cur_F_np = np.rot90(original_F_np, k=-2)
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 5
                        pos_x = 1 * self.image_size + 127
                        pos_y = 3 * self.image_size + 127
                        # left diag = rot90 ccw + flip vertical
                        cur_F_np = np.fliplr(np.rot90(original_F_np, k=-1))
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 6
                        pos_x = 2 * self.image_size + 127
                        pos_y = 3 * self.image_size + 127
                        # rot270
                        cur_F_np = np.rot90(original_F_np, k=-3)
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 7
                        pos_x = 2 * self.image_size + 127
                        pos_y = 2 * self.image_size + 127
                        # vertical flip
                        cur_F_np = np.fliplr(original_F_np)
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                    else:
                        original_F_pil = Image.open('symbols/F_reversed.png').convert('RGBA')
                        # convert to torch tensor
                        original_F_np = np.array(original_F_pil)
                        original_F_np = np.transpose(original_F_np, axes=(1, 0, 2))

                        """
                        2'  2(Rot90)            1(right diag flip)   1'
                        3'  3(hori flip)        0(original)          0'
                        4'  4(Rot180)           7(vert flip)         7'
                        5'  5(left diag flip)   6(Rot270)            6'
                        """
                        # 0'
                        pos_x = 3 * self.image_size + 127
                        pos_y = 1 * self.image_size + 127
                        cur_F_np = original_F_np
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 1'
                        pos_x = 3 * self.image_size + 127
                        pos_y = 0 * self.image_size + 127
                        # right diag = rot90 ccw + horizontal flip
                        cur_F_np = np.flipud(np.rot90(original_F_np, k=-1))
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 2'
                        pos_x = 0 * self.image_size + 127
                        pos_y = 0 * self.image_size + 127
                        # rot90
                        cur_F_np = np.rot90(original_F_np, k=-1)
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 3'
                        pos_x = 0 * self.image_size + 127
                        pos_y = 1 * self.image_size + 127
                        # horizontal flip
                        cur_F_np = np.flipud(original_F_np)
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 4'
                        pos_x = 0 * self.image_size + 127
                        pos_y = 2 * self.image_size + 127
                        # rot180
                        cur_F_np = np.rot90(original_F_np, k=-2)
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 5'
                        pos_x = 0 * self.image_size + 127
                        pos_y = 3 * self.image_size + 127
                        # left diag = rot90 ccw + flip vertical
                        cur_F_np = np.fliplr(np.rot90(original_F_np, k=-1))
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 6'
                        pos_x = 3 * self.image_size + 127
                        pos_y = 3 * self.image_size + 127
                        # rot270
                        cur_F_np = np.rot90(original_F_np, k=-3)
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

                        # 7'
                        pos_x = 3 * self.image_size + 127
                        pos_y = 2 * self.image_size + 127
                        # vertical flip
                        cur_F_np = np.fliplr(original_F_np)
                        cur_F_image = pg.ImageItem()
                        cur_F_image.setImage(cur_F_np)
                        cur_F_image.setPos(pos_x, pos_y)
                        heatmap_plot.addItem(cur_F_image)

            # Not letting user zoom
            heatmap_plot.setMouseEnabled(x=False, y=False)

        # demo mode has colorbar elsewhere
        if self.demo:
            self.color_bar.setImageItem(heatmap)
        else:
            self.color_bar.setImageItem(heatmap, insert_in=heatmap_plot)

        return heatmap_plot

    # helper function on drawing detailed heatmap (called by both individual and aggregate cases in PIV)
    def draw_piv_detail_heatmap(self, data, heatmap):
        view_box = pg.ViewBox(invertY=True)
        view_box.setAspectLocked(lock=True)
        detail_heatmap_plot = pg.PlotItem(viewBox=view_box)
        heatmap.setOpts(axisOrder='row-major')
        heatmap.setImage(data)
        # use the same colorbar as the individual NERO plot
        self.color_bar.setImageItem(heatmap)
        # add image to the viewbox
        view_box.addItem(heatmap)
        # so that showing indicator at the boundary does not jitter the plot
        view_box.disableAutoRange()

        # small indicator on where the quiver plot displays
        detail_scatter_item = pg.ScatterPlotItem(pxMode=False)
        detail_scatter_item.setSymbol('s')
        detail_scatter_point = [
            {
                'pos': (self.detail_rect_x, self.detail_rect_y),
                'size': 8,
                'pen': {'color': 'red', 'width': 4},
                'brush': (0, 0, 0, 0),
            }
        ]
        # add points to the item
        detail_scatter_item.setData(detail_scatter_point)
        detail_heatmap_plot.addItem(detail_scatter_item)

        detail_heatmap_plot.getAxis('bottom').setStyle(tickLength=0, showValues=False)
        detail_heatmap_plot.getAxis('left').setStyle(tickLength=0, showValues=False)

        return detail_heatmap_plot

    # draw NERO plots for COCO experiment
    def draw_coco_nero(self, mode):

        # used to pass into subclass
        outer_self = self
        # subclass of ImageItem that reimplements the control methods
        class COCO_heatmap(pg.ImageItem):
            def __init__(self, plot_type, index):
                super().__init__()
                self.plot_type = plot_type
                self.index = index

            def mouseClickEvent(self, event):
                if self.plot_type == 'single':
                    print(f'Clicked on heatmap at ({event.pos().x()}, {event.pos().y()})')
                    # the position of un-repeated aggregate result
                    outer_self.block_x = int(
                        np.floor(event.pos().x() // outer_self.translation_step_single)
                    )
                    outer_self.block_y = int(
                        np.floor(event.pos().y() // outer_self.translation_step_single)
                    )

                    # in COCO mode, clicked location indicates translation
                    # draw a point(rect) that represents current selection of location
                    # although in this case we are taking results from the aggregate result
                    # we need these locations for input modification
                    outer_self.cur_x_tran = (
                        int(np.floor(event.pos().x() // outer_self.translation_step_single))
                        * outer_self.translation_step_single
                    )
                    outer_self.cur_y_tran = (
                        int(np.floor(event.pos().y() // outer_self.translation_step_single))
                        * outer_self.translation_step_single
                    )
                    outer_self.x_tran = outer_self.cur_x_tran + outer_self.x_translation[0]
                    outer_self.y_tran = outer_self.cur_y_tran + outer_self.y_translation[0]

                    # udpate the correct coco label
                    outer_self.update_coco_label()

                    # update the input image with FOV mask and ground truth labelling
                    outer_self.display_coco_image()

                    # redisplay model output (result taken from the aggregate results)
                    if outer_self.data_mode == 'aggregate':
                        outer_self.draw_model_output(take_from_aggregate_output=True)
                    else:
                        outer_self.draw_model_output()

                    # remove existing selection indicater from both scatter plots
                    outer_self.heatmap_plot_1.removeItem(outer_self.scatter_item_1)
                    outer_self.heatmap_plot_2.removeItem(outer_self.scatter_item_2)

                    # new scatter points
                    scatter_point = [
                        {
                            'pos': (
                                outer_self.cur_x_tran + outer_self.translation_step_single // 2,
                                outer_self.cur_y_tran + outer_self.translation_step_single // 2,
                            ),
                            'size': outer_self.translation_step_single,
                            'pen': {'color': 'red', 'width': 3},
                            'brush': (0, 0, 0, 0),
                        }
                    ]

                    # add points to both views
                    outer_self.scatter_item_1.setData(scatter_point)
                    outer_self.scatter_item_2.setData(scatter_point)
                    outer_self.heatmap_plot_1.addItem(outer_self.scatter_item_1)
                    outer_self.heatmap_plot_2.addItem(outer_self.scatter_item_2)

            def mouseDragEvent(self, event):
                if self.plot_type == 'single':
                    # if event.button() != QtCore.Qt.LeftButton:
                    #     event.ignore()
                    #     return
                    # print(event.pos())
                    if event.isStart():
                        print('Dragging starts', event.pos())

                    elif event.isFinish():
                        print('Dragging stops', event.pos())

                    else:
                        print('Drag', event.pos())

            def hoverEvent(self, event):
                if not event.isExit():
                    block_x = int(np.floor(event.pos().x() // outer_self.translation_step_single))
                    block_y = int(np.floor(event.pos().y() // outer_self.translation_step_single))
                    if self.index == 1:
                        hover_text = str(
                            round(outer_self.cur_aggregate_plot_quantity_1[block_y][block_x], 3)
                        )
                    elif self.index == 2:
                        hover_text = str(
                            round(outer_self.cur_aggregate_plot_quantity_2[block_y][block_x], 3)
                        )

                    self.setToolTip(hover_text)

        # add to general layout
        if mode == 'single':
            # check if the data is in shape (self.image_size, self.image_size)
            if self.cur_single_plot_quantity_1.shape != (self.image_size, self.image_size):
                # repeat in row
                temp = np.repeat(
                    self.cur_single_plot_quantity_1,
                    self.image_size / self.cur_single_plot_quantity_1.shape[1],
                    axis=0,
                )
                # repeat in column
                data_1 = np.repeat(
                    temp, self.image_size / self.cur_single_plot_quantity_1.shape[0], axis=1
                )
            else:
                data_1 = self.cur_single_plot_quantity_1

            if self.cur_single_plot_quantity_2.shape != (self.image_size, self.image_size):
                # repeat in row
                temp = np.repeat(
                    self.cur_single_plot_quantity_2,
                    self.image_size / self.cur_single_plot_quantity_2.shape[1],
                    axis=0,
                )
                # repeat in column
                data_2 = np.repeat(
                    temp, self.image_size / self.cur_single_plot_quantity_2.shape[0], axis=1
                )
            else:
                data_2 = self.cur_single_plot_quantity_2

            # both heatmap views
            self.heatmap_view_1 = pg.GraphicsLayoutWidget()
            self.heatmap_view_1.ci.layout.setContentsMargins(0, 0, 0, 0)  # left top right bottom
            self.heatmap_view_1.setFixedSize(self.plot_size * 1.2, self.plot_size * 1.2)
            self.heatmap_view_2 = pg.GraphicsLayoutWidget()
            self.heatmap_view_2.ci.layout.setContentsMargins(0, 0, 0, 0)  # left top right bottom
            self.heatmap_view_2.setFixedSize(self.plot_size * 1.2, self.plot_size * 1.2)

            self.single_nero_1 = COCO_heatmap(plot_type='single', index=1)
            self.single_nero_2 = COCO_heatmap(plot_type='single', index=2)
            self.scatter_item_1 = pg.ScatterPlotItem(pxMode=False)
            self.scatter_item_1.setSymbol('s')
            self.scatter_item_2 = pg.ScatterPlotItem(pxMode=False)
            self.scatter_item_2.setSymbol('s')
            # all quantities plotted will have range from 0 to 1
            self.cm_range = (0, 1)
            self.heatmap_plot_1 = self.draw_individual_heatmap(
                'single', data_1, self.single_nero_1, self.scatter_item_1
            )
            self.heatmap_plot_2 = self.draw_individual_heatmap(
                'single', data_2, self.single_nero_2, self.scatter_item_2
            )

            # add to view
            self.heatmap_view_1.addItem(self.heatmap_plot_1)
            self.heatmap_view_2.addItem(self.heatmap_plot_2)

            if self.data_mode == 'single':
                self.single_result_layout.addWidget(self.heatmap_view_1, 1, 1)
                self.single_result_layout.addWidget(self.heatmap_view_2, 1, 2)
            elif self.data_mode == 'aggregate':
                if self.demo:
                    self.demo_layout.addWidget(self.heatmap_view_1, 3, 2, 1, 1)
                    self.demo_layout.addWidget(self.heatmap_view_2, 5, 2, 1, 1)
                else:
                    self.aggregate_result_layout.addWidget(self.heatmap_view_1, 1, 4)
                    self.aggregate_result_layout.addWidget(self.heatmap_view_2, 1, 5)

        elif mode == 'aggregate':
            # check if the data is in shape (self.image_size, self.image_size)
            if self.cur_aggregate_plot_quantity_1.shape != (self.image_size, self.image_size):
                # repeat in row
                temp = np.repeat(
                    self.cur_aggregate_plot_quantity_1,
                    self.image_size / self.cur_aggregate_plot_quantity_1.shape[1],
                    axis=0,
                )
                # repeat in column
                data_1 = np.repeat(
                    temp, self.image_size / self.cur_aggregate_plot_quantity_1.shape[0], axis=1
                )
            else:
                data_1 = self.cur_aggregate_plot_quantity_1

            if self.cur_aggregate_plot_quantity_2.shape != (self.image_size, self.image_size):
                # repeat in row
                temp = np.repeat(
                    self.cur_aggregate_plot_quantity_2,
                    self.image_size / self.cur_aggregate_plot_quantity_2.shape[1],
                    axis=0,
                )
                # repeat in column
                data_2 = np.repeat(
                    temp, self.image_size / self.cur_aggregate_plot_quantity_2.shape[0], axis=1
                )
            else:
                data_2 = self.cur_aggregate_plot_quantity_2

            # heatmap view
            self.aggregate_heatmap_view_1 = pg.GraphicsLayoutWidget()
            self.aggregate_heatmap_view_1.ci.layout.setContentsMargins(
                0, 0, 0, 0
            )  # left top right bottom
            self.aggregate_heatmap_view_1.setFixedSize(
                self.plot_size * 1.35, self.plot_size * 1.35
            )

            self.aggregate_heatmap_view_2 = pg.GraphicsLayoutWidget()
            self.aggregate_heatmap_view_2.ci.layout.setContentsMargins(
                0, 0, 0, 0
            )  # left top right bottom
            self.aggregate_heatmap_view_2.setFixedSize(
                self.plot_size * 1.35, self.plot_size * 1.35
            )

            self.aggregate_nero_1 = COCO_heatmap(plot_type='aggregate', index=1)
            self.aggregate_nero_2 = COCO_heatmap(plot_type='aggregate', index=2)

            self.cm_range = (0, 1)

            self.aggregate_nero_1 = COCO_heatmap(plot_type='aggregate', index=1)
            self.aggregate_nero_2 = COCO_heatmap(plot_type='aggregate', index=2)
            self.aggregate_heatmap_plot_1 = self.draw_individual_heatmap(
                'aggregate', data_1, self.aggregate_nero_1
            )
            self.aggregate_heatmap_plot_2 = self.draw_individual_heatmap(
                'aggregate', data_2, self.aggregate_nero_2
            )

            # add to view
            self.aggregate_heatmap_view_1.addItem(self.aggregate_heatmap_plot_1)
            self.aggregate_heatmap_view_2.addItem(self.aggregate_heatmap_plot_2)

            if self.demo:
                self.demo_layout.addWidget(self.aggregate_heatmap_view_1, 2, 0, 3, 1)
                self.demo_layout.addWidget(self.aggregate_heatmap_view_2, 4, 0, 3, 1)
            else:
                self.aggregate_result_layout.addWidget(self.aggregate_heatmap_view_1, 1, 1)
                self.aggregate_result_layout.addWidget(self.aggregate_heatmap_view_2, 1, 2)

    def draw_aggregate_polar(self):
        # initialize view and plot
        polar_view = pg.GraphicsLayoutWidget()
        polar_view.setBackground('white')
        # polar plot larger than others because it occupies two rows
        polar_view.setFixedSize(self.plot_size * 1.7, self.plot_size * 1.7)
        polar_view.ci.layout.setContentsMargins(0, 0, 50, 0)
        self.aggregate_polar_plot = polar_view.addPlot(row=0, col=0)
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
        # Aggregated NERO plot for model 1
        for i in range(len(self.all_aggregate_angles)):
            radian = self.all_aggregate_angles[i] / 180 * np.pi
            # plot selected digit's average accuracy/confidence across all rotations
            if self.quantity_name == 'Accuracy':
                if self.class_selection == 'all':
                    cur_quantity_1 = self.all_avg_accuracy_1[i]
                else:
                    cur_quantity_1 = self.all_avg_accuracy_per_digit_1[i][self.class_selection]
            elif self.quantity_name == 'Confidence':
                all_confidences = []
                if self.class_selection == 'all':
                    # output of each class's probablity of all samples, has shape (num_rotations, num_samples, 10)
                    for j in range(len(self.all_outputs_1[0])):
                        all_confidences.append(
                            self.all_outputs_1[i][j][self.loaded_images_labels[j]]
                        )

                    # take the mean correct confidence
                    cur_quantity_1 = np.mean(all_confidences)
                else:
                    for j in range(len(self.all_outputs_1[0])):
                        if self.loaded_images_labels[j] == self.class_selection:
                            all_confidences.append(
                                self.all_outputs_1[i][j][self.loaded_images_labels[j]]
                            )

                    # take the mean correct confidence
                    cur_quantity_1 = np.mean(all_confidences)

            # Transform to cartesian and plot
            x_1 = cur_quantity_1 * np.cos(radian)
            y_1 = cur_quantity_1 * np.sin(radian)
            all_x_1.append(x_1)
            all_y_1.append(y_1)
            all_points_1.append(
                {
                    'pos': (x_1, y_1),
                    'size': 0.05,
                    'pen': {'color': 'w', 'width': 0.1},
                    'brush': QtGui.QColor('blue'),
                }
            )

            # Aggregated NERO plot for model 2
            if self.quantity_name == 'Accuracy':
                if self.class_selection == 'all':
                    cur_quantity_2 = self.all_avg_accuracy_2[i]
                else:
                    cur_quantity_2 = self.all_avg_accuracy_per_digit_2[i][self.class_selection]
            elif self.quantity_name == 'Confidence':
                all_confidences = []
                if self.class_selection == 'all':
                    # output of each class's probablity of all samples, has shape (num_rotations, num_samples, 10)
                    for j in range(len(self.all_outputs_2[0])):
                        all_confidences.append(
                            self.all_outputs_2[i][j][self.loaded_images_labels[j]]
                        )

                    # take the mean correct confidence
                    cur_quantity_2 = np.mean(all_confidences)
                else:
                    for j in range(len(self.all_outputs_2[0])):
                        if self.loaded_images_labels[j] == self.class_selection:
                            all_confidences.append(
                                self.all_outputs_2[i][j][self.loaded_images_labels[j]]
                            )

                    # take the mean correct confidence
                    cur_quantity_2 = np.mean(all_confidences)

            # Transform to cartesian and plot
            x_2 = cur_quantity_2 * np.cos(radian)
            y_2 = cur_quantity_2 * np.sin(radian)
            all_x_2.append(x_2)
            all_y_2.append(y_2)
            all_points_2.append(
                {
                    'pos': (x_2, y_2),
                    'size': 0.05,
                    'pen': {'color': 'w', 'width': 0.1},
                    'brush': QtGui.QColor('magenta'),
                }
            )

        # draw lines to better show shape
        self.aggregate_polar_plot.plot(all_x_1, all_y_1, pen=QtGui.QPen(QtGui.Qt.blue, 0.03))
        self.aggregate_polar_plot.plot(all_x_2, all_y_2, pen=QtGui.QPen(QtGui.Qt.magenta, 0.03))

        # add points to the item
        self.aggregate_scatter_items.addPoints(all_points_1)
        self.aggregate_scatter_items.addPoints(all_points_2)

        # add points to the plot
        self.aggregate_polar_plot.addItem(self.aggregate_scatter_items)

        # fix zoom level
        # self.polar_plot.vb.scaleBy((0.5, 0.5))
        self.aggregate_polar_plot.setMouseEnabled(x=False, y=False)

        # add the plot view to the layout
        if self.demo:
            self.demo_layout.addWidget(polar_view, 2, 0, 5, 1)
        else:
            self.aggregate_result_layout.addWidget(polar_view, 1, 1, 2, 2)

    def draw_triangle(
        self, painter, points, pen_color=None, brush_color=None, boundary_width=None
    ):

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

    # draws dihedral 4 visualization to be the NERO plot for PIV experiment
    def draw_piv_nero(self, mode):

        # used to pass into subclass
        outer_self = self

        # individual nero plot class
        # subclass of ImageItem that reimplements the control methods
        class PIV_heatmap(pg.ImageItem):
            def mouseClickEvent(self, event):
                # if outer_self.piv_heatmap_click_enable:
                print(
                    f'Clicked on individual PIV NERO plot at ({event.pos().x()}, {event.pos().y()})'
                )
                # in PIV mode, a pop up window shows the nearby area's quiver plot
                rect_x = int(event.pos().x() // outer_self.image_size)
                rect_y = int(event.pos().y() // outer_self.image_size)

                # current/new rectangle selection index
                outer_self.rectangle_index = outer_self.piv_nero_layout[rect_y, rect_x]
                # from rectangle index get the needed transformations to the original image
                """
                2'  2(Rot90)            1(right diag flip)   1'
                3'  3(hori flip)        0(original)          0'
                4'  4(Rot180)           7(vert flip)         7'
                5'  5(left diag flip)   6(Rot270)            6'
                """
                is_reversed = outer_self.rectangle_index // 8
                transformation_index = outer_self.rectangle_index % 8

                if is_reversed:
                    start_image_1_pt, start_image_2_pt, _ = nero_transform.time_reverse_piv_data(
                        outer_self.cur_image_1_pt, outer_self.cur_image_2_pt, np.zeros(1)
                    )
                else:
                    start_image_1_pt = outer_self.cur_image_1_pt.clone()
                    start_image_2_pt = outer_self.cur_image_2_pt.clone()

                # no transformation
                if transformation_index == 0:
                    temp_image_1_pt = start_image_1_pt.clone()
                    temp_image_2_pt = start_image_2_pt.clone()

                # 1: right diagonal flip (/)
                if transformation_index == 1:

                    temp_image_1_pt, temp_image_2_pt, _ = nero_transform.flip_piv_data(
                        start_image_1_pt, start_image_2_pt, np.zeros(1), flip_type='right-diagonal'
                    )

                # 2: counter-clockwise 90 rotation
                elif transformation_index == 2:
                    temp_image_1_pt, temp_image_2_pt, _ = nero_transform.rotate_piv_data(
                        start_image_1_pt, start_image_2_pt, np.zeros(1), 90
                    )
                # 3: horizontal flip (by y axis)
                elif transformation_index == 3:
                    temp_image_1_pt, temp_image_2_pt, _ = nero_transform.flip_piv_data(
                        start_image_1_pt, start_image_2_pt, np.zeros(1), flip_type='horizontal'
                    )
                # 4: counter-clockwise 180 rotation
                elif transformation_index == 4:
                    temp_image_1_pt, temp_image_2_pt, _ = nero_transform.rotate_piv_data(
                        start_image_1_pt, start_image_2_pt, np.zeros(1), 180
                    )
                # 5: \ diagnal flip
                elif transformation_index == 5:
                    temp_image_1_pt, temp_image_2_pt, _ = nero_transform.flip_piv_data(
                        start_image_1_pt, start_image_2_pt, np.zeros(1), flip_type='left-diagonal'
                    )
                # 6: counter-clockwise 270 rotation
                elif transformation_index == 6:
                    temp_image_1_pt, temp_image_2_pt, _ = nero_transform.rotate_piv_data(
                        start_image_1_pt, start_image_2_pt, np.zeros(1), 270
                    )
                # 7: vertical flip (by x axis)
                elif transformation_index == 7:
                    temp_image_1_pt, temp_image_2_pt, _ = nero_transform.flip_piv_data(
                        start_image_1_pt, start_image_2_pt, np.zeros(1), flip_type='vertical'
                    )

                # create new GIF
                display_image_1_pil = Image.fromarray(temp_image_1_pt.numpy(), 'RGB')
                display_image_2_pil = Image.fromarray(temp_image_2_pt.numpy(), 'RGB')
                other_images_pil = [
                    display_image_1_pil,
                    display_image_2_pil,
                    display_image_2_pil,
                    outer_self.blank_image_pil,
                ]
                outer_self.gif_path = os.path.join(outer_self.cache_dir, '_clicked.gif')
                display_image_1_pil.save(
                    fp=outer_self.gif_path,
                    format='GIF',
                    append_images=other_images_pil,
                    save_all=True,
                    duration=300,
                    loop=0,
                )

                # display the input image
                outer_self.display_image()

                # redraw the nero plot with new rectangle display
                outer_self.draw_piv_nero('single')

                # the detailed plot of PIV
                outer_self.detail_rect_x = outer_self.image_size // 2
                outer_self.detail_rect_y = outer_self.image_size // 2
                outer_self.draw_piv_details()

            def hoverEvent(self, event):
                if not event.isExit():
                    rect_x = int(event.pos().x() // outer_self.image_size)
                    rect_y = int(event.pos().y() // outer_self.image_size)

                    self.hover_text = outer_self.piv_nero_layout_names[rect_y][rect_x]
                    self.setToolTip(self.hover_text)

        # detail plot of the selected orbit position from individual nero plot
        # subclass of ImageItem that reimplements the control methods
        class PIV_detail_heatmap(pg.ImageItem):
            def mouseClickEvent(self, event):
                print(f'Clicked on detail heatmap at ({event.pos().x()}, {event.pos().y()})')
                outer_self.detail_rect_x = event.pos().x()
                outer_self.detail_rect_y = event.pos().y()

                # udpate the detail plot
                outer_self.draw_piv_nero(mode='single')
                # draw the quiver plot
                outer_self.draw_piv_details()

            def mouseDragEvent(self, event):
                if self.plot_type == 'single':
                    # if event.button() != QtCore.Qt.LeftButton:
                    #     event.ignore()
                    #     return
                    # print(event.pos())
                    if event.isStart():
                        print('Dragging starts', event.pos())

                    elif event.isFinish():
                        print('Dragging stops', event.pos())

                    else:
                        print('Drag', event.pos())

        # helper function on reshaping data
        def prepare_plot_data(input_data):
            grid_size = input_data.shape[1]
            # extra pixels are for lines
            output_data = np.zeros((4 * grid_size, 4 * grid_size))
            # compose plot data into the big rectangle that is consist of
            # 16 rectangles where each contains a heatmap of the error heatmap
            # the layout is
            """
            2'  2   1   1'
            3'  3   0   0'
            4'  4   7   7'
            5'  5   6   6'
            """
            # where the index is the same as in data
            # the meaning of 0 to 7 could be found at lines 316-333
            # first row
            output_data[0 * grid_size : 1 * grid_size, 0 * grid_size : 1 * grid_size] = input_data[
                8 + 2
            ]
            output_data[0 * grid_size : 1 * grid_size, 1 * grid_size : 2 * grid_size] = input_data[
                0 + 2
            ]
            output_data[0 * grid_size : 1 * grid_size, 2 * grid_size : 3 * grid_size] = input_data[
                0 + 1
            ]
            output_data[0 * grid_size : 1 * grid_size, 3 * grid_size : 4 * grid_size] = input_data[
                8 + 1
            ]

            # second row
            output_data[1 * grid_size : 2 * grid_size, 0 * grid_size : 1 * grid_size] = input_data[
                8 + 3
            ]
            output_data[1 * grid_size : 2 * grid_size, 1 * grid_size : 2 * grid_size] = input_data[
                0 + 3
            ]
            output_data[1 * grid_size : 2 * grid_size, 2 * grid_size : 3 * grid_size] = input_data[
                0 + 0
            ]
            output_data[1 * grid_size : 2 * grid_size, 3 * grid_size : 4 * grid_size] = input_data[
                8 + 0
            ]

            # third row
            output_data[2 * grid_size : 3 * grid_size, 0 * grid_size : 1 * grid_size] = input_data[
                8 + 4
            ]
            output_data[2 * grid_size : 3 * grid_size, 1 * grid_size : 2 * grid_size] = input_data[
                0 + 4
            ]
            output_data[2 * grid_size : 3 * grid_size, 2 * grid_size : 3 * grid_size] = input_data[
                0 + 7
            ]
            output_data[2 * grid_size : 3 * grid_size, 3 * grid_size : 4 * grid_size] = input_data[
                8 + 7
            ]

            # fourth row
            output_data[3 * grid_size : 4 * grid_size, 0 * grid_size : 1 * grid_size] = input_data[
                8 + 5
            ]
            output_data[3 * grid_size : 4 * grid_size, 1 * grid_size : 2 * grid_size] = input_data[
                0 + 5
            ]
            output_data[3 * grid_size : 4 * grid_size, 2 * grid_size : 3 * grid_size] = input_data[
                0 + 6
            ]
            output_data[3 * grid_size : 4 * grid_size, 3 * grid_size : 4 * grid_size] = input_data[
                8 + 6
            ]

            return output_data

        # add to general layout
        if mode == 'single':
            self.single_result_existed = True
            # prepare data for piv individual nero plot (heatmap)
            self.data_1 = prepare_plot_data(self.cur_single_plot_quantity_1)
            self.data_2 = prepare_plot_data(self.cur_single_plot_quantity_2)
            # heatmap view
            self.heatmap_view_1 = pg.GraphicsLayoutWidget()
            # left top right bottom
            self.heatmap_view_1.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.heatmap_view_1.setFixedSize(self.plot_size * 1.2, self.plot_size * 1.2)
            self.heatmap_view_2 = pg.GraphicsLayoutWidget()
            # left top right bottom
            self.heatmap_view_2.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.heatmap_view_2.setFixedSize(self.plot_size * 1.2, self.plot_size * 1.2)

            self.single_nero_1 = PIV_heatmap()
            self.single_nero_2 = PIV_heatmap()
            self.scatter_item_1 = pg.ScatterPlotItem(pxMode=False)
            self.scatter_item_1.setSymbol('s')
            self.scatter_item_2 = pg.ScatterPlotItem(pxMode=False)
            self.scatter_item_2.setSymbol('s')
            # color map is flipped so that low error is bright
            self.cm_range = (self.loss_high_bound, self.loss_low_bound)
            self.heatmap_plot_1 = self.draw_individual_heatmap(
                'single', self.data_1, self.single_nero_1, self.scatter_item_1
            )

            self.heatmap_plot_2 = self.draw_individual_heatmap(
                'single', self.data_2, self.single_nero_2, self.scatter_item_2
            )

            # add to view
            self.heatmap_view_1.addItem(self.heatmap_plot_1)
            self.heatmap_view_2.addItem(self.heatmap_plot_2)

            # add to layout
            if self.data_mode == 'single':
                self.single_result_layout.addWidget(self.heatmap_view_1, 1, 1)
                self.single_result_layout.addWidget(self.heatmap_view_2, 1, 2)
            elif self.data_mode == 'aggregate':
                if self.demo:
                    self.demo_layout.addWidget(self.heatmap_view_1, 3, 2, 1, 1)
                    self.demo_layout.addWidget(self.heatmap_view_2, 5, 2, 1, 1)
                else:
                    self.aggregate_result_layout.addWidget(self.heatmap_view_1, 1, 4)
                    self.aggregate_result_layout.addWidget(self.heatmap_view_2, 1, 5)

            # detail (selected from individual NERO plot) heatmap
            # heatmap view
            self.detail_heatmap_view_1 = pg.GraphicsLayoutWidget()
            self.detail_heatmap_view_1.ci.layout.setContentsMargins(
                0, 0, 0, 0
            )  # left top right bottom
            self.detail_heatmap_view_1.setFixedSize(self.plot_size * 1.2, self.plot_size * 1.2)
            self.detail_heatmap_view_2 = pg.GraphicsLayoutWidget()
            self.detail_heatmap_view_2.ci.layout.setContentsMargins(
                0, 0, 0, 0
            )  # left top right bottom
            self.detail_heatmap_view_2.setFixedSize(self.plot_size * 1.2, self.plot_size * 1.2)

            # heatmap plot
            self.detail_heatmap_1 = PIV_detail_heatmap()
            self.detail_heatmap_2 = PIV_detail_heatmap()
            # the data is based on selection in individual NERO plot
            self.detail_data_1 = self.data_1[
                self.rect_index_y * self.image_size : (self.rect_index_y + 1) * self.image_size,
                self.rect_index_x * self.image_size : (self.rect_index_x + 1) * self.image_size,
            ]
            self.detail_data_2 = self.data_2[
                self.rect_index_y * self.image_size : (self.rect_index_y + 1) * self.image_size,
                self.rect_index_x * self.image_size : (self.rect_index_x + 1) * self.image_size,
            ]
            # color map is flipped so that low error is bright
            self.detail_plot_1 = self.draw_piv_detail_heatmap(
                self.detail_data_1, self.detail_heatmap_1
            )
            self.detail_plot_2 = self.draw_piv_detail_heatmap(
                self.detail_data_2, self.detail_heatmap_2
            )

            # add to view
            self.detail_heatmap_view_1.addItem(self.detail_plot_1)
            self.detail_heatmap_view_2.addItem(self.detail_plot_2)

            # add to layout
            if self.data_mode == 'single':
                self.single_result_layout.addWidget(self.detail_heatmap_view_1, 1, 1)
                self.single_result_layout.addWidget(self.detail_heatmap_view_2, 1, 2)
            elif self.data_mode == 'aggregate':
                if self.demo:
                    self.demo_layout.addWidget(self.detail_heatmap_view_1, 3, 3, 1, 1)
                    self.demo_layout.addWidget(self.detail_heatmap_view_2, 5, 3, 1, 1)
                else:
                    self.aggregate_result_layout.addWidget(self.detail_heatmap_view_1, 1, 4)
                    self.aggregate_result_layout.addWidget(self.detail_heatmap_view_2, 1, 5)

        elif mode == 'aggregate':
            # prepare data for piv individual nero plot (heatmap)
            self.data_1 = prepare_plot_data(self.cur_aggregate_plot_quantity_1)
            self.data_2 = prepare_plot_data(self.cur_aggregate_plot_quantity_2)
            # heatmap view
            self.aggregate_heatmap_view_1 = pg.GraphicsLayoutWidget()
            # left top right bottom
            self.aggregate_heatmap_view_1.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.aggregate_heatmap_view_1.setFixedSize(self.plot_size * 1.2, self.plot_size * 1.2)
            self.aggregate_heatmap_view_2 = pg.GraphicsLayoutWidget()
            # left top right bottom
            self.aggregate_heatmap_view_2.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.aggregate_heatmap_view_2.setFixedSize(self.plot_size * 1.2, self.plot_size * 1.2)
            # color map is flipped so that low error is bright
            self.cm_range = (self.loss_high_bound, self.loss_low_bound)
            self.aggregate_heatmap_plot_1 = self.draw_individual_heatmap('aggregate', self.data_1)
            self.aggregate_heatmap_plot_2 = self.draw_individual_heatmap('aggregate', self.data_2)

            # add to view
            self.aggregate_heatmap_view_1.addItem(self.aggregate_heatmap_plot_1)
            self.aggregate_heatmap_view_2.addItem(self.aggregate_heatmap_plot_2)

            if self.demo:
                self.demo_layout.addWidget(self.aggregate_heatmap_view_1, 3, 0, 1, 1)
                self.demo_layout.addWidget(self.aggregate_heatmap_view_2, 5, 0, 1, 1)
            else:
                self.aggregate_result_layout.addWidget(self.aggregate_heatmap_view_1, 1, 1)
                self.aggregate_result_layout.addWidget(self.aggregate_heatmap_view_2, 1, 2)

    # helper function that draws the quiver plot with input vector fields
    def draw_quiver_plot(
        self, ground_truth_vectors, pred_vectors, gt_color, pred_color, title=None
    ):
        # test qiv stuff
        # ground_truth_vectors_np = ground_truth_vectors.numpy()
        # print(ground_truth_vectors_np[:4, :4, 0].flags)
        # print(ground_truth_vectors_np[:4, :4, 0].shape)
        # print(ground_truth_vectors_np[:4, :4, 0])
        # ground_truth_vectors_qiv = qiv.from_numpy(
        #     np.ascontiguousarray(ground_truth_vectors_np[:4, :4, 0])
        # )
        # qiv.qivArraySave(f'ground_truth_vectors.txt'.encode('utf-8'), ground_truth_vectors_qiv)
        # exit()

        class MyArrowItem(pg.ArrowItem):
            def paint(self, p, *args):
                p.translate(-self.boundingRect().center() * 2)
                pg.ArrowItem.paint(self, p, *args)

        quiver_plot = pg.PlotItem(title=title)
        # so that showing indicator at the boundary does not jitter the plot
        quiver_plot.vb.disableAutoRange()
        quiver_plot.setXRange(-1, len(ground_truth_vectors) + 1, padding=0)
        quiver_plot.setYRange(-1, len(ground_truth_vectors) + 1, padding=0)
        quiver_plot.hideAxis('bottom')
        quiver_plot.hideAxis('left')
        # Not letting user zoom out past axis limit
        quiver_plot.vb.setLimits(
            xMin=-1,
            xMax=len(ground_truth_vectors) + 1,
            yMin=-1,
            yMax=len(ground_truth_vectors) + 1,
        )

        # largest and smallest in ground truth
        v_max = np.max(
            np.sqrt(
                np.power(ground_truth_vectors[:, :, 0].numpy(), 2)
                + np.power(ground_truth_vectors[:, :, 1].numpy(), 2)
            )
        )

        # all the ground truth vectors
        for y in range(len(ground_truth_vectors)):
            for x in range(len(ground_truth_vectors[y])):
                # ground truth vector
                cur_gt_vector = ground_truth_vectors[y, x]
                # convert to polar coordinate
                r_gt = np.sqrt((cur_gt_vector[0] ** 2 + cur_gt_vector[1] ** 2))
                theta_gt = np.arctan2(cur_gt_vector[1], cur_gt_vector[0]) / np.pi * 180
                # creat ground truth arrow
                # arrow item has 0 degree set as to left
                cur_arrow_gt = MyArrowItem(
                    pxMode=True,
                    angle=180 + theta_gt,
                    headLen=20 * r_gt / v_max,
                    tailLen=20 * r_gt / v_max,
                    tipAngle=40,
                    baseAngle=0,
                    tailWidth=5,
                    pen=QtGui.QPen(gt_color),
                    brush=QtGui.QBrush(gt_color),
                )

                # coordinate in y are flipped for later be used in image
                cur_arrow_gt.setPos(x, len(ground_truth_vectors) - 1 - y)
                quiver_plot.addItem(cur_arrow_gt)

                # model predicted vector
                cur_pred_vector = pred_vectors[y, x]
                # convert to polar coordinate
                r_pred = np.sqrt((cur_pred_vector[0] ** 2 + cur_pred_vector[1] ** 2))
                # print(f'y={y}, x={x}, r_gt={r_gt}, r_pred={r_pred}')
                theta_pred = np.arctan2(cur_pred_vector[1], cur_pred_vector[0]) / np.pi * 180
                # creat ground truth arrow
                cur_arrow_pred = MyArrowItem(
                    pxMode=True,
                    angle=180 + theta_pred,
                    headLen=20 * r_pred / v_max,
                    tailLen=20 * r_pred / v_max,
                    tipAngle=40,
                    baseAngle=0,
                    tailWidth=5,
                    pen=QtGui.QPen(pred_color),
                    brush=QtGui.QBrush(pred_color),
                )

                # coordinate in y are flipped for later be used in image
                cur_arrow_pred.setPos(x, len(ground_truth_vectors) - 1 - y)
                quiver_plot.addItem(cur_arrow_pred)

        quiver_plot.getAxis('bottom').setStyle(tickLength=0, showValues=False)
        quiver_plot.getAxis('left').setStyle(tickLength=0, showValues=False)

        return quiver_plot

    # draw quiver plot between PIV ground truth and model predictions
    def draw_piv_details(self):

        # get the selected vector data from ground turth and models' outputs
        # local position within this transformation
        detail_rect_x_local = int(self.detail_rect_x)
        detail_rect_y_local = int(self.detail_rect_y)

        # vector field around the selected center
        detail_ground_truth = self.all_ground_truths[self.rectangle_index][
            detail_rect_y_local - 4 : detail_rect_y_local + 4,
            detail_rect_x_local - 4 : detail_rect_x_local + 4,
        ]

        detail_vectors_1 = self.all_quantities_1[self.rectangle_index][
            detail_rect_y_local - 4 : detail_rect_y_local + 4,
            detail_rect_x_local - 4 : detail_rect_x_local + 4,
        ]

        detail_vectors_2 = self.all_quantities_2[self.rectangle_index][
            detail_rect_y_local - 4 : detail_rect_y_local + 4,
            detail_rect_x_local - 4 : detail_rect_x_local + 4,
        ]

        # save vector fields for glk
        # qiv.qivArraySave(
        #     f'ground_truth.nrrd'.encode('utf-8'),
        #     qiv.from_numpy(self.all_ground_truths[self.rectangle_index].numpy()),
        # )
        # qiv.qivArraySave(
        #     f'ml_pred.nrrd'.encode('utf-8'),
        #     qiv.from_numpy(self.all_quantities_1[self.rectangle_index].numpy()),
        # )
        # qiv.qivArraySave(
        #     f'farneback_pred.nrrd'.encode('utf-8'),
        #     qiv.from_numpy(self.all_quantities_2[self.rectangle_index].numpy()),
        # )

        # view 1
        self.piv_detail_view_1 = pg.GraphicsLayoutWidget()
        self.piv_detail_view_1.ci.layout.setContentsMargins(0, 0, 0, 0)  # left top right bottom
        self.piv_detail_view_1.setFixedSize(self.plot_size * 1.4, self.plot_size * 1.4)

        # view 2
        self.piv_detail_view_2 = pg.GraphicsLayoutWidget()
        self.piv_detail_view_2.ci.layout.setContentsMargins(0, 0, 0, 0)  # left top right bottom
        self.piv_detail_view_2.setFixedSize(self.plot_size * 1.4, self.plot_size * 1.4)

        # plot both quiver plots
        gt_color = QtGui.QColor('black')
        gt_color.setAlpha(128)
        model_1_color = QtGui.QColor('blue')
        model_1_color.setAlpha(128)
        model_2_color = QtGui.QColor('magenta')
        model_2_color.setAlpha(128)
        self.piv_detail_plot_1 = self.draw_quiver_plot(
            detail_ground_truth, detail_vectors_1, gt_color, model_1_color
        )

        self.piv_detail_plot_2 = self.draw_quiver_plot(
            detail_ground_truth, detail_vectors_2, gt_color, model_2_color
        )

        # add to view
        self.piv_detail_view_1.addItem(self.piv_detail_plot_1)
        self.piv_detail_view_2.addItem(self.piv_detail_plot_2)

        # add view to general layout
        if self.data_mode == 'single':
            self.single_result_layout.addWidget(self.piv_detail_view_1, 2, 1)
            self.single_result_layout.addWidget(self.piv_detail_view_2, 2, 2)
        elif self.data_mode == 'aggregate':
            if self.demo:
                self.demo_layout.addWidget(self.piv_detail_view_1, 2, 4, 3, 1)
                self.demo_layout.addWidget(self.piv_detail_view_2, 4, 4, 3, 1)
            else:
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
        if not self.demo:
            self.aggregate_result_layout.addWidget(
                self.model_1_menu, 0, 1, 1, 1, QtCore.Qt.AlignCenter
            )
            self.aggregate_result_layout.addWidget(
                self.model_2_menu, 0, 2, 1, 1, QtCore.Qt.AlignCenter
            )

        # move run button in the first column (after aggregate heatmap control) in non-demo mode
        if not self.demo:
            self.aggregate_plot_control_layout.addWidget(self.run_button, 4, 0)
            self.aggregate_plot_control_layout.addWidget(self.use_cache_checkbox, 5, 0)
        else:
            self.run_button_layout.removeWidget(self.run_button)
            self.run_button_layout.removeWidget(self.use_cache_checkbox)

        self.aggregate_result_existed = True

        # drop down menu on selection which quantity to plot
        # title
        # draw text
        plot_quantity_pixmap = QPixmap(300, 50)
        plot_quantity_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(plot_quantity_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 300, 50, QtGui.Qt.AlignLeft, 'NERO Metric: ')
        painter.end()

        # create label to contain the texts
        self.plot_quantity_label = QLabel(self)
        self.plot_quantity_label.setFixedSize(QtCore.QSize(300, 50))
        self.plot_quantity_label.setPixmap(plot_quantity_pixmap)
        self.plot_quantity_label.setContentsMargins(0, 0, 0, 0)

        # drop down menu on selection which quantity to plot
        quantity_menu = QtWidgets.QComboBox()
        quantity_menu.setFixedSize(QtCore.QSize(220, 50))
        quantity_menu.setStyleSheet(
            'color: black; font-family: Helvetica; font-style: normal; font-size: 34px'
        )
        quantity_menu.setEditable(True)
        quantity_menu.lineEdit().setReadOnly(True)
        quantity_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)

        quantity_menu.addItem('Accuracy')  # for aggregate case only
        quantity_menu.addItem('Confidence')
        quantity_menu.setCurrentText('Confidence')
        self.quantity_name = quantity_menu.currentText()

        # connect the drop down menu with actions
        quantity_menu.currentTextChanged.connect(polar_quantity_changed)
        if self.demo:
            self.plot_info_layout = QtWidgets.QHBoxLayout()
            self.plot_info_layout.addWidget(self.plot_quantity_label)
            self.plot_info_layout.addWidget(quantity_menu)
            self.plot_info_layout.setContentsMargins(20, 0, 0, 0)
            self.demo_layout.addLayout(self.plot_info_layout, 0, 2)
        else:
            self.aggregate_plot_control_layout.addWidget(quantity_menu, 1, 0)

        # draw the aggregate polar plot
        self.draw_aggregate_polar()

    # display MNIST single results
    def display_mnist_single_result(self, type):

        self.single_result_existed = True

        # draw result using bar plot
        if type == 'bar':
            self.bar_plot = pg.plot()
            # constrain plot showing limit by setting view box
            self.bar_plot.plotItem.vb.setLimits(xMin=-0.5, xMax=9.5, yMin=0, yMax=1.2)
            self.bar_plot.setBackground('w')
            self.bar_plot.setFixedSize(self.plot_size * 1.7, self.plot_size * 1.7)
            self.bar_plot.setContentsMargins(20, 0, 0, 0)
            x_label_style = {'color': 'black', 'font-size': '16pt', 'text': 'Digit'}
            y_label_style = {'color': 'black', 'font-size': '16pt', 'text': 'Confidence'}
            self.bar_plot.getAxis('bottom').setLabel(**x_label_style)
            self.bar_plot.getAxis('left').setLabel(**y_label_style)
            tick_font = QFont('Helvetica', 16)
            self.bar_plot.getAxis('bottom').setTickFont(tick_font)
            self.bar_plot.getAxis('left').setTickFont(tick_font)
            self.bar_plot.getAxis('bottom').setTextPen('black')
            self.bar_plot.getAxis('left').setTextPen('black')

            graph_1 = pg.BarGraphItem(
                x=np.arange(len(self.output_1)) - 0.2,
                height=list(self.output_1),
                width=0.4,
                brush='blue',
            )
            graph_2 = pg.BarGraphItem(
                x=np.arange(len(self.output_1)) + 0.2,
                height=list(self.output_2),
                width=0.4,
                brush='magenta',
            )
            self.bar_plot.addItem(graph_1)
            self.bar_plot.addItem(graph_2)
            # disable moving around
            self.bar_plot.setMouseEnabled(x=False, y=False)
            if self.data_mode == 'single':
                self.single_result_layout.addWidget(self.bar_plot, 1, 2)
            elif self.data_mode == 'aggregate':
                if self.demo:
                    self.demo_layout.addWidget(self.bar_plot, 2, 3, 5, 1)
                else:
                    self.aggregate_result_layout.addWidget(self.bar_plot, 2, 6)

        elif type == 'polar':

            # helper function for clicking inside demension reduced scatter plot
            # def clicked(item, points):
            #     # clear manual mode line
            #     if self.cur_line:
            #         self.polar_plot.removeItem(self.cur_line)

            #     # clear previously selected point's visual cue
            #     if self.last_clicked:
            #         self.last_clicked.resetPen()
            #         self.last_clicked.setBrush(self.old_brush)

            #     # clicked point's position
            #     x_pos = points[0].pos().x()
            #     y_pos = points[0].pos().y()

            #     # convert back to polar coordinate
            #     radius = np.sqrt(x_pos**2 + y_pos**2)
            #     self.cur_rotation_angle = np.arctan2(y_pos, x_pos) / np.pi * 180

            #     # update the current image's angle and rotate the display image
            #     # rotate the image tensor
            #     self.cur_image_pt = nero_transform.rotate_mnist_image(
            #         self.loaded_image_pt, self.cur_rotation_angle
            #     )
            #     # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(angle), QtCore.Qt.SmoothTransformation)
            #     # convert image tensor to qt image and resize for display
            #     self.cur_display_image = nero_utilities.tensor_to_qt_image(
            #         self.cur_image_pt, self.display_image_size
            #     )
            #     # prepare image tensor for model purpose
            #     self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)
            #     # update the pixmap and label
            #     self.image_pixmap = QPixmap(self.cur_display_image)
            #     self.image_label.setPixmap(self.image_pixmap)

            #     # update the model output
            #     if self.single_result_existed:
            #         self.run_model_once()

            #     # only allow clicking one point at a time
            #     # save the old brush
            #     if points[0].brush() == pg.mkBrush(QtGui.QColor("blue")):
            #         self.old_brush = pg.mkBrush(QtGui.QColor("blue"))

            #     elif points[0].brush() == pg.mkBrush(QtGui.QColor("magenta")):
            #         self.old_brush = pg.mkBrush(QtGui.QColor("magenta"))

            #     # create new brush
            #     new_brush = pg.mkBrush(255, 0, 0, 255)
            #     points[0].setBrush(new_brush)
            #     points[0].setPen(5)

            #     self.last_clicked = points[0]

            # initialize view and plot
            polar_view = pg.GraphicsLayoutWidget()
            polar_view.setBackground('white')
            polar_view.setFixedSize(self.plot_size * 1.7, self.plot_size * 1.7)
            polar_view.ci.layout.setContentsMargins(0, 0, 50, 0)
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
                all_points_1.append(
                    {
                        'pos': (x_1, y_1),
                        'size': 0.05,
                        'pen': {'color': 'w', 'width': 0.1},
                        'brush': QtGui.QColor('blue'),
                    }
                )

                # model 2 quantity
                cur_quantity_2 = self.all_quantities_2[i]
                # Transform to cartesian and plot
                x_2 = cur_quantity_2 * np.cos(radian)
                y_2 = cur_quantity_2 * np.sin(radian)
                all_x_2.append(x_2)
                all_y_2.append(y_2)
                all_points_2.append(
                    {
                        'pos': (x_2, y_2),
                        'size': 0.05,
                        'pen': {'color': 'w', 'width': 0.1},
                        'brush': QtGui.QColor('magenta'),
                    }
                )

            # draw lines to better show shape
            self.polar_plot.plot(all_x_1, all_y_1, pen=QtGui.QPen(QtGui.Qt.blue, 0.03))
            self.polar_plot.plot(all_x_2, all_y_2, pen=QtGui.QPen(QtGui.QColor('magenta'), 0.03))

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
                    self.cur_image_pt = nero_transform.rotate_mnist_image(
                        self.loaded_image_pt, self.cur_rotation_angle
                    )
                    # convert image tensor to qt image and resize for display
                    self.cur_display_image = nero_utilities.tensor_to_qt_image(
                        self.cur_image_pt, self.display_image_size, revert_color=True
                    )
                    # update the pixmap and label
                    self.image_pixmap = QPixmap(self.cur_display_image)
                    self.image_label.setPixmap(self.image_pixmap)
                    # prepare image tensor for model purpose
                    self.cur_image_pt = nero_transform.prepare_mnist_image(self.cur_image_pt)

                    # update the model output
                    if self.single_result_existed:
                        self.run_model_once()

                    # remove old line and circle
                    if self.cur_line:
                        self.polar_plot.removeItem(self.cur_line)
                        self.polar_plot.removeItem(self.circle_1)
                        self.polar_plot.removeItem(self.circle_2)

                    # draw a line that represents current angle of rotation
                    cur_x = 1 * np.cos(self.cur_rotation_angle / 180 * np.pi)
                    cur_y = 1 * np.sin(self.cur_rotation_angle / 180 * np.pi)
                    line_x = [0, cur_x]
                    line_y = [0, cur_y]
                    self.cur_line = self.polar_plot.plot(
                        line_x, line_y, pen=QtGui.QPen(QtGui.Qt.green, 0.02)
                    )

                    # display current results on the line
                    self.draw_circle_on_polar()

            self.polar_clicked = False
            self.polar_plot.scene().sigMouseClicked.connect(polar_mouse_clicked)
            self.polar_plot.scene().sigMouseMoved.connect(polar_mouse_moved)

            # fix zoom level
            self.polar_plot.setMouseEnabled(x=False, y=False)

            # add the plot view to the layout
            if self.data_mode == 'single':
                self.single_result_layout.addWidget(polar_view, 1, 3)
            elif self.data_mode == 'aggregate':
                if self.demo:
                    self.demo_layout.addWidget(polar_view, 2, 2, 5, 1)
                else:
                    self.aggregate_result_layout.addWidget(polar_view, 1, 6)

        else:
            raise Exception('Unsupported display mode')

    # function that computes consensus among different experiments
    def compute_consensus(self, model_outputs):
        # consensus for object detection is consist of averaged bounding boxes centers, widths and heights, each image will have a consensus as an estimation for ground truth
        if self.mode == 'object_detection':

            # initialize outputs
            aggregate_consensus = np.zeros((len(self.all_images_paths), 5))

            # each image has one consensus as an approximate for ground truth
            # consensus has layout [num_images, x1, y1, x2, y2, class_label]
            # compute consensus by avareging best model outputs
            # highest confidence output is at the top of aggregate_outputs_1
            for i in range(len(self.all_images_paths)):
                # model output layout: x1, y1, x2, y2, conf, class_pred, iou, label correctness
                x1_sum = 0
                y1_sum = 0
                x2_sum = 0
                y2_sum = 0
                label_sum = 0
                for y, y_tran in enumerate(self.y_translation):
                    for x, x_tran in enumerate(self.x_translation):
                        x1_sum += model_outputs[y, x][i][0, 0] - x_tran
                        y1_sum += model_outputs[y, x][i][0, 1] - y_tran
                        x2_sum += model_outputs[y, x][i][0, 2] - x_tran
                        y2_sum += model_outputs[y, x][i][0, 3] - y_tran
                        label_sum += model_outputs[y, x][i][0, 5]

                aggregate_consensus[i] = [
                    x1_sum / (model_outputs.shape[0] * model_outputs.shape[1]),
                    y1_sum / (model_outputs.shape[0] * model_outputs.shape[1]),
                    x2_sum / (model_outputs.shape[0] * model_outputs.shape[1]),
                    y2_sum / (model_outputs.shape[0] * model_outputs.shape[1]),
                    int(label_sum / (model_outputs.shape[0] * model_outputs.shape[1])),
                ]

            return aggregate_consensus

        elif self.mode == 'piv':
            raise NotImplementedError

    def compute_consensus_losses(self, model_outputs, consensus):

        if self.mode == 'object_detection':
            consensus_outputs = np.zeros(
                (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
            )
            consensus_precision = np.zeros(
                (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
            )
            consensus_recall = np.zeros(
                (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
            )
            consensus_F_measure = np.zeros(
                (len(self.y_translation), len(self.x_translation)), dtype=np.ndarray
            )
            consensus_mAP = np.zeros((len(self.y_translation), len(self.x_translation)))

            # compute losses using consensus
            for y, y_tran in enumerate(self.y_translation):
                for x, x_tran in enumerate(self.x_translation):
                    print(f'y_tran = {y_tran}, x_tran = {x_tran}')
                    # current model outputs after shifting back
                    cur_model_outputs = model_outputs[y, x]

                    # transform unshifted consensus for current model outputs
                    cur_consensus = consensus.copy()
                    cur_consensus[0] = consensus[0] - x_tran
                    cur_consensus[1] = consensus[1] - y_tran
                    cur_consensus[2] = consensus[2] - x_tran
                    cur_consensus[3] = consensus[3] - y_tran
                    # cur_consensus = np.zeros(consensus.shape)
                    # cur_consensus[:, 4] = consensus[:, 4]
                    # for i in range(len(consensus)):
                    #     cur_consensus[i, :4] = self.update_consensus(
                    #         consensus[i, :4], (self.image_size, self.image_size), x_tran, y_tran
                    #     )

                    # compute losses using consensus
                    (
                        cur_augmented_outputs,
                        cur_consensus_precision,
                        cur_consensus_recall,
                        cur_consensus_F_measure,
                    ) = nero_run_model.evaluate_coco_with_consensus(
                        cur_model_outputs, cur_consensus
                    )

                    # record
                    consensus_outputs[y, x] = cur_augmented_outputs
                    consensus_precision[y, x] = cur_consensus_precision
                    consensus_recall[y, x] = cur_consensus_recall
                    # compute mAP from precision and recall
                    consensus_mAP[y, x] = nero_utilities.compute_ap(
                        cur_consensus_recall, cur_consensus_precision
                    )
                    consensus_F_measure[y, x] = cur_consensus_F_measure

            return (
                consensus_outputs,
                consensus_precision,
                consensus_recall,
                consensus_mAP,
                consensus_F_measure,
            )

        elif self.mode == 'piv':
            raise NotImplementedError

    # display COCO aggregate results
    def display_coco_aggregate_result(self):

        if not self.demo:
            # move the model menu on top of the each aggregate NERO plot
            self.aggregate_result_layout.addWidget(
                self.model_1_menu, 0, 1, 1, 1, QtCore.Qt.AlignCenter
            )
            self.aggregate_result_layout.addWidget(
                self.model_2_menu, 0, 2, 1, 1, QtCore.Qt.AlignCenter
            )
            # move run button in the first column (after aggregate heatmap control)
            self.aggregate_plot_control_layout.addWidget(self.run_button, 4, 0)
            self.aggregate_plot_control_layout.addWidget(self.use_cache_checkbox, 5, 0)

        self.aggregate_result_existed = True

        @QtCore.Slot()
        def coco_nero_quantity_changed(text):
            print('Plotting:', text, 'on aggregate NERO plot')
            self.quantity_name = text
            if text == 'Conf*IOU':
                self.cur_aggregate_plot_quantity_1 = (
                    self.aggregate_avg_conf_correctness_1 * self.aggregate_avg_iou_correctness_1
                )
                self.cur_aggregate_plot_quantity_2 = (
                    self.aggregate_avg_conf_correctness_2 * self.aggregate_avg_iou_correctness_2
                )
            elif text == 'Confidence':
                self.cur_aggregate_plot_quantity_1 = self.aggregate_avg_conf_1
                self.cur_aggregate_plot_quantity_2 = self.aggregate_avg_conf_2
            elif text == 'IOU':
                self.cur_aggregate_plot_quantity_1 = self.aggregate_avg_iou_correctness_1
                self.cur_aggregate_plot_quantity_2 = self.aggregate_avg_iou_correctness_2

            # below quantities won't show in the demo mode
            elif text == 'Precision':
                self.cur_aggregate_plot_quantity_1 = self.aggregate_avg_precision_1
                self.cur_aggregate_plot_quantity_2 = self.aggregate_avg_precision_2
            elif text == 'Recall':
                self.cur_aggregate_plot_quantity_1 = self.aggregate_avg_recall_1
                self.cur_aggregate_plot_quantity_2 = self.aggregate_avg_recall_2
            elif text == 'F1 score':
                self.cur_aggregate_plot_quantity_1 = self.aggregate_avg_F_measure_1
                self.cur_aggregate_plot_quantity_2 = self.aggregate_avg_F_measure_2
            elif text == 'AP':
                self.cur_aggregate_plot_quantity_1 = self.aggregate_mAP_1
                self.cur_aggregate_plot_quantity_2 = self.aggregate_mAP_2

            # re-display the heatmap
            self.draw_coco_nero(mode='aggregate')

            # re-run dimension reduction and show result
            if self.dr_result_existed:
                self.run_dimension_reduction()

            # if available, update single NERO plot as well
            if self.single_result_existed:
                print('Plotting:', text, 'on single NERO plot')
                self.quantity_name = text

                if text == 'Conf*IOU':
                    if self.data_mode == 'single':
                        self.cur_single_plot_quantity_1 = (
                            self.all_quantities_1[:, :, 4] * self.all_quantities_1[:, :, 6]
                        )
                        self.cur_single_plot_quantity_2 = (
                            self.all_quantities_2[:, :, 4] * self.all_quantities_2[:, :, 6]
                        )
                    elif self.data_mode == 'aggregate':
                        # current selected individual images' result on all transformations
                        for y in range(len(self.y_translation)):
                            for x in range(len(self.x_translation)):
                                self.cur_single_plot_quantity_1[y, x] = (
                                    self.aggregate_outputs_1[y, x][self.image_index][0, 4]
                                    * self.aggregate_outputs_1[y, x][self.image_index][0, 6]
                                )
                                self.cur_single_plot_quantity_2[y, x] = (
                                    self.aggregate_outputs_2[y, x][self.image_index][0, 4]
                                    * self.aggregate_outputs_2[y, x][self.image_index][0, 6]
                                )

                if text == 'Conf*IOU*Correctness':
                    if self.data_mode == 'single':
                        self.cur_single_plot_quantity_1 = (
                            self.all_quantities_1[:, :, 4]
                            * self.all_quantities_1[:, :, 6]
                            * self.all_quantities_1[:, :, 7]
                        )
                        self.cur_single_plot_quantity_2 = (
                            self.all_quantities_2[:, :, 4]
                            * self.all_quantities_2[:, :, 6]
                            * self.all_quantities_2[:, :, 7]
                        )
                    elif self.data_mode == 'aggregate':
                        # current selected individual images' result on all transformations
                        for y in range(len(self.y_translation)):
                            for x in range(len(self.x_translation)):
                                self.cur_single_plot_quantity_1[y, x] = (
                                    self.aggregate_outputs_1[y, x][self.image_index][0, 4]
                                    * self.aggregate_outputs_1[y, x][self.image_index][0, 6]
                                    * self.aggregate_outputs_1[y, x][self.image_index][0, 7]
                                )
                                self.cur_single_plot_quantity_2[y, x] = (
                                    self.aggregate_outputs_2[y, x][self.image_index][0, 4]
                                    * self.aggregate_outputs_2[y, x][self.image_index][0, 6]
                                    * self.aggregate_outputs_2[y, x][self.image_index][0, 7]
                                )

                elif text == 'Confidence':
                    if self.data_mode == 'single':
                        self.cur_single_plot_quantity_1 = self.all_quantities_1[:, :, 4]
                        self.cur_single_plot_quantity_2 = self.all_quantities_2[:, :, 4]
                    elif self.data_mode == 'aggregate':
                        # current selected individual images' result on all transformations
                        for y in range(len(self.y_translation)):
                            for x in range(len(self.x_translation)):
                                self.cur_single_plot_quantity_1[y, x] = self.aggregate_outputs_1[
                                    y, x
                                ][self.image_index][0, 4]
                                self.cur_single_plot_quantity_2[y, x] = self.aggregate_outputs_2[
                                    y, x
                                ][self.image_index][0, 4]

                elif text == 'IOU':
                    if self.data_mode == 'single':
                        self.cur_single_plot_quantity_1 = self.all_quantities_1[:, :, 6]
                        self.cur_single_plot_quantity_2 = self.all_quantities_2[:, :, 6]
                    elif self.data_mode == 'aggregate':
                        # current selected individual images' result on all transformations
                        for y in range(len(self.y_translation)):
                            for x in range(len(self.x_translation)):
                                self.cur_single_plot_quantity_1[y, x] = self.aggregate_outputs_1[
                                    y, x
                                ][self.image_index][0, 6]
                                self.cur_single_plot_quantity_2[y, x] = self.aggregate_outputs_2[
                                    y, x
                                ][self.image_index][0, 6]

                # re-display the heatmap
                self.draw_coco_nero(mode='single')

        # drop down menu on selection which quantity to plot
        # title
        # draw text
        plot_quantity_pixmap = QPixmap(300, 50)
        plot_quantity_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(plot_quantity_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 300, 50, QtGui.Qt.AlignLeft, 'NERO Metric: ')
        painter.end()

        # create label to contain the texts
        self.plot_quantity_label = QLabel(self)
        self.plot_quantity_label.setFixedSize(QtCore.QSize(300, 50))
        self.plot_quantity_label.setPixmap(plot_quantity_pixmap)
        self.plot_quantity_label.setContentsMargins(0, 0, 0, 0)

        # menu
        quantity_menu = QtWidgets.QComboBox()
        quantity_menu.setFixedSize(QtCore.QSize(220, 50))
        quantity_menu.setStyleSheet(
            'color: black; font-family: Helvetica; font-style: normal; font-size: 34px'
        )
        quantity_menu.setEditable(True)
        quantity_menu.lineEdit().setReadOnly(True)
        quantity_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)

        quantity_menu.addItem('Confidence')
        quantity_menu.addItem('IOU')
        quantity_menu.addItem('Conf*IOU')
        # some extra qualities to plot (not available in individual NERO plot)
        if not self.demo:
            quantity_menu.addItem('Precision')
            quantity_menu.addItem('Recall')
            quantity_menu.addItem('AP')
            quantity_menu.addItem('F1 Score')

        # define default plotting quantity
        # self.quantity_menu.setCurrentIndex(0)
        quantity_menu.setCurrentText('Conf*IOU')
        self.quantity_name = 'Conf*IOU'
        # quantity_menu.setCurrentText('Confidence')
        # self.quantity_name = 'Confidence'

        # connect the drop down menu with actions
        quantity_menu.currentTextChanged.connect(coco_nero_quantity_changed)
        if self.demo:
            self.plot_info_layout = QtWidgets.QHBoxLayout()
            self.plot_info_layout.addWidget(self.plot_quantity_label)
            self.plot_info_layout.addWidget(quantity_menu)
            self.plot_info_layout.setContentsMargins(20, 0, 0, 0)
            self.demo_layout.addLayout(self.plot_info_layout, 0, 2)
        else:
            self.aggregate_plot_control_layout.addWidget(quantity_menu, 1, 0)

        # model 1
        # averaged confidence and iou of the top results (ranked by IOU)
        self.aggregate_avg_conf_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
        # conf * label_correctness
        self.aggregate_avg_conf_correctness_1 = np.zeros(
            (len(self.y_translation), len(self.x_translation))
        )
        # iou
        self.aggregate_avg_iou_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
        # iou * label_correctness
        self.aggregate_avg_iou_correctness_1 = np.zeros(
            (len(self.y_translation), len(self.x_translation))
        )
        # precision
        self.aggregate_avg_precision_1 = np.zeros(
            (len(self.y_translation), len(self.x_translation))
        )
        # recall
        self.aggregate_avg_recall_1 = np.zeros((len(self.y_translation), len(self.x_translation)))
        # F-score
        self.aggregate_avg_F_measure_1 = np.zeros(
            (len(self.y_translation), len(self.x_translation))
        )

        # model 2
        # confidence
        self.aggregate_avg_conf_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
        # conf * label_correctness
        self.aggregate_avg_conf_correctness_2 = np.zeros(
            (len(self.y_translation), len(self.x_translation))
        )
        # iou
        self.aggregate_avg_iou_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
        # iou * label_correctness
        self.aggregate_avg_iou_correctness_2 = np.zeros(
            (len(self.y_translation), len(self.x_translation))
        )
        # precision
        self.aggregate_avg_precision_2 = np.zeros(
            (len(self.y_translation), len(self.x_translation))
        )
        # recall
        self.aggregate_avg_recall_2 = np.zeros((len(self.y_translation), len(self.x_translation)))
        # F-score
        self.aggregate_avg_F_measure_2 = np.zeros(
            (len(self.y_translation), len(self.x_translation))
        )

        for y in range(len(self.y_translation)):
            for x in range(len(self.x_translation)):
                # model 1
                all_samples_conf_sum_1 = []
                all_sampels_conf_correctness_sum_1 = []
                all_samples_iou_sum_1 = []
                all_sampels_iou_correctness_sum_1 = []
                all_samples_precision_sum_1 = []
                all_samples_recall_sum_1 = []
                all_samples_F_measure_sum_1 = []

                # model 2
                all_samples_conf_sum_2 = []
                all_sampels_conf_correctness_sum_2 = []
                all_samples_iou_sum_2 = []
                all_sampels_iou_correctness_sum_2 = []
                all_samples_precision_sum_2 = []
                all_samples_recall_sum_2 = []
                all_samples_F_measure_sum_2 = []

                # either all the classes or one specific class
                # if (
                #     self.class_selection == 'all'
                #     or self.class_selection == self.loaded_images_labels[i]
                # ):
                if self.use_consensus:
                    for i in range(len(self.aggregate_consensus_outputs_1[y, x])):
                        # model 1
                        all_samples_conf_sum_1.append(
                            self.aggregate_consensus_outputs_1[y, x][i][0, 4]
                        )
                        all_sampels_conf_correctness_sum_1.append(
                            self.aggregate_consensus_outputs_1[y, x][i][0, 4]
                            * self.aggregate_consensus_outputs_1[y, x][i][0, 7]
                        )
                        all_samples_iou_sum_1.append(
                            self.aggregate_consensus_outputs_1[y, x][i][0, 6]
                        )
                        all_sampels_iou_correctness_sum_1.append(
                            self.aggregate_consensus_outputs_1[y, x][i][0, 6]
                            * self.aggregate_consensus_outputs_1[y, x][i][0, 7]
                        )
                        all_samples_precision_sum_1.append(
                            self.aggregate_consensus_precision_1[y, x][i]
                        )
                        all_samples_recall_sum_1.append(self.aggregate_consensus_recall_1[y, x][i])
                        all_samples_F_measure_sum_1.append(
                            self.aggregate_consensus_F_measure_1[y, x][i]
                        )

                        # model 2
                        all_samples_conf_sum_2.append(
                            self.aggregate_consensus_outputs_2[y, x][i][0, 4]
                        )
                        all_sampels_conf_correctness_sum_2.append(
                            self.aggregate_consensus_outputs_2[y, x][i][0, 4]
                            * self.aggregate_consensus_outputs_2[y, x][i][0, 7]
                        )
                        all_samples_iou_sum_2.append(
                            self.aggregate_consensus_outputs_2[y, x][i][0, 6]
                        )
                        all_sampels_iou_correctness_sum_2.append(
                            self.aggregate_consensus_outputs_2[y, x][i][0, 6]
                            * self.aggregate_consensus_outputs_2[y, x][i][0, 7]
                        )
                        all_samples_precision_sum_2.append(
                            self.aggregate_consensus_precision_2[y, x][i]
                        )
                        all_samples_recall_sum_2.append(self.aggregate_consensus_recall_2[y, x][i])
                        all_samples_F_measure_sum_2.append(
                            self.aggregate_consensus_F_measure_2[y, x][i]
                        )
                else:
                    for i in range(len(self.aggregate_outputs_1[y, x])):
                        # model 1
                        all_samples_conf_sum_1.append(self.aggregate_outputs_1[y, x][i][0, 4])
                        all_sampels_conf_correctness_sum_1.append(
                            self.aggregate_outputs_1[y, x][i][0, 4]
                            * self.aggregate_outputs_1[y, x][i][0, 7]
                        )
                        all_samples_iou_sum_1.append(self.aggregate_outputs_1[y, x][i][0, 6])
                        all_sampels_iou_correctness_sum_1.append(
                            self.aggregate_outputs_1[y, x][i][0, 6]
                            * self.aggregate_outputs_1[y, x][i][0, 7]
                        )
                        all_samples_precision_sum_1.append(self.aggregate_precision_1[y, x][i])
                        all_samples_recall_sum_1.append(self.aggregate_recall_1[y, x][i])
                        all_samples_F_measure_sum_1.append(self.aggregate_F_measure_1[y, x][i])

                        # model 2
                        all_samples_conf_sum_2.append(self.aggregate_outputs_2[y, x][i][0, 4])
                        all_sampels_conf_correctness_sum_2.append(
                            self.aggregate_outputs_2[y, x][i][0, 4]
                            * self.aggregate_outputs_2[y, x][i][0, 7]
                        )
                        all_samples_iou_sum_2.append(self.aggregate_outputs_2[y, x][i][0, 6])
                        all_sampels_iou_correctness_sum_2.append(
                            self.aggregate_outputs_2[y, x][i][0, 6]
                            * self.aggregate_outputs_2[y, x][i][0, 7]
                        )
                        all_samples_precision_sum_2.append(self.aggregate_precision_2[y, x][i])
                        all_samples_recall_sum_2.append(self.aggregate_recall_2[y, x][i])
                        all_samples_F_measure_sum_2.append(self.aggregate_F_measure_2[y, x][i])

                # take the average result
                self.aggregate_avg_conf_1[y, x] = np.mean(all_samples_conf_sum_1)
                self.aggregate_avg_conf_correctness_1[y, x] = np.mean(
                    all_sampels_conf_correctness_sum_1
                )
                self.aggregate_avg_iou_1[y, x] = np.mean(all_samples_iou_sum_1)
                self.aggregate_avg_iou_correctness_1[y, x] = np.mean(
                    all_sampels_iou_correctness_sum_1
                )
                self.aggregate_avg_precision_1[y, x] = np.mean(all_samples_precision_sum_1)
                self.aggregate_avg_recall_1[y, x] = np.mean(all_samples_recall_sum_1)
                self.aggregate_avg_F_measure_1[y, x] = np.mean(all_samples_F_measure_sum_1)

                self.aggregate_avg_conf_2[y, x] = np.mean(all_samples_conf_sum_2)
                self.aggregate_avg_conf_correctness_2[y, x] = np.mean(
                    all_sampels_conf_correctness_sum_2
                )
                self.aggregate_avg_iou_2[y, x] = np.mean(all_samples_iou_sum_2)
                self.aggregate_avg_iou_correctness_2[y, x] = np.mean(
                    all_sampels_iou_correctness_sum_2
                )
                self.aggregate_avg_precision_2[y, x] = np.mean(all_samples_precision_sum_2)
                self.aggregate_avg_recall_2[y, x] = np.mean(all_samples_recall_sum_2)
                self.aggregate_avg_F_measure_2[y, x] = np.mean(all_samples_F_measure_sum_2)

        # default plotting quantity
        if self.quantity_name == 'Confidence':
            self.cur_aggregate_plot_quantity_1 = self.aggregate_avg_conf_1
            self.cur_aggregate_plot_quantity_2 = self.aggregate_avg_conf_2
        elif self.quantity_name == 'IOU':
            self.cur_aggregate_plot_quantity_1 = self.aggregate_avg_iou_correctness_1
            self.cur_aggregate_plot_quantity_2 = self.aggregate_avg_iou_correctness_2
        elif self.quantity_name == 'Conf*IOU':
            self.cur_aggregate_plot_quantity_1 = (
                self.aggregate_avg_conf_1 * self.aggregate_avg_iou_correctness_1
            )
            self.cur_aggregate_plot_quantity_2 = (
                self.aggregate_avg_conf_2 * self.aggregate_avg_iou_correctness_2
            )
        else:
            raise Exception(f'Unknown quantity {self.quantity_name}')
        # print(self.cur_aggregate_plot_quantity_1[0, 0], self.cur_aggregate_plot_quantity_2[0, 0])
        # draw the heatmap
        self.draw_coco_nero(mode='aggregate')

    # display COCO single results
    def display_coco_single_result(self):

        # if single mode, change control menus' locations
        if self.data_mode == 'single':
            # move the model menu on top of the each individual NERO plot when in single mode
            if not self.demo:
                self.single_result_layout.addWidget(
                    self.model_1_menu, 0, 1, 1, 1, QtCore.Qt.AlignCenter
                )
                self.single_result_layout.addWidget(
                    self.model_2_menu, 0, 2, 1, 1, QtCore.Qt.AlignCenter
                )

                # move run button below the displayed image
                self.single_result_layout.addWidget(self.run_button, 2, 0)
                self.single_result_layout.addWidget(self.use_cache_checkbox, 3, 0)

        # plot current field-of-view's detailed prediction results
        self.draw_model_output()

        self.single_result_existed = True

        # draw result using heatmaps
        @QtCore.Slot()
        def realtime_inference_checkbox_clicked():
            if self.realtime_inference_checkbox.isChecked():
                self.realtime_inference = True
            else:
                self.realtime_inference = False

        # checkbox on if doing real-time inference
        self.realtime_inference_checkbox = QtWidgets.QCheckBox('Realtime inference when dragging')
        self.realtime_inference_checkbox.setStyleSheet(
            'color: black; font-family: Helvetica; font-style: normal; font-size: 18px; background-color: white;'
        )
        self.realtime_inference_checkbox.setFixedSize(QtCore.QSize(300, 30))
        self.realtime_inference_checkbox.setContentsMargins(0, 0, 0, 0)
        self.realtime_inference_checkbox.stateChanged.connect(realtime_inference_checkbox_clicked)
        if self.realtime_inference:
            self.realtime_inference_checkbox.setChecked(True)
        else:
            self.realtime_inference_checkbox.setChecked(False)

        # layout that controls the plotting items
        if self.demo:
            self.demo_layout.addWidget(self.realtime_inference_checkbox, 0, 4)
        else:
            self.single_plot_control_layout = QtWidgets.QVBoxLayout()
            self.single_plot_control_layout.addWidget(self.realtime_inference_checkbox)

        # define default plotting quantity
        if self.data_mode == 'single':
            # add plot control layout to general layout
            self.single_result_layout.addLayout(self.single_plot_control_layout, 0, 0)
            self.cur_single_plot_quantity_1 = (
                self.all_quantities_1[:, :, 4] * self.all_quantities_1[:, :, 6]
            )
            self.cur_single_plot_quantity_2 = (
                self.all_quantities_2[:, :, 4] * self.all_quantities_2[:, :, 6]
            )
        elif self.data_mode == 'aggregate':
            # add plot control layout to general layout
            if not self.demo:
                self.aggregate_result_layout.addLayout(self.single_plot_control_layout, 2, 3)
            # current selected individual images' result on all transformations
            self.cur_single_plot_quantity_1 = np.zeros(
                (len(self.y_translation), len(self.x_translation))
            )
            self.cur_single_plot_quantity_2 = np.zeros(
                (len(self.y_translation), len(self.x_translation))
            )
            for y in range(len(self.y_translation)):
                for x in range(len(self.x_translation)):
                    self.cur_single_plot_quantity_1[y, x] = (
                        self.aggregate_outputs_1[y, x][self.image_index][0, 4]
                        * self.aggregate_outputs_1[y, x][self.image_index][0, 6]
                    )
                    self.cur_single_plot_quantity_2[y, x] = (
                        self.aggregate_outputs_2[y, x][self.image_index][0, 4]
                        * self.aggregate_outputs_2[y, x][self.image_index][0, 6]
                    )

        # draw the heatmap
        self.draw_coco_nero(mode='single')

    # display COCO aggregate result
    def display_piv_aggregate_result(self):
        if not self.demo:
            # move the model menu on top of the each aggregate NERO plot
            self.aggregate_result_layout.addWidget(
                self.model_1_menu, 0, 1, 1, 1, QtCore.Qt.AlignCenter
            )
            self.aggregate_result_layout.addWidget(
                self.model_2_menu, 0, 2, 1, 1, QtCore.Qt.AlignCenter
            )

            # move run button in the first column (after aggregate heatmap control)
            self.aggregate_plot_control_layout.addWidget(self.run_button, 4, 0)
            self.aggregate_plot_control_layout.addWidget(self.use_cache_checkbox, 5, 0)

        self.aggregate_result_existed = True

        # helper function on compute, normalize the loss and display quantity
        def compute_nero_plot_quantity():
            print('Compute PIV nero plot quantity')
            # compute loss using torch loss module
            if self.quantity_name == 'RMSE':
                self.loss_module = nero_utilities.RMSELoss()
            elif self.quantity_name == 'MSE':
                self.loss_module = torch.nn.MSELoss()
            elif self.quantity_name == 'MAE':
                self.loss_module = torch.nn.L1Loss()
            elif self.quantity_name == 'AEE':
                self.loss_module = nero_utilities.AEELoss()

            # try loading from cache
            cur_losses_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_aggregate_1'
            )
            cur_losses_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_aggregate_2'
            )
            if not self.load_successfully:
                # keep the same dimension
                cur_losses_1 = np.zeros(
                    (
                        self.num_transformations,
                        len(self.aggregate_outputs_1[0]),
                        self.image_size,
                        self.image_size,
                    )
                )
                cur_losses_2 = np.zeros(
                    (
                        self.num_transformations,
                        len(self.aggregate_outputs_1[0]),
                        self.image_size,
                        self.image_size,
                    )
                )
                for i in range(self.num_transformations):
                    for j in range(len(self.aggregate_outputs_1[i])):
                        cur_losses_1[i, j] = (
                            self.loss_module(
                                self.aggregate_ground_truths[i, j],
                                self.aggregate_outputs_1[i, j],
                                reduction='none',
                            )
                            .numpy()
                            .mean(axis=2)
                        )
                        cur_losses_2[i, j] = (
                            self.loss_module(
                                self.aggregate_ground_truths[i, j],
                                self.aggregate_outputs_2[i, j],
                                reduction='none',
                            )
                            .numpy()
                            .mean(axis=2)
                        )

                # save to cache
                self.save_to_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_aggregate_1',
                    cur_losses_1,
                )
                self.save_to_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_aggregate_2',
                    cur_losses_2,
                )

            # get the 0 and 80 percentile as the threshold for colormap
            all_losses = np.concatenate([cur_losses_1.flatten(), cur_losses_2.flatten()])
            self.loss_low_bound = np.percentile(all_losses, 0)
            self.loss_high_bound = np.percentile(all_losses, 80)
            print('Aggregate loss 0 and 80 percentile', self.loss_low_bound, self.loss_high_bound)

            # plot quantity is the average among all samples
            self.cur_aggregate_plot_quantity_1 = cur_losses_1.mean(axis=1)
            self.cur_aggregate_plot_quantity_2 = cur_losses_2.mean(axis=1)

            # compute single result if needed as well
            if self.single_result_existed:
                # try loading from cache
                cur_losses_1 = self.load_from_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_single_1'
                )
                cur_losses_2 = self.load_from_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_single_2'
                )
                if not self.load_successfully:
                    # keep the same dimension
                    cur_losses_1 = np.zeros(
                        (self.num_transformations, self.image_size, self.image_size)
                    )
                    cur_losses_2 = np.zeros(
                        (self.num_transformations, self.image_size, self.image_size)
                    )
                    for i in range(self.num_transformations):
                        cur_losses_1[i] = (
                            self.loss_module(
                                self.all_ground_truths[i],
                                self.all_quantities_1[i],
                                reduction='none',
                            )
                            .numpy()
                            .mean(axis=2)
                        )
                        cur_losses_2[i] = (
                            self.loss_module(
                                self.all_ground_truths[i],
                                self.all_quantities_2[i],
                                reduction='none',
                            )
                            .numpy()
                            .mean(axis=2)
                        )

                    # save to cache
                    self.save_to_cache(
                        f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_single_1',
                        cur_losses_1,
                    )
                    self.save_to_cache(
                        f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_single_2',
                        cur_losses_2,
                    )

                # average element-wise loss to scalar and normalize between 0 and 1
                self.cur_single_plot_quantity_1 = cur_losses_1
                self.cur_single_plot_quantity_2 = cur_losses_2

        @QtCore.Slot()
        def piv_nero_quantity_changed(text):
            print('Plotting:', text, 'on heatmap')
            self.quantity_name = text

            if text == 'RMSE':
                self.quantity_name = text
            elif text == 'MSE':
                self.quantity_name = text
            elif text == 'MAE':
                self.quantity_name = text
            elif text == 'AEE':
                self.quantity_name = text

            # compute the quantity to plot
            compute_nero_plot_quantity()

            # re-display the heatmap
            self.draw_piv_nero(mode='aggregate')

            # re-run dimension reduction and show result
            if self.dr_result_existed:
                self.run_dimension_reduction()

            # re-draw single result if needed
            if self.single_result_existed:
                self.draw_piv_nero(mode='single')

        # title
        # draw text
        plot_quantity_pixmap = QPixmap(300, 50)
        plot_quantity_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(plot_quantity_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 300, 50, QtGui.Qt.AlignLeft, 'NERO Metric: ')
        painter.end()
        # create label to contain the texts
        self.plot_quantity_label = QLabel(self)
        self.plot_quantity_label.setFixedSize(QtCore.QSize(300, 50))
        self.plot_quantity_label.setPixmap(plot_quantity_pixmap)
        # drop down menu on selection which quantity to plot
        quantity_menu = QtWidgets.QComboBox()
        quantity_menu.setFixedSize(QtCore.QSize(150, 50))
        quantity_menu.setStyleSheet(
            'color: black; font-family: Helvetica; font-style: normal; font-size: 34px'
        )
        quantity_menu.setEditable(True)
        quantity_menu.lineEdit().setReadOnly(True)
        quantity_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)

        quantity_menu.addItem('RMSE')
        quantity_menu.addItem('MSE')
        quantity_menu.addItem('MAE')
        quantity_menu.addItem('AEE')
        quantity_menu.setCurrentText('RMSE')

        # connect the drop down menu with actions
        quantity_menu.currentTextChanged.connect(piv_nero_quantity_changed)
        if self.demo:
            self.plot_info_layout = QtWidgets.QHBoxLayout()
            self.plot_info_layout.addWidget(self.plot_quantity_label)
            self.plot_info_layout.addWidget(quantity_menu)
            self.demo_layout.addLayout(self.plot_info_layout, 0, 2, 1, 1)
        else:
            self.aggregate_plot_control_layout.addWidget(quantity_menu, 1, 0)

        # define default plotting quantity (RMSE)
        self.quantity_name = 'RMSE'
        self.aggregate_loss_module = nero_utilities.RMSELoss()

        # compute aggregate plot quantity
        compute_nero_plot_quantity()

        # draw the aggregate NERO plot
        self.draw_piv_nero(mode='aggregate')

    # display PIV single results
    def display_piv_single_result(self):
        # if single mode, change control menus' locations
        if self.data_mode == 'single':
            # move the model menu on top of the each individual NERO plot when in single mode
            self.single_result_layout.addWidget(
                self.model_1_menu, 0, 1, 1, 1, QtCore.Qt.AlignCenter
            )
            self.single_result_layout.addWidget(
                self.model_2_menu, 0, 2, 1, 1, QtCore.Qt.AlignCenter
            )

            # move run button below the displayed image
            self.single_result_layout.addWidget(self.run_button, 3, 0)
            self.single_result_layout.addWidget(self.use_cache_checkbox, 4, 0)

        # helper function on compute, normalize the loss and display quantity
        def compute_single_nero_plot_quantity():
            # compute loss using torch loss module
            if self.quantity_name == 'RMSE':
                self.loss_module = nero_utilities.RMSELoss()
            elif self.quantity_name == 'MSE':
                self.loss_module = torch.nn.MSELoss()
            elif self.quantity_name == 'MAE':
                self.loss_module = torch.nn.L1Loss()
            elif self.quantity_name == 'AEE':
                self.loss_module = nero_utilities.AEELoss()

            # try loading from cache
            cur_losses_1 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_single_1'
            )
            cur_losses_2 = self.load_from_cache(
                f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_single_2'
            )
            if not self.load_successfully:
                # keep the same dimension
                cur_losses_1 = np.zeros(
                    (self.num_transformations, self.image_size, self.image_size)
                )
                cur_losses_2 = np.zeros(
                    (self.num_transformations, self.image_size, self.image_size)
                )
                for i in range(self.num_transformations):
                    cur_losses_1[i] = (
                        self.loss_module(
                            self.all_ground_truths[i], self.all_quantities_1[i], reduction='none'
                        )
                        .numpy()
                        .mean(axis=2)
                    )
                    cur_losses_2[i] = (
                        self.loss_module(
                            self.all_ground_truths[i], self.all_quantities_2[i], reduction='none'
                        )
                        .numpy()
                        .mean(axis=2)
                    )

                # save to cache
                self.save_to_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_single_1',
                    cur_losses_1,
                )
                self.save_to_cache(
                    f'{self.mode}_{self.data_mode}_{self.dataset_name}_{self.model_1_cache_name}_{self.class_selection}_{self.quantity_name}_single_2',
                    cur_losses_2,
                )

            # get the 0 and 80 percentile as the threshold for colormap
            # when in aggregate mode, continue using aggregate range
            if self.data_mode == 'single':
                all_losses = np.concatenate([cur_losses_1.flatten(), cur_losses_2.flatten()])
                self.loss_low_bound = np.percentile(all_losses, 0)
                self.loss_high_bound = np.percentile(all_losses, 80)
                # print(self.loss_low_bound, self.loss_high_bound)

            # average element-wise loss to scalar and normalize between 0 and 1
            self.cur_single_plot_quantity_1 = cur_losses_1
            self.cur_single_plot_quantity_2 = cur_losses_2

        @QtCore.Slot()
        def piv_nero_quantity_changed(text):
            print('Plotting:', text, 'on detailed PIV plots')
            self.quantity_name = text

            # compute the quantity needed to plot individual NERO plot
            compute_single_nero_plot_quantity()

            # plot/update the individual NERO plot
            self.draw_piv_nero(mode='single')

            # update detailed plot of PIV
            self.draw_piv_details()

        # single mode only visualization
        if self.data_mode == 'single':
            # drop down menu on selection which quantity to plot
            # layout that controls the plotting items
            self.single_plot_control_layout = QtWidgets.QVBoxLayout()
            quantity_menu = QtWidgets.QComboBox()
            quantity_menu.setFixedSize(QtCore.QSize(220, 50))
            quantity_menu.setStyleSheet('font-size: 18px')
            quantity_menu.setStyleSheet('color: black')
            quantity_menu.setEditable(True)
            quantity_menu.lineEdit().setReadOnly(True)
            quantity_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)

            # all the different plot quantities (losses)
            self.all_plot_quantities = ['RMSE', 'MSE', 'MAE', 'AEE']
            for cur_quantity in self.all_plot_quantities:
                quantity_menu.addItem(cur_quantity)

            quantity_menu.setCurrentText(self.all_plot_quantities[0])
            # by default the loss is RMSE
            self.quantity_name = 'RMSE'
            self.loss_module = nero_utilities.RMSELoss()

            # connect the drop down menu with actions
            quantity_menu.currentTextChanged.connect(piv_nero_quantity_changed)
            self.single_plot_control_layout.addWidget(quantity_menu)

            # add plot control layout to general layout
            self.single_result_layout.addLayout(self.single_plot_control_layout, 0, 0)
            # compute the plot quantities
            self.cur_single_plot_quantity_1 = np.zeros(
                (self.num_transformations, self.image_size, self.image_size)
            )
            self.cur_single_plot_quantity_2 = np.zeros(
                (self.num_transformations, self.image_size, self.image_size)
            )
            compute_single_nero_plot_quantity()

        # when in three level view
        elif self.data_mode == 'aggregate':
            # plot quantity in individual nero plot
            self.cur_single_plot_quantity_1 = np.zeros(
                (self.num_transformations, self.image_size, self.image_size)
            )
            self.cur_single_plot_quantity_2 = np.zeros(
                (self.num_transformations, self.image_size, self.image_size)
            )
            compute_single_nero_plot_quantity()

        # visualize the individual NERO plot of the current input
        self.draw_piv_nero(mode='single')

        # the detailed plot of PIV
        self.draw_piv_details()

    # mouse move event only applies in the MNIST case
    def mouseMoveEvent(self, event):

        if self.mode == 'digit_recognition' and self.image_existed:
            cur_mouse_pos = [
                event.position().x() - self.image_center_x,
                event.position().y() - self.image_center_y,
            ]

            angle_change = (
                -(
                    (
                        self.prev_mouse_pos[0] * cur_mouse_pos[1]
                        - self.prev_mouse_pos[1] * cur_mouse_pos[0]
                    )
                    / (
                        self.prev_mouse_pos[0] * self.prev_mouse_pos[0]
                        + self.prev_mouse_pos[1] * self.prev_mouse_pos[1]
                    )
                )
                * 180
            )

            self.cur_rotation_angle += angle_change
            # print(f'\nRotated {self.cur_rotation_angle} degrees')
            # rotate the image tensor
            self.cur_image_pt = nero_transform.rotate_mnist_image(
                self.loaded_image_pt, self.cur_rotation_angle
            )
            # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(angle), QtCore.Qt.SmoothTransformation)
            # convert image tensor to qt image and resize for display
            self.cur_display_image = nero_utilities.tensor_to_qt_image(
                self.cur_image_pt, self.display_image_size, revert_color=True
            )
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
                cur_x = 1 * np.cos(self.cur_rotation_angle / 180 * np.pi)
                cur_y = 1 * np.sin(self.cur_rotation_angle / 180 * np.pi)
                line_x = [0, cur_x]
                line_y = [0, cur_y]
                self.cur_line = self.polar_plot.plot(
                    line_x, line_y, pen=QtGui.QPen(QtGui.Qt.green, 0.02)
                )

                # display current results on the line
                self.draw_circle_on_polar()

            self.prev_mouse_pos = cur_mouse_pos

    def mousePressEvent(self, event):
        if self.mode == 'digit_recognition' and self.image_existed:
            self.image_center_x = self.image_label.x() + self.image_label.width() / 2
            self.image_center_y = self.image_label.y() + self.image_label.height() / 2
            self.prev_mouse_pos = [
                event.position().x() - self.image_center_x,
                event.position().y() - self.image_center_y,
            ]

    # called when a key is pressed
    def keyPressEvent(self, event):
        key_pressed = event.text()

        # different key pressed
        if 'h' == key_pressed or '?' == key_pressed:
            self.print_help()

    # print help message
    def print_help(self):
        print('Ah Oh, help not available')


if __name__ == '__main__':

    # input arguments
    parser = argparse.ArgumentParser()
    # mode (digit_recognition, object_detection or piv)
    parser.add_argument('--mode', action='store', nargs=1, dest='mode')
    parser.add_argument('--cache_path', action='store', nargs=1, dest='cache_path')
    parser.add_argument('--demo', action='store_true', dest='demo', default=False)
    args = parser.parse_args()
    if args.mode:
        mode = args.mode[0]
    else:
        mode = None
    if args.cache_path:
        cache_path = args.cache_path[0]
    else:
        cache_path = None
    demo = args.demo

    app = QtWidgets.QApplication([])
    widget = UI_MainWindow(mode, demo, cache_path)
    widget.show()

    # run the app
    app.exec()

    # remove all .GIF from cache
    if mode == 'piv':
        all_gif_paths = glob.glob(os.path.join(os.getcwd(), 'cache', f'{mode}', '*.gif'))
        for gif_path in all_gif_paths:
            os.remove(gif_path)

    # exit the app
    sys.exit()
