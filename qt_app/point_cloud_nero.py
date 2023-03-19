"""
NERO interface for point cloud classification
Author: Zhuokai Zhao
"""

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
from collections import defaultdict

import nero_transform
import nero_utilities
import nero_run_model

import warnings

# import teem
# import qiv
import datasets
import interface_util

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
    def __init__(self, cache_path):
        super().__init__()
        # window size
        self.resize(2280, 1080)
        title_name = 'Point Cloud Classification'
        # set window title
        self.setWindowTitle(f'Non-Equivariance Revealed on Orbits: {title_name}')
        # white background color
        self.setStyleSheet('background-color: rgb(255, 255, 255);')
        # general layout
        self.interface_layout = QtWidgets.QGridLayout(self)
        self.interface_layout.setAlignment(QtCore.Qt.AlignCenter)
        # left, top, right, and bottom margins
        self.interface_layout.setContentsMargins(10, 10, 10, 10)
        # spacing
        self.interface_layout.setHorizontalSpacing(0)
        self.interface_layout.setVerticalSpacing(0)

        # still define a mode parameter so we can have a unified cache for multiple application
        self.mode = 'point_cloud_classification'

        # initialize status of various contents in the interface
        self.display_existed = False   # point cloud to display
        self.data_existed = False   # inference data
        self.aggregate_result_existed = False   # result for aggregate NERO plots
        self.single_result_existed = False   # result for single NERO plots
        self.dr_result_existed = False   # result for DR plot

        # image size that is used for display
        self.display_size = 320
        # heatmap and detailed image plot size
        self.plot_size = 320

        # load/initialize program cache
        self.use_cache = False

        # when we have an input cache path
        if cache_path != None:
            self.cache_path = cache_path
            self.cache_dir = self.cache_path.removesuffix(self.cache_path.split('/')[-1])
        # initialize a new cache
        else:
            self.cache_dir = os.path.join(os.getcwd(), 'cache')
            if not os.path.isdir(self.cache_dir):
                os.mkdir(self.cache_dir)

            self.cache_path = os.path.join(self.cache_dir, f'{self.mode}', 'nero_cache.npz')
            # if not exist, creat one
            if not os.path.isfile(self.cache_path):
                np.savez(self.cache_path)

        # cache
        self.cache = dict(np.load(self.cache_path, allow_pickle=True))

        # if we are doing real-time inference when dragging the field of view
        if torch.cuda.is_available():
            self.realtime_inference = True
        else:
            self.realtime_inference = False

        # define dataset path
        self.data_dir = f'./example_data/{self.mode}/modelnet_normal_resampled'

        # Data and models loading
        # initialize data loading and associated interface
        self.init_point_cloud_data()
        self.init_data_loading_interface()
        # initialize data loading and associated interface
        self.init_point_cloud_models()
        self.init_model_loading_interface()

        # initialize general control panel
        self.init_general_control_interface()

        # Aggregate NERO plot
        self.prepare_aggregate_results()
        self.init_aggregate_plot_interface()
        self.draw_point_cloud_aggregate_nero()

        # DR plot
        self.prepare_dr_results()
        self.init_dr_plot_interface()
        self.draw_dr_plot()

        # Individual NERO plot

        # Detailed plot

        print(f'\nFinished rendering main layout')

    ################## Data Loading Related ##################
    def init_point_cloud_data(self):
        # modelnet40 and modelnet10
        # data samples paths
        self.all_nums_classes = [40, 10]
        self.all_data_paths = [
            os.path.join(self.data_dir, f'modelnet{i}_test.txt') for i in self.all_nums_classes
        ]
        # classes names paths
        self.all_names_paths = [
            os.path.join(self.data_dir, f'modelnet{i}_shape_names.txt')
            for i in self.all_nums_classes
        ]

        # when initializing, take the first path (index 0 is the prompt)
        # when changed, we should have dataset_index defined ready from interface
        self.dataset_index = 1

    def init_data_loading_interface(self):

        # dataset selection
        # text prompt
        model_pixmap = QPixmap(350, 50)
        model_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(model_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 350, 50, QtGui.Qt.AlignLeft, 'Data Set:')
        painter.end()
        # create label to contain the texts
        self.model_label = QLabel(self)
        self.model_label.setFixedSize(QtCore.QSize(300, 50))
        self.model_label.setPixmap(model_pixmap)
        # add to the layout
        self.interface_layout.addWidget(self.model_label, 0, 0)
        # drop down menu
        self.aggregate_image_menu = QtWidgets.QComboBox()
        self.aggregate_image_menu.setFixedSize(QtCore.QSize(220, 50))
        self.aggregate_image_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        # prompt in drop down menu
        self.aggregate_image_menu.addItem('Input dataset')
        # load all dataset names from the data paths
        for i in range(len(self.all_data_paths)):
            # cur_name = self.all_data_paths[i].split('/')[-1].split('.')[0]
            cur_name = self.all_data_paths[i].split('/')[-1].split('.')[0].split('_')[0]
            self.aggregate_image_menu.addItem(cur_name)
        # set default data selection
        self.aggregate_image_menu.setCurrentIndex(self.dataset_index)
        self.dataset_name = self.aggregate_image_menu.currentText()

        # connect the drop down menu with actions
        self.aggregate_image_menu.currentTextChanged.connect(self._dataset_selection_changed)
        self.aggregate_image_menu.setEditable(True)
        self.aggregate_image_menu.lineEdit().setReadOnly(True)
        self.aggregate_image_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        aggregate_image_menu_layout = QtWidgets.QHBoxLayout()
        aggregate_image_menu_layout.setContentsMargins(150, 0, 0, 0)
        aggregate_image_menu_layout.addWidget(self.aggregate_image_menu)
        self.interface_layout.addLayout(aggregate_image_menu_layout, 0, 0)

        # drop down menu on choosing the class within the dataset
        # QPixmap that contains the title text
        class_selection_pixmap = QPixmap(300, 50)
        class_selection_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(class_selection_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 350, 50, QtGui.Qt.AlignLeft, 'Subset: ')
        painter.end()
        # QLabel that contains the QPixmap
        self.class_selection_label = QLabel(self)
        self.class_selection_label.setFixedSize(QtCore.QSize(400, 50))
        self.class_selection_label.setPixmap(class_selection_pixmap)
        # add QLabel to the layout
        self.interface_layout.addWidget(self.class_selection_label, 1, 0)
        # create the drop down menu
        self.class_selection_menu = QtWidgets.QComboBox()
        self.class_selection_menu.setFixedSize(QtCore.QSize(220, 50))
        self.class_selection_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        self.class_selection_menu.addItem(f'All classes')
        # set default to 'all', which means averaged over all classes
        self.class_selection = 'all'
        self.class_selection_menu.setCurrentIndex(0)
        # need to load data here otherwise we don't have cur_classes_names
        self.load_point_cloud_data()

        # add all classes as items
        for cur_class in self.cur_classes_names:
            self.class_selection_menu.addItem(f'{cur_class}')
        # connect the drop down menu with actions
        self.class_selection_menu.currentTextChanged.connect(self._class_selection_changed)
        self.class_selection_menu.setEditable(True)
        self.class_selection_menu.lineEdit().setReadOnly(True)
        self.class_selection_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        # local layout that contains the drop down menu
        class_selection_menu_layout = QtWidgets.QHBoxLayout()
        class_selection_menu_layout.setContentsMargins(150, 0, 0, 0)
        class_selection_menu_layout.addWidget(self.class_selection_menu)
        # add the general layout
        self.interface_layout.addLayout(class_selection_menu_layout, 1, 0)

    def load_point_cloud_data(self):

        # get data and classes names path from selected 1-based index
        self.cur_data_path = self.all_data_paths[self.dataset_index - 1]
        self.cur_name_path = self.all_names_paths[self.dataset_index - 1]
        print(f'\nLoading data from {self.cur_data_path}')
        # load all the point cloud ids (such as airplane_0627)
        point_cloud_ids = [line.rstrip() for line in open(self.cur_data_path)]
        # point cloud ids have name_index format, load the names (such as airplane)
        point_cloud_names = ['_'.join(x.split('_')[0:-1]) for x in point_cloud_ids]
        # point cloud ground truth label (name) and path pair
        self.point_cloud_paths = [
            (
                point_cloud_names[i],
                os.path.join(self.data_dir, point_cloud_names[i], point_cloud_ids[i]) + '.txt',
            )
            for i in range(len(point_cloud_ids))
        ]

        # load the name files
        self.cur_classes_names = nero_utilities.load_modelnet_classes_file(self.cur_name_path)
        self.cur_num_classes = len(self.cur_classes_names)

        # get the data indices that the user selects
        self.cur_class_indices = []
        if self.class_selection == 'all':
            self.cur_class_indices = list(range(len(self.point_cloud_paths)))
        else:
            for i in range(len(self.point_cloud_paths)):
                if self.point_cloud_paths[i][0] == self.class_selection:
                    self.cur_class_indices.append(i)

        # dataset that can be converted to dataloader later
        print(
            f'Loaded {len(self.point_cloud_paths)} point cloud samples belonging to {self.cur_num_classes} classes'
        )

    ################## Models Loading Related ##################
    # Initialize options for loading point cloud classification models.
    # This must be called after init_point_cloud_data
    def init_point_cloud_models(self):

        # model args (for now this is only for point transformer model)
        self.pt_model_cfg = {}
        self.pt_model_cfg['num_classes'] = self.cur_num_classes
        self.pt_model_cfg['num_blocks'] = 4
        self.pt_model_cfg['num_points'] = 1024
        self.pt_model_cfg['num_neighbors'] = 16
        self.pt_model_cfg['input_dim'] = 3
        self.pt_model_cfg['transformer_dim'] = 512

    def init_model_loading_interface(self):
        # load models interface
        # draw text
        model_selection_pixmap = QPixmap(450, 50)
        model_selection_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(model_selection_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 450, 50, QtGui.Qt.AlignLeft, 'Models in Comparisons: ')
        painter.end()
        # create label to contain the texts
        self.model_selection_label = QLabel(self)
        self.model_selection_label.setFixedSize(QtCore.QSize(500, 50))
        self.model_selection_label.setPixmap(model_selection_pixmap)
        self.model_selection_label.setContentsMargins(20, 0, 0, 0)

        # model 1
        # graphic representation
        self.model_1_label = QLabel(self)
        self.model_1_label.setContentsMargins(0, 0, 0, 0)
        self.model_1_label.setAlignment(QtCore.Qt.AlignCenter)
        model_1_icon = QPixmap(25, 25)
        model_1_icon.fill(QtCore.Qt.white)
        # draw model representation
        painter = QtGui.QPainter(model_1_icon)
        interface_util.draw_circle(painter, 12, 12, 10, 'blue')
        self.model_1_menu = QtWidgets.QComboBox()
        self.model_1_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        # model 1 names
        self.model_1_menu.setFixedSize(QtCore.QSize(200, 50))
        self.model_1_menu.addItem(model_1_icon, 'Original')
        self.model_1_menu.addItem(model_1_icon, 'Data Aug')
        self.model_1_menu.setCurrentText('Original')
        # connect the drop down menu with actions
        self.model_1_menu.currentTextChanged.connect(self._model_1_selection_changed)
        self.model_1_menu.setEditable(True)
        self.model_1_menu.lineEdit().setReadOnly(True)
        self.model_1_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        # preload model 1
        self.model_1_name = self.model_1_menu.currentText()
        self.model_1_cache_name = self.model_1_name.split(' ')[0]
        self.model_1 = self.load_point_cloud_model(self.model_1_name)

        # model 2
        # graphic representation
        self.model_2_label = QLabel(self)
        self.model_2_label.setContentsMargins(0, 0, 0, 0)
        self.model_2_label.setAlignment(QtCore.Qt.AlignCenter)
        model_2_icon = QPixmap(25, 25)
        model_2_icon.fill(QtCore.Qt.white)
        # draw model representation
        painter = QtGui.QPainter(model_2_icon)
        interface_util.draw_circle(painter, 12, 12, 10, 'magenta')
        self.model_2_menu = QtWidgets.QComboBox()
        self.model_2_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        self.model_2_menu.setFixedSize(QtCore.QSize(200, 50))
        self.model_2_menu.addItem(model_2_icon, 'Original')
        self.model_2_menu.addItem(model_2_icon, 'Data Aug')
        self.model_2_menu.setCurrentText('Data Aug')
        # connect the drop down menu with actions
        self.model_2_menu.currentTextChanged.connect(self._model_2_selection_changed)
        self.model_2_menu.setEditable(True)
        self.model_2_menu.lineEdit().setReadOnly(True)
        self.model_2_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        # preload model 2
        self.model_2_name = self.model_2_menu.currentText()
        self.model_2_cache_name = self.model_2_name.split(' ')[0]
        self.model_2 = self.load_point_cloud_model(self.model_2_name)

        # create model menu layout
        model_menus_layout = QtWidgets.QGridLayout()
        model_menus_layout.addWidget(self.model_1_menu, 0, 0)
        model_menus_layout.addWidget(self.model_2_menu, 0, 1)
        # add to demo layout
        self.interface_layout.addWidget(self.model_selection_label, 1, 2)
        self.interface_layout.addLayout(model_menus_layout, 2, 2)

    def load_point_cloud_model(self, model_name):
        # load the mode
        if model_name == 'Original':
            model_path = glob.glob(
                os.path.join(os.getcwd(), 'example_models', self.mode, 'non_eqv', '*.pth')
            )[0]
            # load model
            model = nero_run_model.load_model(self.mode, 'non-eqv', model_path, self.pt_model_cfg)
        elif model_name == 'Data Aug':
            model_path = glob.glob(
                os.path.join(os.getcwd(), 'example_models', self.mode, 'rot_eqv', '*.pth')
            )[0]
            # load model
            model = nero_run_model.load_model(self.mode, 'aug-eqv', model_path, self.pt_model_cfg)

        return model

    ################## General NERO Plots Settings Related ##################
    def init_general_control_interface(self):
        # drop down menu on selection which quantity to plot
        self.metric_layout = QtWidgets.QHBoxLayout()
        self.metric_layout.setContentsMargins(20, 0, 0, 0)
        # QPixmap that contains the title text
        metric_pixmap = QPixmap(300, 50)
        metric_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(metric_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 300, 50, QtGui.Qt.AlignLeft, 'NERO Metric: ')
        painter.end()
        # QLabel that contains the QPixmap
        self.metric_label = QLabel(self)
        self.metric_label.setFixedSize(QtCore.QSize(300, 50))
        self.metric_label.setPixmap(metric_pixmap)
        self.metric_label.setContentsMargins(0, 0, 0, 0)
        # create the drop down menu
        self.metric_menu = QtWidgets.QComboBox()
        self.metric_menu.setFixedSize(QtCore.QSize(220, 50))
        self.metric_menu.setStyleSheet(
            'color: black; font-family: Helvetica; font-style: normal; font-size: 34px'
        )
        self.metric_menu.setEditable(True)
        self.metric_menu.lineEdit().setReadOnly(True)
        self.metric_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.metric_menu.addItem('Instance Acc')
        self.metric_menu.addItem('Class Acc')
        # define default plotting quantity
        self.metric_menu.setCurrentIndex(0)
        self.cur_metric = self.metric_menu.currentText()
        # connect the drop down menu with actions
        self.metric_menu.currentTextChanged.connect(self._nero_metric_changed)
        # add both text and drop down menu to the layout
        self.metric_layout.addWidget(self.metric_label)
        self.metric_layout.addWidget(self.metric_menu)
        # add layout to the interface
        self.interface_layout.addLayout(self.metric_layout, 0, 2)

        # radio buttons on which plane we are rotating in
        self.plane_layout = QtWidgets.QHBoxLayout()
        self.plane_layout.setContentsMargins(0, 0, 0, 0)
        # QPixmap that contains the title text
        plane_pixmap = QPixmap(300, 50)
        plane_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(plane_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 300, 50, QtGui.Qt.AlignLeft, 'Slice Plane: ')
        painter.end()
        # QLabel that contains the QPixmap
        self.plane_label = QLabel(self)
        self.plane_label.setFixedSize(QtCore.QSize(300, 50))
        self.plane_label.setPixmap(plane_pixmap)
        self.plane_label.setContentsMargins(0, 0, 0, 0)
        # radio buttons
        # xy button
        self.xy_radio = QRadioButton('xy')
        self.xy_radio.setFixedSize(QtCore.QSize(160, 50))
        self.xy_radio.setContentsMargins(0, 0, 0, 0)
        self.xy_radio.setStyleSheet(
            'color: black; font-style: normal; font-family: Helvetica; font-size: 28px;'
        )
        self.xy_radio.pressed.connect(self._xy_plane_selected)
        # xz button
        self.xz_radio = QRadioButton('xz')
        self.xz_radio.setFixedSize(QtCore.QSize(160, 50))
        self.xz_radio.setContentsMargins(0, 0, 0, 0)
        self.xz_radio.setStyleSheet(
            'color: black; font-style: normal; font-family: Helvetica; font-size: 28px;'
        )
        self.xz_radio.pressed.connect(self._xz_plane_selected)
        # yz button
        self.yz_radio = QRadioButton('yz')
        self.yz_radio.setFixedSize(QtCore.QSize(160, 50))
        self.yz_radio.setContentsMargins(0, 0, 0, 0)
        self.yz_radio.setStyleSheet(
            'color: black; font-style: normal; font-family: Helvetica; font-size: 28px;'
        )
        self.yz_radio.pressed.connect(self._yz_plane_selected)
        # set xy selected by default
        self.xy_radio.setChecked(True)
        self.cur_plane = 'xy'
        # add both text and drop down menu to the layout
        self.plane_layout.addWidget(self.plane_label)
        self.plane_layout.addWidget(self.xy_radio)
        self.plane_layout.addWidget(self.xz_radio)
        self.plane_layout.addWidget(self.yz_radio)
        self.interface_layout.addLayout(self.plane_layout, 2, 0)

    ################## Aggregate NERO Plots Related ##################
    def prepare_aggregate_results(self):
        self.all_planes = ['xy', 'xz', 'yz']

        # axis angles
        self.all_axis_angles, successful = interface_util.load_from_cache(
            'all_axis_angles', self.cache
        )
        if not successful:
            self.all_axis_angles = list(range(-180, 181, 30))
            interface_util.save_to_cache(
                'all_axis_angles', self.all_axis_angles, self.cache, self.cache_path
            )

        # rotation angles
        self.all_rot_angles, successful = interface_util.load_from_cache(
            'all_rot_angles', self.cache
        )
        if not successful:
            self.all_rot_angles = list(range(0, 181, 30))
            interface_util.save_to_cache(
                'all_rot_angles', self.all_rot_angles, self.cache, self.cache_path
            )

        # aggregate test results for model 1
        self.all_avg_instance_accuracies_1, successful_avg_ins = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_instance_accuracies',
            self.cache,
        )
        self.all_avg_class_accuracies_1, successful_avg_cls = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_class_accuracies',
            self.cache,
        )
        self.all_avg_accuracies_per_class_1, successful_cls = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_accuracies_per_class',
            self.cache,
        )
        self.all_outputs_1, successful_output = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_outputs', self.cache
        )

        # if any of the result for model 1 is missing, run aggregate test
        if (
            not successful_avg_ins
            or not successful_avg_cls
            or not successful_cls
            or not successful_output
        ):
            print(f'\nRunning aggregate test for model 1')
            (
                self.all_avg_instance_accuracy_1,
                self.all_avg_class_accuracies_1,
                self.all_avg_accuracies_per_class_1,
                self.all_outputs_1,
            ) = nero_run_model.run_point_cloud(
                self.model_1,
                self.cur_num_classes,
                self.cur_name_path,
                self.point_cloud_paths,
                self.all_planes,
                self.all_axis_angles,
                self.all_rot_angles,
            )

            # save to cache
            interface_util.save_to_cache(
                [
                    f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_instance_accuracies',
                    f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_class_accuracies',
                    f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_accuracies_per_class',
                    f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_outputs',
                ],
                [
                    self.all_avg_instance_accuracy_1,
                    self.all_avg_class_accuracies_1,
                    self.all_avg_accuracies_per_class_1,
                    self.all_outputs_1,
                ],
                self.cache,
                self.cache_path,
            )

        # aggregate test results for model 2
        self.all_avg_instance_accuracies_2, successful = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_instance_accuracies',
            self.cache,
        )
        self.all_avg_class_accuracies_2, successful = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_class_accuracies',
            self.cache,
        )
        self.all_avg_accuracies_per_class_2, successful = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_accuracies_per_class',
            self.cache,
        )
        self.all_outputs_2, successful = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_outputs', self.cache
        )

        # if any of the result for model 1 is missing, run aggregate test
        if not successful:
            print(f'\nRunning aggregate test for model 2')
            (
                self.all_avg_instance_accuracy_2,
                self.all_avg_class_accuracies_2,
                self.all_avg_accuracies_per_class_2,
                self.all_outputs_2,
            ) = nero_run_model.run_point_cloud(
                self.model_2,
                self.cur_num_classes,
                self.cur_name_path,
                self.point_cloud_paths,
                self.all_planes,
                self.all_axis_angles,
                self.all_rot_angles,
            )

            # save to cache
            interface_util.save_to_cache(
                [
                    f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_instance_accuracies',
                    f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_class_accuracies',
                    f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_accuracies_per_class',
                    f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_outputs',
                ],
                [
                    self.all_avg_instance_accuracy_2,
                    self.all_avg_class_accuracies_2,
                    self.all_avg_accuracies_per_class_2,
                    self.all_outputs_2,
                ],
                self.cache,
                self.cache_path,
            )

    def init_aggregate_plot_interface(self):
        # display in heatmap
        # heatmap view
        self.aggregate_heatmap_view_1 = pg.GraphicsLayoutWidget()
        self.aggregate_heatmap_view_1.ci.layout.setContentsMargins(
            0, 0, 0, 0
        )  # left top right bottom
        self.aggregate_heatmap_view_1.setFixedSize(self.plot_size * 1.35, self.plot_size * 1.35)

        self.aggregate_heatmap_view_2 = pg.GraphicsLayoutWidget()
        self.aggregate_heatmap_view_2.ci.layout.setContentsMargins(
            0, 0, 0, 0
        )  # left top right bottom
        self.aggregate_heatmap_view_2.setFixedSize(self.plot_size * 1.35, self.plot_size * 1.35)
        # draw the plot
        # cm_range, color map and color bar will be shared with DR and individual NERO plot
        self.cm_range = (0, 1)
        self.color_map = pg.colormap.get('viridis')
        self.color_bar = pg.ColorBarItem(
            values=self.cm_range,
            colorMap=self.color_map,
            interactive=False,
            orientation='horizontal',
            width=30,
        )

        # add view to layout
        self.interface_layout.addWidget(self.aggregate_heatmap_view_1, 3, 0, 3, 1)
        self.interface_layout.addWidget(self.aggregate_heatmap_view_2, 5, 0, 3, 1)

    def draw_point_cloud_aggregate_nero(self):
        # determine the plane index
        self.cur_plane_index = self.all_planes.index(self.cur_plane)
        # select the data that we are using to draw
        if self.class_selection == 'all':
            if self.cur_metric == 'Instance Acc':
                self.cur_aggregate_plot_quantity_1 = self.all_avg_instance_accuracies_1[
                    self.cur_plane_index
                ]
                self.cur_aggregate_plot_quantity_2 = self.all_avg_instance_accuracies_2[
                    self.cur_plane_index
                ]
            elif self.cur_metric == 'Class Acc':
                self.cur_aggregate_plot_quantity_1 = self.all_avg_class_accuracies_1[
                    self.cur_plane_index
                ]
                self.cur_aggregate_plot_quantity_2 = self.all_avg_class_accuracies_2[
                    self.cur_plane_index
                ]
        # when we are selecting a specific class
        else:
            # when a specific class is chosen, Instance Acc is enforced
            self.cur_metric = 'Instance Acc'
            self.metric_menu.setCurrentText(self.cur_metric)
            self.cur_aggregate_plot_quantity_1 = self.all_avg_accuracies_per_class_1[
                self.cur_plane_index, :, :, self.cur_classes_names.index(self.class_selection)
            ]
            self.cur_aggregate_plot_quantity_2 = self.all_avg_accuracies_per_class_2[
                self.cur_plane_index, :, :, self.cur_classes_names.index(self.class_selection)
            ]

        # size of each block (rectangle)
        self.block_size = 50
        # convert the result to polar plot data fit in rectangular array
        # model 1 results
        (
            self.processed_aggregate_quantity_1,
            self.processed_aggregate_quaternion_1,
        ) = interface_util.process_point_cloud_result(
            self.cur_aggregate_plot_quantity_1,
            self.cur_plane,
            self.all_axis_angles,
            self.all_rot_angles,
            block_size=self.block_size,
        )
        # model 2 results
        (
            self.processed_aggregate_quantity_2,
            self.processed_aggregate_quaternion_2,
        ) = interface_util.process_point_cloud_result(
            self.cur_aggregate_plot_quantity_2,
            self.cur_plane,
            self.all_axis_angles,
            self.all_rot_angles,
            block_size=self.block_size,
        )
        # draw the heatmap
        self.aggregate_heatmap_plot_1 = interface_util.draw_individual_heatmap(
            self.processed_aggregate_quantity_1, self.color_bar
        )
        self.aggregate_heatmap_plot_2 = interface_util.draw_individual_heatmap(
            self.processed_aggregate_quantity_2, self.color_bar
        )
        # add plot to view
        self.aggregate_heatmap_view_1.clear()
        self.aggregate_heatmap_view_1.addItem(self.aggregate_heatmap_plot_1)
        self.aggregate_heatmap_view_2.clear()
        self.aggregate_heatmap_view_2.addItem(self.aggregate_heatmap_plot_2)

    ################## DR Plots Related ##################
    def prepare_dr_results(self):
        self.all_dr_results_1 = {}
        self.all_dr_results_2 = {}
        high_dim_points_constructed_1 = False
        high_dim_points_constructed_2 = False
        self.all_dr_algorithms = ['PCA', 'ICA', 'ISOMAP', 't-SNE', 'UMAP']
        # iteracte through each dr method
        for cur_algo in self.all_dr_algorithms:
            # for model 1
            self.all_high_dim_points_1, successful_high = interface_util.load_from_cache(
                f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_high_dim',
                self.cache,
            )
            self.all_dr_results_1[cur_algo], successful_low = interface_util.load_from_cache(
                f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_{cur_algo}_low_dim',
                self.cache,
            )
            # when we don't have in the cache
            if not successful_high or not successful_low:
                if not high_dim_points_constructed_1:
                    # construct high dimension data first
                    self.all_high_dim_points_1 = np.zeros(
                        (
                            len(self.point_cloud_paths),
                            len(self.all_planes)
                            * len(self.all_axis_angles)
                            * len(self.all_rot_angles),
                        )
                    )
                    # get the pred classification results under all rotations for each sample
                    all_results_1 = self.all_outputs_1.reshape(
                        (
                            len(self.point_cloud_paths),
                            len(self.all_planes)
                            * len(self.all_axis_angles)
                            * len(self.all_rot_angles),
                            self.cur_num_classes,
                        )
                    )
                    for i in range(len(self.point_cloud_paths)):
                        cur_ground_truth_index = self.cur_classes_names.index(
                            self.point_cloud_paths[i][0]
                        )
                        self.all_high_dim_points_1[i] = all_results_1[i, :, cur_ground_truth_index]

                    high_dim_points_constructed_1 = True
                    interface_util.save_to_cache(
                        f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_high_dim',
                        self.all_high_dim_points_1,
                        self.cache,
                        self.cache_path,
                    )

                # compute the dr results from model outputs
                self.all_dr_results_1[cur_algo] = interface_util.dimension_reduce(
                    cur_algo, self.all_high_dim_points_1, 2
                )
                interface_util.save_to_cache(
                    f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_{cur_algo}_low_dim',
                    self.all_dr_results_1[cur_algo],
                    self.cache,
                    self.cache_path,
                )

            # for model 2
            self.all_high_dim_points_2, successful_high = interface_util.load_from_cache(
                f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_high_dim',
                self.cache,
            )
            self.all_dr_results_2[cur_algo], successful_low = interface_util.load_from_cache(
                f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_{cur_algo}_low_dim',
                self.cache,
            )

            # when we don't have in the cache
            if not successful_high or not successful_low:
                if not high_dim_points_constructed_2:
                    # construct high dimension data first
                    self.all_high_dim_points_2 = np.zeros(
                        (
                            len(self.point_cloud_paths),
                            len(self.all_planes)
                            * len(self.all_axis_angles)
                            * len(self.all_rot_angles),
                        )
                    )
                    # get the pred classification results under all rotations for each sample
                    all_results_2 = self.all_outputs_2.reshape(
                        (
                            len(self.point_cloud_paths),
                            len(self.all_planes)
                            * len(self.all_axis_angles)
                            * len(self.all_rot_angles),
                            self.cur_num_classes,
                        )
                    )
                    for i in range(len(self.point_cloud_paths)):
                        cur_ground_truth_index = self.cur_classes_names.index(
                            self.point_cloud_paths[i][0]
                        )
                        self.all_high_dim_points_2[i] = all_results_2[i, :, cur_ground_truth_index]

                    high_dim_points_constructed_2 = True
                    interface_util.save_to_cache(
                        f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_high_dim',
                        self.all_high_dim_points_2,
                        self.cache,
                        self.cache_path,
                    )

                # compute the dr results from model outputs
                self.all_dr_results_2[cur_algo] = interface_util.dimension_reduce(
                    cur_algo, self.all_high_dim_points_2, 2
                )
                interface_util.save_to_cache(
                    f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_{cur_algo}_low_dim',
                    self.all_dr_results_2[cur_algo],
                    self.cache,
                    self.cache_path,
                )

    def init_dr_plot_interface(self):
        # data indices that we are plotting
        self.cur_class_indices = []
        if self.class_selection == 'all':
            self.cur_class_indices = list(range(len(self.point_cloud_paths)))
        else:
            for i in range(len(self.point_cloud_paths)):
                if self.point_cloud_paths[i][0] == self.class_selection:
                    self.cur_class_indices.append(i)

        # drop down menu on choosing the dimension reduction method
        # QPixmap that contains the title text
        dr_selection_pixmap = QPixmap(330, 60)
        dr_selection_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(dr_selection_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 330, 60, QtGui.Qt.AlignLeft, 'DR Plot:')
        painter.end()
        # QLabel that contains the QPixmap
        self.dr_selection_label = QLabel(self)
        self.dr_selection_label.setFixedSize(QtCore.QSize(330, 60))
        self.dr_selection_label.setPixmap(dr_selection_pixmap)
        # add to the layout
        dr_selection_layout = QtWidgets.QHBoxLayout()
        dr_selection_layout.addWidget(self.dr_selection_label)
        # create the drop down menu
        self.dr_selection_menu = QtWidgets.QComboBox()
        self.dr_selection_menu.setFixedSize(QtCore.QSize(150, 50))
        self.dr_selection_menu.setContentsMargins(0, 0, 0, 0)
        self.dr_selection_menu.setStyleSheet(
            'color: black; font-family: Helvetica; font-style: normal; font-size: 34px'
        )
        for cur_algo in self.all_dr_algorithms:
            self.dr_selection_menu.addItem(f'{cur_algo}')
        # set default to digit 0, which means PCA
        self.dr_selection_menu.setCurrentIndex(0)
        self.cur_dr_algorithm = self.dr_selection_menu.currentText()
        # connect the drop down menu with actions
        self.dr_selection_menu.currentTextChanged.connect(self._dr_selection_changed)
        self.dr_selection_menu.setEditable(True)
        self.dr_selection_menu.lineEdit().setReadOnly(True)
        self.dr_selection_menu.lineEdit().setAlignment(QtCore.Qt.AlignRight)
        # add to local layout
        dr_selection_layout.addWidget(self.dr_selection_menu)
        self.interface_layout.addLayout(dr_selection_layout, 0, 1, 1, 1)

        # radio buttons on using mean or variance for color-encoding
        # Title on the two radio buttons
        intensity_button_pixmap = QPixmap(300, 60)
        intensity_button_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(intensity_button_pixmap)
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
        self.scatterplot_sorting_layout = QtWidgets.QGridLayout()
        # the title occupies two rows because we have two selections (mean and variance)
        self.scatterplot_sorting_layout.addWidget(intensity_button_label, 0, 0, 2, 1)
        # mean button
        self.mean_intensity_button = QRadioButton('Mean')
        self.mean_intensity_button.setFixedSize(QtCore.QSize(160, 50))
        self.mean_intensity_button.setContentsMargins(0, 0, 0, 0)
        self.mean_intensity_button.setStyleSheet(
            'color: black; font-style: normal; font-family: Helvetica; font-size: 28px;'
        )
        self.mean_intensity_button.pressed.connect(self._mean_intensity_button_clicked)
        self.scatterplot_sorting_layout.addWidget(self.mean_intensity_button, 1, 1, 1, 1)
        # variance button
        self.variance_intensity_button = QRadioButton('Variance')
        self.variance_intensity_button.setFixedSize(QtCore.QSize(160, 50))
        self.variance_intensity_button.setContentsMargins(0, 0, 0, 0)
        self.variance_intensity_button.setStyleSheet(
            'color: black; font-style: normal; font-family: Helvetica; font-size: 28px;'
        )
        self.variance_intensity_button.pressed.connect(self._variance_intensity_button_clicked)
        self.scatterplot_sorting_layout.addWidget(self.variance_intensity_button, 2, 1, 1, 1)
        # self.scatterplot_sorting_layout.setContentsMargins(0, 0, 30, 0)
        self.interface_layout.addLayout(self.scatterplot_sorting_layout, 1, 1, 2, 1)

        # by default the intensities are computed via mean
        self.mean_intensity_button.setChecked(True)
        self.intensity_method = 'mean'

        # initialize all the views when the first time
        # scatter item size
        self.scatter_item_size = 12
        # dr plot for model 1
        self.low_dim_scatter_view_1 = pg.GraphicsLayoutWidget()
        self.low_dim_scatter_view_1.setBackground('white')
        self.low_dim_scatter_view_1.setFixedSize(self.plot_size * 1.3, self.plot_size * 1.3)
        self.low_dim_scatter_view_1.ci.setContentsMargins(20, 0, 0, 0)
        # add plot
        self.low_dim_scatter_plot_1 = self.low_dim_scatter_view_1.addPlot()
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

        # dr plot for model 2
        self.low_dim_scatter_view_2 = pg.GraphicsLayoutWidget()
        self.low_dim_scatter_view_2.setBackground('white')
        self.low_dim_scatter_view_2.setFixedSize(self.plot_size * 1.25, self.plot_size * 1.25)
        self.low_dim_scatter_view_2.ci.setContentsMargins(20, 0, 0, 0)
        # add plot
        self.low_dim_scatter_plot_2 = self.low_dim_scatter_view_2.addPlot()
        self.low_dim_scatter_plot_2.hideAxis('left')
        self.low_dim_scatter_plot_2.hideAxis('bottom')
        # set axis range
        self.low_dim_scatter_plot_2.setXRange(-1.2, 1.2, padding=0)
        self.low_dim_scatter_plot_2.setYRange(-1.2, 1.2, padding=0)
        # Not letting user zoom out past axis limit
        self.low_dim_scatter_plot_2.vb.setLimits(xMin=-1.2, xMax=1.2, yMin=-1.2, yMax=1.2)

        # same colorbar in dr plot as used in aggregate and individual NERO plot
        color_bar_view = pg.GraphicsLayoutWidget()
        color_bar_plot = pg.PlotItem()
        color_bar_plot.layout.setContentsMargins(0, 0, 0, 0)
        color_bar_plot.setFixedHeight(0)
        color_bar_plot.setFixedWidth(self.plot_size * 1.2)
        color_bar_plot.hideAxis('bottom')
        color_bar_plot.hideAxis('left')
        color_bar_view.addItem(color_bar_plot)
        color_bar_image = pg.ImageItem()
        self.color_bar.setImageItem(color_bar_image, insert_in=color_bar_plot)
        self.scatterplot_sorting_layout.addWidget(color_bar_view, 3, 0, 1, 2)
        color_bar_plot.layout.setContentsMargins(50, 0, 0, 0)

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
            self._dr_result_selection_slider_1_changed
        )
        self.slider_1_layout.addWidget(self.dr_result_selection_slider_1, 0, 0, 1, 3)
        # left and right buttons to move the slider around, with number in the middle
        # left button
        self.slider_1_left_button = QtWidgets.QToolButton()
        self.slider_1_left_button.setArrowType(QtCore.Qt.LeftArrow)
        self.slider_1_left_button.clicked.connect(self._slider_1_left_button_clicked)
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
        self.slider_1_right_button.clicked.connect(self._slider_1_right_button_clicked)
        self.slider_1_layout.addWidget(self.slider_1_right_button, 1, 2, 1, 1)
        # initialize slider selection
        self.slider_1_selected_index = None
        # add slider 1 layout to the general layout
        self.interface_layout.addLayout(self.slider_1_layout, 4, 1, 1, 1)
        self.slider_1_layout.setContentsMargins(40, 0, 0, 0)

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
            self._dr_result_selection_slider_2_changed
        )
        self.slider_2_layout.addWidget(self.dr_result_selection_slider_2, 0, 0, 1, 3)
        # left and right buttons to move the slider around, with number in the middle
        # left button
        self.slider_2_left_button = QtWidgets.QToolButton()
        self.slider_2_left_button.setArrowType(QtCore.Qt.LeftArrow)
        self.slider_2_left_button.setFixedSize(30, 30)
        self.slider_2_left_button.setStyleSheet('color: black')
        self.slider_2_left_button.clicked.connect(self._slider_2_left_button_clicked)
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
        self.slider_2_right_button.clicked.connect(self._slider_2_right_button_clicked)
        self.slider_2_layout.addWidget(self.slider_2_right_button, 1, 2, 1, 1)
        # initialize slider selection
        self.slider_2_selected_index = None
        # add slider 2 layout to the general layout
        self.interface_layout.addLayout(self.slider_2_layout, 6, 1, 1, 1)
        self.slider_2_layout.setContentsMargins(40, 0, 0, 0)

    def draw_dr_plot(self):

        # get the dimension reduced points
        self.low_dim_1 = self.all_dr_results_1[self.cur_dr_algorithm]
        self.low_dim_2 = self.all_dr_results_2[self.cur_dr_algorithm]

        # use each sample's metric average or variance across all transformations as intensity
        self.all_intensity_1 = interface_util.compute_intensity(
            self.all_high_dim_points_1, self.intensity_method
        )
        self.all_intensity_2 = interface_util.compute_intensity(
            self.all_high_dim_points_2, self.intensity_method
        )

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

        self._draw_scatter_plot(
            self.low_dim_scatter_plot_1,
            self.low_dim_1,
            self.sorted_intensity_1,
            self.sorted_class_indices_1,
            self.slider_1_selected_index,
        )

        self._draw_scatter_plot(
            self.low_dim_scatter_plot_2,
            self.low_dim_2,
            self.sorted_intensity_2,
            self.sorted_class_indices_2,
            self.slider_2_selected_index,
        )

    ################## Individual NERO Plot Related ##################

    ################## Detail Plot Related ##################

    ################## All closed functions ##################
    # dataset drop-down menu
    @QtCore.Slot()
    def _dataset_selection_changed(self, text):
        # filter out 0 selection signal
        if text == 'Input dataset':
            return

        self.dataset_name = text
        self.dataset_index = self.aggregate_image_menu.currentIndex()

        # re-load the data
        self.load_point_cloud_data()

        # the models have default, just run
        self.run_button_clicked()

    # dataset class selection drop-down menu
    @QtCore.Slot()
    def _class_selection_changed(self, text):
        # re-initialize the scatter plot
        self.dr_result_existed = False

        # for object detection (COCO)
        if text.split(' ')[0] == 'All':
            self.class_selection = 'all'
        else:
            self.class_selection = text

        # get the data indices that the user selects
        self.cur_class_indices = []
        if self.class_selection == 'all':
            self.cur_class_indices = list(range(len(self.point_cloud_paths)))
        else:
            for i in range(len(self.point_cloud_paths)):
                if self.point_cloud_paths[i][0] == self.class_selection:
                    self.cur_class_indices.append(i)

        # display the plot
        self.draw_point_cloud_aggregate_nero()

        # # after change class, run new dimension reduction if previously run
        # if self.demo or self.dr_result_existed:
        #     self.run_dimension_reduction()

    # drop down menu that let user select model 1
    @QtCore.Slot()
    def _model_1_selection_changed(self, text):
        print('Model 1:', text)
        self.model_1_name = text
        # Original or DA
        self.model_1_cache_name = self.model_1_name.split(' ')[0]

        # load the model
        self.model_1 = self.load_point_cloud_model(self.model_1_name)

        # # when loaded data is available, just show the result without clicking the button
        # self.run_model_aggregated()
        # self.aggregate_result_existed = True

        # # run dimension reduction if previously run
        # if self.dr_result_existed:
        #     self.run_dimension_reduction()

    # drop down menu that lets user select model 2
    @QtCore.Slot()
    def _model_2_selection_changed(self, text):
        print('Model 2:', text)
        self.model_2_name = text
        # Original or DA
        self.model_2_cache_name = self.model_2_name.split(' ')[0]

        # load the model
        self.model_2 = self.load_point_cloud_model(self.model_2_name)

        # # when loaded data is available, just show the result without clicking the button
        # self.run_model_aggregated()
        # self.aggregate_result_existed = True

        # # run dimension reduction if previously run
        # if self.dr_result_existed:
        #     self.run_dimension_reduction()

    # drop down menu that lets user select NERO metric
    @QtCore.Slot()
    def _nero_metric_changed(self, text):
        print(f'\nNERO metric changed to {text}')
        # save the selection
        self.cur_metric = text
        # plot
        self.draw_point_cloud_aggregate_nero()
        # re-display DR plot
        self.draw_dr_plot()

    @QtCore.Slot()
    def _xy_plane_selected(self):
        print(f'xy plane')
        self.cur_plane = 'xy'
        # re-draw aggregate nero plot
        self.draw_point_cloud_aggregate_nero()
        # re-display DR plot
        self.draw_dr_plot()

    @QtCore.Slot()
    def _xz_plane_selected(self):
        print(f'xz plane')
        self.cur_plane = 'xz'
        # re-draw aggregate nero plot
        self.draw_point_cloud_aggregate_nero()
        # re-display DR plot
        self.draw_dr_plot()

    @QtCore.Slot()
    def _yz_plane_selected(self):
        print(f'yz plane')
        self.cur_plane = 'yz'
        # re-draw aggregate nero plot
        self.draw_point_cloud_aggregate_nero()
        # re-display DR plot
        self.draw_dr_plot()

    @QtCore.Slot()
    # change different dimension reduction algorithms
    def _dr_selection_changed(self, text):
        # update dimension reduction algorithm
        self.cur_dr_algorithm = text

        # re-display dr show result
        self.draw_dr_plot()

    # radio buttons on choosing quantity used to compute intensity
    @QtCore.Slot()
    def _mean_intensity_button_clicked(self):
        self.intensity_method = 'mean'
        print(f'DR plots color encoded based on {self.intensity_method}')
        self.all_intensity_1 = np.mean(self.all_high_dim_points_1, axis=1)
        self.all_intensity_2 = np.mean(self.all_high_dim_points_2, axis=1)

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
        self.draw_dr_plot()

    @QtCore.Slot()
    def _variance_intensity_button_clicked(self):
        self.intensity_method = 'variance'
        print(f'DR plots color encoded based on {self.intensity_method}')
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
        self.draw_dr_plot()

    # update the slider 1's text
    @QtCore.Slot()
    def _update_slider_1_text(self):
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
    @QtCore.Slot()
    def _update_slider_2_text(self):
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

    # slider for dr plot 1
    @QtCore.Slot()
    def _dr_result_selection_slider_1_changed(self):

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
            self._update_slider_1_text()

            # change the other slider's value
            self.slider_2_locked = True
            self.slider_2_selected_index = self.sorted_class_indices_2.index(self.image_index)
            self.dr_result_selection_slider_2.setValue(self.slider_2_selected_index)
            # update the text
            self._update_slider_2_text()
            self.slider_2_locked = False

            # update the scatter plot without re-computing dimension reduction algorithm
            self.draw_dr_plot()

            # get the corresponding point cloud data path
            if self.mode == 'digit_recognition' or self.mode == 'object_detection':
                self.cur_point_cloud_path = self.point_cloud_paths[self.image_index][1]
                print(f'Selected image at {self.image_path}')

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

    # slider for dr plot 2
    @QtCore.Slot()
    def _dr_result_selection_slider_2_changed(self):

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
            self._update_slider_2_text()

            # change the other slider's value
            self.slider_1_locked = True
            self.slider_1_selected_index = self.sorted_class_indices_1.index(self.image_index)
            self.dr_result_selection_slider_1.setValue(self.slider_1_selected_index)
            # update the text
            self._update_slider_1_text()
            self.slider_1_locked = False

            # update the scatter plot
            self.draw_dr_plot()

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

    @QtCore.Slot()
    def _slider_1_left_button_clicked(self):
        self.dr_result_selection_slider_1.setValue(self.dr_result_selection_slider_1.value() - 1)
        # update the text
        self._update_slider_1_text()

    @QtCore.Slot()
    def _slider_1_right_button_clicked(self):
        self.dr_result_selection_slider_1.setValue(self.dr_result_selection_slider_1.value() + 1)
        # update the text
        self._update_slider_1_text()

    @QtCore.Slot()
    def _slider_2_left_button_clicked(self):
        self.dr_result_selection_slider_2.setValue(self.dr_result_selection_slider_2.value() - 1)
        # update the text
        self._update_slider_2_text()

    @QtCore.Slot()
    def _slider_2_right_button_clicked(self):
        self.dr_result_selection_slider_2.setValue(self.dr_result_selection_slider_2.value() + 1)
        # update the text
        self._update_slider_2_text()

    # plot all the scatter items with brush color reflecting the intensity
    def _draw_scatter_plot(
        self,
        low_dim_scatter_plot,
        low_dim,
        sorted_intensity,
        sorted_class_indices,
        slider_selected_index,
    ):

        # quantize all the intensity into color
        color_indices = []
        scatter_lut = self.color_map.getLookupTable(
            start=self.cm_range[0], stop=self.cm_range[1], nPts=500, alpha=False
        )
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
            low_dim_scatter_item.sigClicked.connect(self._low_dim_scatter_clicked)
            low_dim_scatter_item.sigHovered.connect(self._low_dim_scatter_hovered)
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
        low_dim_scatter_item.sigClicked.connect(self._low_dim_scatter_clicked)
        low_dim_scatter_item.sigHovered.connect(self._low_dim_scatter_hovered)
        # add points to the plot
        low_dim_scatter_plot.addItem(low_dim_scatter_item)

    # when clicked on the scatter plot item
    @QtCore.Slot()
    def _low_dim_scatter_clicked(self, item=None, points=None):
        # get the clicked scatter item's information
        # when item is not none, it is from real click
        if item != None:
            self.point_cloud_index = int(item.opts['name'])
            print(f'clicked image index {self.point_cloud_index}')
        # when the input is empty, it is called automatically
        else:
            # image index should be defined
            if self.point_cloud_index == None:
                raise Exception(
                    'point_cloud_index should be defined prior to calling run_dimension_reduction'
                )

        # get the ranking in each colorbar and change its value while locking both sliders
        # slider 1
        self.slider_1_locked = True
        self.slider_2_locked = True
        self.slider_1_selected_index = self.sorted_class_indices_1.index(self.image_index)
        self.dr_result_selection_slider_1.setValue(self.slider_1_selected_index)
        # update the text
        self._update_slider_1_text()
        # slider 2
        self.slider_2_selected_index = self.sorted_class_indices_2.index(self.image_index)
        self.dr_result_selection_slider_2.setValue(self.slider_2_selected_index)
        # update the text
        self._update_slider_2_text()
        # update the indicator of current selected item
        self.draw_dr_plot()
        # unlock after changing the values
        self.slider_1_locked = False
        self.slider_2_locked = False

        # get the corresponding point cloud path
        self.point_cloud_path = self.point_cloud_paths[self.point_cloud_index]
        print(f'Selected point cloud at {self.point_cloud_path}')

        # load the image
        # self.load_single_image()

        # display individual view
        # TODO: visualize point cloud input

        # TODO: update Individual NERO plot
        # self.run_model_single()

    # when hovered on the scatter plot item
    def _low_dim_scatter_hovered(self, item, points):
        item.setToolTip(item.opts['hover_text'])


if __name__ == '__main__':

    # input arguments
    parser = argparse.ArgumentParser()
    # mode (digit_recognition, object_detection or piv)
    parser.add_argument('--mode', action='store', nargs=1, dest='mode')
    parser.add_argument('--cache_path', action='store', nargs=1, dest='cache_path')
    args = parser.parse_args()
    if args.mode:
        mode = args.mode[0]
    else:
        mode = None
    if args.cache_path:
        cache_path = args.cache_path[0]
    else:
        cache_path = None

    # initialize the app
    app = QtWidgets.QApplication([])
    widget = UI_MainWindow(cache_path)
    widget.show()

    # run the app
    app.exec()

    # exit the app
    sys.exit()
