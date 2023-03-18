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

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import manifold
from sklearn.manifold import TSNE
import umap

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

        # use in some paths
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

        # Plots section
        # prepare results NERO test results
        self.prepare_aggregate_results()
        # Aggregate NERO plot
        self.init_aggregate_plot_interface()
        self.draw_point_cloud_aggregate_nero()
        # DR plot
        self.prepare_dr_results()
        self.init_dr_plot_interface()

        print(f'\nFinished rendering main layout')

    def init_point_cloud_interface(self):

        # data loading interface
        self.init_data_loading_interface()

        # model loading interface
        self.init_model_loading_interface()

        # input display interface

        # aggregate NERO plots interface

        # DR plots interface

        # individual NERO plots interface

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
        # load aggregate dataset drop-down menu
        @QtCore.Slot()
        def dataset_selection_changed(text):
            # filter out 0 selection signal
            if text == 'Input dataset':
                return

            self.dataset_name = text
            self.dataset_index = self.aggregate_image_menu.currentIndex()

            # re-load the data
            self.load_point_cloud_data()

            # the models have default, just run
            self.run_button_clicked()

        # draw text regarding datasets
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

        # aggregate images loading drop down menu
        self.aggregate_image_menu = QtWidgets.QComboBox()
        self.aggregate_image_menu.setFixedSize(QtCore.QSize(220, 50))
        self.aggregate_image_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        self.aggregate_image_menu.addItem('Input dataset')

        # load all images in the folder
        for i in range(len(self.all_data_paths)):
            cur_name = self.all_data_paths[i].split('/')[-1].split('.')[0]
            # cur_name = self.all_data_paths[i].split('/')[-1].split('.')[0].split('_')[0]
            self.aggregate_image_menu.addItem(cur_name)

        # set default data selection
        self.aggregate_image_menu.setCurrentIndex(self.dataset_index)
        self.dataset_name = self.aggregate_image_menu.currentText()

        # load default dataset
        self.load_point_cloud_data()

        # connect the drop down menu with actions
        self.aggregate_image_menu.currentTextChanged.connect(dataset_selection_changed)
        self.aggregate_image_menu.setEditable(True)
        self.aggregate_image_menu.lineEdit().setReadOnly(True)
        self.aggregate_image_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        aggregate_image_menu_layout = QtWidgets.QHBoxLayout()
        aggregate_image_menu_layout.setContentsMargins(150, 0, 0, 0)
        aggregate_image_menu_layout.addWidget(self.aggregate_image_menu)
        self.interface_layout.addLayout(aggregate_image_menu_layout, 0, 0)

    def load_point_cloud_data(self):

        # get data and classes names path from selected 1-based index
        self.cur_data_path = self.all_data_paths[self.dataset_index - 1]
        self.cur_name_path = self.all_names_paths[self.dataset_index - 1]
        print(f'\nLoading data from {self.cur_data_path}')
        # load all the point cloud names
        point_cloud_ids = [line.rstrip() for line in open(self.cur_data_path)]
        # point cloud ids have name_index format
        point_cloud_names = ['_'.join(x.split('_')[0:-1]) for x in point_cloud_ids]
        # all the point cloud samples paths of the current dataset
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

        # dataset that can be converted to dataloader later
        self.data_existed = True
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

        # two drop down menus that let user choose models
        @QtCore.Slot()
        def model_1_selection_changed(text):
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

        @QtCore.Slot()
        def model_2_selection_changed(text):
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
        self.model_1_menu.currentTextChanged.connect(model_1_selection_changed)
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
        self.model_2_menu.currentTextChanged.connect(model_2_selection_changed)
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
        @QtCore.Slot()
        def nero_metric_changed(text):
            print(f'\nNERO metric changed to {text}')
            # save the selection
            self.cur_metric = text

            # plot
            self.draw_point_cloud_aggregate_nero()

            # re-display DR plot
            self.draw_dr_plot()

        @QtCore.Slot()
        def xy_plane_selected():
            print(f'xy plane')
            self.cur_plane = 'xy'
            # re-draw aggregate nero plot
            self.draw_point_cloud_aggregate_nero()

        @QtCore.Slot()
        def xz_plane_selected():
            print(f'xz plane')
            self.cur_plane = 'xz'
            # re-draw aggregate nero plot
            self.draw_point_cloud_aggregate_nero()

        @QtCore.Slot()
        def yz_plane_selected():
            print(f'yz plane')
            self.cur_plane = 'yz'
            # re-draw aggregate nero plot
            self.draw_point_cloud_aggregate_nero()

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
        metric_menu = QtWidgets.QComboBox()
        metric_menu.setFixedSize(QtCore.QSize(220, 50))
        metric_menu.setStyleSheet(
            'color: black; font-family: Helvetica; font-style: normal; font-size: 34px'
        )
        metric_menu.setEditable(True)
        metric_menu.lineEdit().setReadOnly(True)
        metric_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        metric_menu.addItem('Instance Acc')
        metric_menu.addItem('Class Acc')
        # define default plotting quantity
        metric_menu.setCurrentIndex(0)
        self.cur_metric = metric_menu.currentText()
        # connect the drop down menu with actions
        metric_menu.currentTextChanged.connect(nero_metric_changed)
        # add both text and drop down menu to the layout
        self.metric_layout.addWidget(self.metric_label)
        self.metric_layout.addWidget(metric_menu)
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
        self.xy_radio.pressed.connect(xy_plane_selected)
        # xz button
        self.xz_radio = QRadioButton('xz')
        self.xz_radio.setFixedSize(QtCore.QSize(160, 50))
        self.xz_radio.setContentsMargins(0, 0, 0, 0)
        self.xz_radio.setStyleSheet(
            'color: black; font-style: normal; font-family: Helvetica; font-size: 28px;'
        )
        self.xz_radio.pressed.connect(xz_plane_selected)
        # yz button
        self.yz_radio = QRadioButton('yz')
        self.yz_radio.setFixedSize(QtCore.QSize(160, 50))
        self.yz_radio.setContentsMargins(0, 0, 0, 0)
        self.yz_radio.setStyleSheet(
            'color: black; font-style: normal; font-family: Helvetica; font-size: 28px;'
        )
        self.yz_radio.pressed.connect(yz_plane_selected)
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
        # TODO: create user interface for selecting planes
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
        self.all_avg_instance_accuracies_1, successful = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_instance_accuracies',
            self.cache,
        )
        self.all_avg_class_accuracies_1, successful = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_class_accuracies',
            self.cache,
        )
        self.all_avg_accuracies_per_class_1, successful = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_accuracies_per_class',
            self.cache,
        )
        self.all_outputs_1, successful = interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_outputs', self.cache
        )

        # if any of the result for model 1 is missing, run aggregate test
        if not successful:
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
        # aggregate class selection drop-down menu
        @QtCore.Slot()
        def aggregate_class_selection_changed(text):
            # re-initialize the scatter plot
            self.dr_result_existed = False

            # for object detection (COCO)
            if text.split(' ')[0] == 'All':
                self.class_selection = 'all'
            else:
                self.class_selection = text

            # # display the plot
            # self.display_coco_aggregate_result()

            # # after change class, run new dimension reduction if previously run
            # if self.demo or self.dr_result_existed:
            #     self.run_dimension_reduction()

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
        # add all classes as items
        for cur_class in self.cur_classes_names:
            self.class_selection_menu.addItem(f'{cur_class}')
        # set default to 'all', which means averaged over all classes
        self.class_selection = 'all'
        self.class_selection_menu.setCurrentIndex(0)
        # connect the drop down menu with actions
        self.class_selection_menu.currentTextChanged.connect(aggregate_class_selection_changed)
        self.class_selection_menu.setEditable(True)
        self.class_selection_menu.lineEdit().setReadOnly(True)
        self.class_selection_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        # local layout that contains the drop down menu
        class_selection_menu_layout = QtWidgets.QHBoxLayout()
        class_selection_menu_layout.setContentsMargins(150, 0, 0, 0)
        class_selection_menu_layout.addWidget(self.class_selection_menu)
        # add the general layout
        self.interface_layout.addLayout(class_selection_menu_layout, 1, 0)

    def draw_point_cloud_aggregate_nero(self):
        # used to pass into subclass
        outer_self = self
        # subclass of ImageItem that reimplements the control methods
        class NERO_heatmap(pg.ImageItem):
            def __init__(self, plot_type, index):
                super().__init__()
                self.plot_type = plot_type
                self.index = index

            def mouseClickEvent(self, event):
                if self.plot_type == 'single':
                    print(f'Clicked on heatmap at ({event.pos().x()}, {event.pos().y()})')
                    # the position of un-repeated aggregate result
                    outer_self.block_x = int(np.floor(event.pos().x() // outer_self.block_size))
                    outer_self.block_y = int(np.floor(event.pos().y() // outer_self.block_size))

                    # draw a rectangle to highlight current selection of location
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

                    # redisplay individual output (result taken from the aggregate results)
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

        # determine the plane index
        self.cur_plane_index = self.all_planes.index(self.cur_plane)
        # select the data that we are using to draw
        # re-display aggregate NERO plot
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
        self.aggregate_heatmap_plot_1 = interface_util.draw_individual_heatmap(
            self.processed_aggregate_quantity_1
        )
        self.aggregate_heatmap_plot_2 = interface_util.draw_individual_heatmap(
            self.processed_aggregate_quantity_2
        )
        # add plot to view
        self.aggregate_heatmap_view_1.addItem(self.aggregate_heatmap_plot_1)
        self.aggregate_heatmap_view_2.addItem(self.aggregate_heatmap_plot_2)
        # add view to layout
        self.interface_layout.addWidget(self.aggregate_heatmap_view_1, 3, 0, 3, 1)
        self.interface_layout.addWidget(self.aggregate_heatmap_view_2, 5, 0, 3, 1)

    ################## DR Plots Related ##################
    def prepare_dr_results(self):
        self.all_dr_algorithms = ['PCA', 'ICA', 'ISOMAP', 't-SNE', 'UMAP']
        # load dr results from each algorithm
        self.all_axis_angles, successful = interface_util.load_from_cache(
            'all_axis_angles', self.cache
        )
        if not successful:
            self.all_axis_angles = list(range(-180, 181, 30))
            interface_util.save_to_cache(
                'all_axis_angles', self.all_axis_angles, self.cache, self.cache_path
            )

    def init_dr_plot_interface(self):
        @QtCore.Slot()
        # change different dimension reduction algorithms
        def dr_selection_changed(text):
            # update dimension reduction algorithm
            self.cur_dr_algorithm = text

            # re-display dr show result
            self.draw_dr_plot()

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
        self.dr_selection_menu.currentTextChanged.connect(dr_selection_changed)
        self.dr_selection_menu.setEditable(True)
        self.dr_selection_menu.lineEdit().setReadOnly(True)
        self.dr_selection_menu.lineEdit().setAlignment(QtCore.Qt.AlignRight)
        # add to local layout
        dr_selection_layout.addWidget(self.dr_selection_menu)
        self.interface_layout.addLayout(dr_selection_layout, 0, 1, 1, 1)

    def draw_dr_plot(self):
        pass


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
