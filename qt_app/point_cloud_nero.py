"""
NERO interface for point cloud classification
Author: Zhuokai Zhao
"""
import os
import sys
import glob
import torch
import argparse
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import QWidget, QLabel, QRadioButton
import pyqtgraph.opengl as gl
import warnings

warnings.filterwarnings('ignore')

import nero_utilities
import nero_run_model
import nero_custom_plots
import nero_interface_util


# globa configurations
pg.setConfigOptions(antialias=True, background='w')
# use pyside gpu acceleration if gpu detected
if torch.cuda.is_available():
    pg.setConfigOption('useCupy', True)
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
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignCenter)
        # left, top, right, and bottom margins
        self.layout.setContentsMargins(10, 10, 10, 10)
        # spacing
        self.layout.setHorizontalSpacing(20)
        self.layout.setVerticalSpacing(20)

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

        # # Individual NERO plot
        self.init_individual_plot_interface()
        self.draw_point_cloud_individual_nero()

        # # Detailed plot
        self.init_detail_plot_interface()
        self.draw_point_cloud_detail_plot()

        # visualize point cloud sample selected from DR plot
        self.init_point_cloud_vis_interface()
        self.draw_point_cloud()

        print(f'\nNERO interface ready')

    ################## Data Loading Related ##################
    def init_point_cloud_data(self):
        # modelnet40 and modelnet10
        # data samples paths
        self.all_nums_classes = [10, 40]
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
        # data loading interface layout
        data_loading_layout = QtWidgets.QGridLayout()
        data_loading_layout.setAlignment(QtCore.Qt.AlignLeft)
        data_loading_layout.setHorizontalSpacing(0)
        data_loading_layout.setVerticalSpacing(0)
        # dataset selection text prompt
        dataset_selection_pixmap = QPixmap(300, 50)
        dataset_selection_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(dataset_selection_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 300, 50, QtGui.Qt.AlignLeft, 'Data Set:')
        painter.end()
        # create label to contain the texts
        dataset_selection_label = QLabel(self)
        dataset_selection_label.setFixedSize(QtCore.QSize(350, 50))
        dataset_selection_label.setPixmap(dataset_selection_pixmap)
        # dataset selection drop down menu
        self.dataset_selection_menu = QtWidgets.QComboBox()
        self.dataset_selection_menu.setFixedSize(QtCore.QSize(220, 50))
        self.dataset_selection_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        # prompt in drop down menu
        self.dataset_selection_menu.addItem('Input dataset')
        # load all dataset names from the data paths
        for i in range(len(self.all_data_paths)):
            cur_name = self.all_data_paths[i].split('/')[-1].split('.')[0].split('_')[0]
            self.dataset_selection_menu.addItem(cur_name)
        # set default data selection
        self.dataset_selection_menu.setCurrentIndex(self.dataset_index)
        self.dataset_name = self.dataset_selection_menu.currentText()
        # connect the drop down menu with actions
        self.dataset_selection_menu.currentTextChanged.connect(self._dataset_selection_changed)
        self.dataset_selection_menu.setEditable(True)
        self.dataset_selection_menu.lineEdit().setReadOnly(True)
        self.dataset_selection_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        # add components to layout
        data_loading_layout.addWidget(dataset_selection_label, 0, 0)
        data_loading_layout.addWidget(self.dataset_selection_menu, 0, 1)
        self.layout.addLayout(data_loading_layout, 0, 0, 1, 1)

        # class loading interface layout
        class_loading_layout = QtWidgets.QGridLayout()
        class_loading_layout.setAlignment(QtCore.Qt.AlignLeft)
        class_loading_layout.setHorizontalSpacing(0)
        class_loading_layout.setVerticalSpacing(0)
        # class selection text prompt
        class_selection_pixmap = QPixmap(300, 50)
        class_selection_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(class_selection_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 300, 50, QtGui.Qt.AlignLeft, 'Subset: ')
        painter.end()
        # QLabel that contains the QPixmap
        class_selection_label = QLabel(self)
        class_selection_label.setFixedSize(QtCore.QSize(350, 50))
        class_selection_label.setPixmap(class_selection_pixmap)

        # class selection drop down menu
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
        # add components to layout
        class_loading_layout.addWidget(class_selection_label, 0, 0)
        class_loading_layout.addWidget(self.class_selection_menu, 0, 1)
        self.layout.addLayout(class_loading_layout, 1, 0, 1, 1)

    def load_point_cloud_data(self):

        # get data and classes names path from selected 1-based index
        self.cur_data_path = self.all_data_paths[self.dataset_index - 1]
        self.cur_name_path = self.all_names_paths[self.dataset_index - 1]
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
        print(f'\nLoaded data from {self.cur_data_path}')
        print(
            f'Data includes {len(self.point_cloud_paths)} point cloud samples belonging to {self.cur_num_classes} classes'
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
        # model loading interface layout
        model_loading_layout = QtWidgets.QGridLayout()
        model_loading_layout.setAlignment(QtCore.Qt.AlignLeft)
        model_loading_layout.setHorizontalSpacing(0)
        model_loading_layout.setVerticalSpacing(0)

        # models selection text prompt
        model_selection_pixmap = QPixmap(450, 50)
        model_selection_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(model_selection_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 450, 50, QtGui.Qt.AlignLeft, 'Models in Comparisons: ')
        painter.end()
        # create label to contain the texts
        model_selection_label = QLabel(self)
        model_selection_label.setFixedSize(QtCore.QSize(500, 50))
        model_selection_label.setPixmap(model_selection_pixmap)
        model_selection_label.setContentsMargins(0, 0, 0, 0)

        # model 1 icon (used to add to model selection drop down menu)
        self.model_1_label = QLabel(self)
        self.model_1_label.setContentsMargins(0, 0, 0, 0)
        self.model_1_label.setAlignment(QtCore.Qt.AlignCenter)
        model_1_icon = QPixmap(25, 25)
        model_1_icon.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(model_1_icon)
        nero_interface_util.draw_circle(painter, 12, 12, 10, 'blue')

        # model 1 selection drop down menu
        self.model_1_menu = QtWidgets.QComboBox()
        self.model_1_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
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

        # model 2 icon (used to add to model selection drop down menu)
        self.model_2_label = QLabel(self)
        self.model_2_label.setContentsMargins(0, 0, 0, 0)
        self.model_2_label.setAlignment(QtCore.Qt.AlignCenter)
        model_2_icon = QPixmap(25, 25)
        model_2_icon.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(model_2_icon)
        nero_interface_util.draw_circle(painter, 12, 12, 10, 'magenta')

        # model 1 selection drop down menu
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
        model_loading_layout.addWidget(model_selection_label, 0, 0, 1, 2)
        model_loading_layout.addWidget(self.model_1_menu, 1, 0, 1, 1)
        model_loading_layout.addWidget(self.model_2_menu, 1, 1, 1, 1)
        self.layout.addLayout(model_loading_layout, 1, 2, 2, 1)

    def load_point_cloud_model(self, model_name):
        # load the mode
        if model_name == 'Original':
            model_path = glob.glob(
                os.path.join(
                    os.getcwd(),
                    'example_models',
                    self.mode,
                    f'{self.cur_num_classes}_classes',
                    'non_eqv',
                    '*.pth',
                )
            )[0]
            # load model
            model = nero_run_model.load_model(self.mode, 'non-eqv', model_path, self.pt_model_cfg)
        elif model_name == 'Data Aug':
            model_path = glob.glob(
                os.path.join(
                    os.getcwd(),
                    'example_models',
                    self.mode,
                    f'{self.cur_num_classes}_classes',
                    'rot_eqv',
                    '*.pth',
                )
            )[0]
            # load model
            model = nero_run_model.load_model(self.mode, 'aug-eqv', model_path, self.pt_model_cfg)

        return model

    ################## General NERO Plots Settings Related ##################
    def init_general_control_interface(self):
        # metric selection interface layout
        metric_selection_layout = QtWidgets.QGridLayout()
        metric_selection_layout.setAlignment(QtCore.Qt.AlignLeft)
        metric_selection_layout.setHorizontalSpacing(0)
        metric_selection_layout.setVerticalSpacing(0)
        # NERO metric text prompt
        metric_pixmap = QPixmap(300, 50)
        metric_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(metric_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 300, 50, QtGui.Qt.AlignLeft, 'Metric: ')
        painter.end()
        metric_label = QLabel(self)
        metric_label.setFixedSize(QtCore.QSize(300, 50))
        metric_label.setPixmap(metric_pixmap)
        metric_label.setContentsMargins(0, 0, 0, 0)
        # NERO metric drop down menu
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
        # add components to the layout
        metric_selection_layout.addWidget(metric_label, 0, 0)
        metric_selection_layout.addWidget(self.metric_menu, 0, 1)
        self.layout.addLayout(metric_selection_layout, 0, 2)

        # radio buttons on which plane we are rotating in
        plane_selection_layout = QtWidgets.QGridLayout()
        plane_selection_layout.setAlignment(QtCore.Qt.AlignLeft)
        plane_selection_layout.setHorizontalSpacing(0)
        plane_selection_layout.setVerticalSpacing(0)
        # plane_selection_layout.setColumnStretch(0, 2)
        # plane selection text prompt
        plane_selection_pixmap = QPixmap(300, 50)
        plane_selection_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(plane_selection_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 300, 50, QtGui.Qt.AlignLeft, 'View Plane: ')
        painter.end()
        # QLabel that contains the QPixmap
        plane_selection_label = QLabel(self)
        plane_selection_label.setFixedSize(QtCore.QSize(350, 50))
        plane_selection_label.setPixmap(plane_selection_pixmap)
        plane_selection_label.setContentsMargins(0, 0, 0, 0)
        # NERO metric drop down menu
        self.plane_selection_menu = QtWidgets.QComboBox()
        self.plane_selection_menu.setFixedSize(QtCore.QSize(220, 50))
        self.plane_selection_menu.setStyleSheet(
            'color: black; font-family: Helvetica; font-style: normal; font-size: 34px'
        )
        self.plane_selection_menu.setEditable(True)
        self.plane_selection_menu.lineEdit().setReadOnly(True)
        self.plane_selection_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.plane_selection_menu.addItem('xy')
        self.plane_selection_menu.addItem('xz')
        self.plane_selection_menu.addItem('yz')
        # define default plotting quantity
        self.plane_selection_menu.setCurrentIndex(0)
        self.cur_plane = self.plane_selection_menu.currentText()
        # connect the drop down menu with actions
        self.plane_selection_menu.currentTextChanged.connect(self._view_plane_changed)
        # add both text and drop down menu to the layout
        plane_selection_layout.addWidget(plane_selection_label, 0, 0)
        plane_selection_layout.addWidget(self.plane_selection_menu, 0, 1)
        self.layout.addLayout(plane_selection_layout, 2, 0)

        # controls for dimension reduction algorithm selection
        dr_selection_layout = QtWidgets.QGridLayout()
        dr_selection_layout.setAlignment(QtCore.Qt.AlignLeft)
        dr_selection_layout.setHorizontalSpacing(0)
        dr_selection_layout.setVerticalSpacing(0)
        # dr selection text prompt
        dr_selection_pixmap = QPixmap(300, 50)
        dr_selection_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(dr_selection_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 300, 50, QtGui.Qt.AlignLeft, 'DR Plot:')
        painter.end()
        # QLabel that contains the QPixmap
        dr_selection_label = QLabel(self)
        dr_selection_label.setFixedSize(QtCore.QSize(350, 50))
        dr_selection_label.setPixmap(dr_selection_pixmap)
        # create the drop down menu
        self.dr_selection_menu = QtWidgets.QComboBox()
        self.dr_selection_menu.setFixedSize(QtCore.QSize(220, 50))
        self.dr_selection_menu.setContentsMargins(0, 0, 0, 0)
        self.dr_selection_menu.setStyleSheet(
            'color: black; font-family: Helvetica; font-style: normal; font-size: 34px'
        )
        self.all_dr_algorithms = ['PCA', 'ICA', 'ISOMAP', 't-SNE', 'UMAP']
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
        # add components to layout
        dr_selection_layout.addWidget(dr_selection_label, 0, 0)
        dr_selection_layout.addWidget(self.dr_selection_menu, 0, 1)
        self.layout.addLayout(dr_selection_layout, 0, 1, 1, 1)

        # radio buttons on using mean or variance for color-encoding
        dr_encoding_layout = QtWidgets.QGridLayout()
        dr_encoding_layout.setAlignment(QtCore.Qt.AlignLeft)
        dr_encoding_layout.setHorizontalSpacing(0)
        dr_encoding_layout.setVerticalSpacing(0)
        dr_encoding_layout.setColumnStretch(0, 3)
        dr_encoding_layout.setColumnStretch(1, 0)
        dr_encoding_layout.setColumnStretch(2, 0)
        # color-encoding text prompt
        encoding_button_pixmap = QPixmap(350, 50)
        encoding_button_pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(encoding_button_pixmap)
        painter.setFont(QFont('Helvetica', 30))
        painter.drawText(0, 0, 350, 50, QtGui.Qt.AlignLeft, 'Color Encoding:')
        painter.end()
        # create label to contain the texts
        encoding_button_label = QLabel(self)
        encoding_button_label.setContentsMargins(0, 0, 0, 0)
        encoding_button_label.setFixedSize(QtCore.QSize(350, 50))
        encoding_button_label.setAlignment(QtCore.Qt.AlignLeft)
        encoding_button_label.setWordWrap(True)
        encoding_button_label.setTextFormat(QtGui.Qt.AutoText)
        encoding_button_label.setPixmap(encoding_button_pixmap)
        encoding_button_label.setContentsMargins(8, 0, 0, 0)
        # mean button
        self.mean_encoding_button = QRadioButton('Mean')
        self.mean_encoding_button.setFixedSize(QtCore.QSize(110, 50))
        self.mean_encoding_button.setContentsMargins(0, 0, 0, 0)
        self.mean_encoding_button.setStyleSheet(
            'color: black; font-style: normal; font-family: Helvetica; font-size: 28px;'
        )
        # by default the intensities are computed via mean
        self.mean_encoding_button.setChecked(True)
        self.intensity_method = 'mean'
        self.mean_encoding_button.pressed.connect(self._mean_encoding_button_clicked)
        # variance button
        self.variance_encoding_button = QRadioButton('Variance')
        self.variance_encoding_button.setFixedSize(QtCore.QSize(130, 50))
        self.variance_encoding_button.setContentsMargins(0, 0, 0, 0)
        self.variance_encoding_button.setStyleSheet(
            'color: black; font-style: normal; font-family: Helvetica; font-size: 28px;'
        )
        self.variance_encoding_button.pressed.connect(self._variance_encoding_button_clicked)
        # add components to layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.mean_encoding_button)
        button_layout.addWidget(self.variance_encoding_button)
        dr_encoding_layout.addWidget(encoding_button_label, 0, 0)
        dr_encoding_layout.addLayout(button_layout, 0, 1)
        self.layout.addLayout(dr_encoding_layout, 1, 1)

        # colorbar that will be shared in aggregate, dr and individual plot
        colorbar_layout = QtWidgets.QGridLayout()
        colorbar_layout.setAlignment(QtCore.Qt.AlignLeft)
        colorbar_layout.setHorizontalSpacing(0)
        colorbar_layout.setVerticalSpacing(0)
        # create viridis color scheme
        self.cm_range = (0, 1)
        self.color_map = pg.colormap.get('viridis')
        self.color_bar = pg.ColorBarItem(
            values=self.cm_range,
            colorMap=self.color_map,
            interactive=False,
            orientation='horizontal',
            width=30,
        )
        color_bar_view = pg.GraphicsLayoutWidget()
        color_bar_plot = pg.PlotItem()
        color_bar_plot.layout.setContentsMargins(0, 0, 0, 0)
        color_bar_plot.setFixedHeight(0)
        color_bar_plot.setFixedWidth(self.plot_size * 1.8)
        color_bar_plot.hideAxis('bottom')
        color_bar_plot.hideAxis('left')
        color_bar_view.addItem(color_bar_plot)
        color_bar_image = pg.ImageItem()
        self.color_bar.setImageItem(color_bar_image, insert_in=color_bar_plot)
        color_bar_plot.layout.setContentsMargins(0, 0, 0, 0)
        # add components to layout
        self.layout.addWidget(color_bar_view, 2, 1)

        # # checkbox on if doing real-time inference
        # self.realtime_inference_checkbox = QtWidgets.QCheckBox('Realtime inference when dragging')
        # self.realtime_inference_checkbox.setStyleSheet(
        #     'color: black; font-family: Helvetica; font-style: normal; font-size: 18px; background-color: white;'
        # )
        # self.realtime_inference_checkbox.setFixedSize(QtCore.QSize(300, 30))
        # self.realtime_inference_checkbox.setContentsMargins(0, 0, 0, 0)
        # self.realtime_inference_checkbox.stateChanged.connect(
        #     self._realtime_inference_checkbox_clicked
        # )
        # if self.realtime_inference:
        #     self.realtime_inference_checkbox.setChecked(True)
        # else:
        #     self.realtime_inference_checkbox.setChecked(False)
        # # layout that controls the plotting items
        # self.layout.addWidget(self.realtime_inference_checkbox, 0, 3)

    ################## Aggregate NERO Plots Related ##################
    def prepare_aggregate_results(self):
        self.all_planes = ['xy', 'xz', 'yz']

        # axis angles
        self.all_axis_angles, successful = nero_interface_util.load_from_cache(
            'all_axis_angles', self.cache
        )
        if not successful:
            self.all_axis_angles = list(range(0, 361, 5))
            nero_interface_util.save_to_cache(
                'all_axis_angles', self.all_axis_angles, self.cache, self.cache_path
            )

        # rotation angles
        self.all_rot_angles, successful = nero_interface_util.load_from_cache(
            'all_rot_angles', self.cache
        )
        if not successful:
            self.all_rot_angles = list(range(0, 181, 15))
            nero_interface_util.save_to_cache(
                'all_rot_angles', self.all_rot_angles, self.cache, self.cache_path
            )

        # aggregate test results for model 1
        (
            self.all_avg_instance_accuracies_1,
            successful_avg_ins,
        ) = nero_interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_instance_accuracies',
            self.cache,
        )
        self.all_avg_class_accuracies_1, successful_avg_cls = nero_interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_class_accuracies',
            self.cache,
        )
        self.all_avg_accuracies_per_class_1, successful_cls = nero_interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_accuracies_per_class',
            self.cache,
        )
        self.all_outputs_1, successful_output = nero_interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_outputs',
            self.cache,
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
                self.all_avg_instance_accuracies_1,
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
            nero_interface_util.save_to_cache(
                [
                    f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_instance_accuracies',
                    f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_class_accuracies',
                    f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_avg_accuracies_per_class',
                    f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_outputs',
                ],
                [
                    self.all_avg_instance_accuracies_1,
                    self.all_avg_class_accuracies_1,
                    self.all_avg_accuracies_per_class_1,
                    self.all_outputs_1,
                ],
                self.cache,
                self.cache_path,
            )

        # aggregate test results for model 2
        self.all_avg_instance_accuracies_2, successful = nero_interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_instance_accuracies',
            self.cache,
        )
        self.all_avg_class_accuracies_2, successful = nero_interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_class_accuracies',
            self.cache,
        )
        self.all_avg_accuracies_per_class_2, successful = nero_interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_accuracies_per_class',
            self.cache,
        )
        self.all_outputs_2, successful = nero_interface_util.load_from_cache(
            f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_outputs',
            self.cache,
        )

        # if any of the result for model 1 is missing, run aggregate test
        if not successful:
            print(f'\nRunning aggregate test for model 2')
            (
                self.all_avg_instance_accuracies_2,
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
            nero_interface_util.save_to_cache(
                [
                    f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_instance_accuracies',
                    f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_class_accuracies',
                    f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_avg_accuracies_per_class',
                    f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_outputs',
                ],
                [
                    self.all_avg_instance_accuracies_2,
                    self.all_avg_class_accuracies_2,
                    self.all_avg_accuracies_per_class_2,
                    self.all_outputs_2,
                ],
                self.cache,
                self.cache_path,
            )

    def init_aggregate_plot_interface(self):
        # aggregate NERO plots interface layout
        aggregate_nero_layout = QtWidgets.QGridLayout()
        aggregate_nero_layout.setAlignment(QtCore.Qt.AlignLeft)
        aggregate_nero_layout.setHorizontalSpacing(0)
        aggregate_nero_layout.setVerticalSpacing(0)
        # display in heatmap
        # heatmap view for model 1
        self.aggregate_nero_view_1 = pg.GraphicsLayoutWidget()
        self.aggregate_nero_view_1.ci.layout.setContentsMargins(
            0, 0, 0, 0  # left top right bottom
        )
        self.aggregate_nero_view_1.setFixedSize(self.plot_size * 1.35, self.plot_size * 1.35)
        # heatmap view for model 2
        self.aggregate_nero_view_2 = pg.GraphicsLayoutWidget()
        self.aggregate_nero_view_2.ci.layout.setContentsMargins(
            0, 0, 0, 0  # left top right bottom
        )
        self.aggregate_nero_view_2.setFixedSize(self.plot_size * 1.35, self.plot_size * 1.35)
        # add view to layout
        aggregate_nero_layout.addWidget(self.aggregate_nero_view_1, 0, 0)
        aggregate_nero_layout.addWidget(self.aggregate_nero_view_2, 1, 0)
        self.layout.addLayout(aggregate_nero_layout, 3, 0)

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
        ) = nero_interface_util.process_point_cloud_result(
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
        ) = nero_interface_util.process_point_cloud_result(
            self.cur_aggregate_plot_quantity_2,
            self.cur_plane,
            self.all_axis_angles,
            self.all_rot_angles,
            block_size=self.block_size,
        )
        # initialize plot, we don't need interaction on aggregate plots
        self.aggregate_nero_1 = nero_custom_plots.NEROHeatmap(
            self, model_index=1, interaction=False
        )
        self.aggregate_nero_2 = nero_custom_plots.NEROHeatmap(
            self, model_index=2, interaction=False
        )
        # draw the heatmap
        self.aggregate_heatmap_plot_1 = self._draw_individual_heatmap(
            self.processed_aggregate_quantity_1, self.aggregate_nero_1
        )
        self.aggregate_heatmap_plot_2 = self._draw_individual_heatmap(
            self.processed_aggregate_quantity_2, self.aggregate_nero_2
        )
        # add plot to view
        self.aggregate_nero_view_1.clear()
        self.aggregate_nero_view_1.addItem(self.aggregate_heatmap_plot_1)
        self.aggregate_nero_view_2.clear()
        self.aggregate_nero_view_2.addItem(self.aggregate_heatmap_plot_2)

    ################## DR Plots Related ##################
    def prepare_dr_results(self):
        self.all_low_dim_points_1 = {}
        self.all_low_dim_points_2 = {}
        high_dim_points_constructed_1 = False
        high_dim_points_constructed_2 = False
        # iteracte through each dr method
        for cur_algo in self.all_dr_algorithms:
            # for model 1
            self.all_high_dim_points_1, successful_high = nero_interface_util.load_from_cache(
                f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_high_dim',
                self.cache,
            )
            (
                self.all_low_dim_points_1[cur_algo],
                successful_low,
            ) = nero_interface_util.load_from_cache(
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
                    all_outputs_reshaped_1 = self.all_outputs_1.reshape(
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
                        self.all_high_dim_points_1[i] = all_outputs_reshaped_1[
                            i, :, cur_ground_truth_index
                        ]

                    high_dim_points_constructed_1 = True
                    nero_interface_util.save_to_cache(
                        f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_high_dim',
                        self.all_high_dim_points_1,
                        self.cache,
                        self.cache_path,
                    )

                # compute the dr results from model outputs
                low_dim_1 = nero_interface_util.dimension_reduce(
                    cur_algo, self.all_high_dim_points_1, 2
                )
                # normalizing low dimension points within [-1, 1] sqaure
                self.all_low_dim_points_1[cur_algo] = nero_interface_util.normalize_low_dim_result(
                    low_dim_1
                )
                nero_interface_util.save_to_cache(
                    f'{self.mode}_{self.dataset_name}_{self.model_1_cache_name}_{cur_algo}_low_dim',
                    self.all_low_dim_points_1[cur_algo],
                    self.cache,
                    self.cache_path,
                )

            # for model 2
            self.all_high_dim_points_2, successful_high = nero_interface_util.load_from_cache(
                f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_high_dim',
                self.cache,
            )
            (
                self.all_low_dim_points_2[cur_algo],
                successful_low,
            ) = nero_interface_util.load_from_cache(
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
                    all_outputs_reshaped_2 = self.all_outputs_2.reshape(
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
                        self.all_high_dim_points_2[i] = all_outputs_reshaped_2[
                            i, :, cur_ground_truth_index
                        ]

                    high_dim_points_constructed_2 = True
                    nero_interface_util.save_to_cache(
                        f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_high_dim',
                        self.all_high_dim_points_2,
                        self.cache,
                        self.cache_path,
                    )

                # compute the dr results from model outputs
                low_dim_2 = nero_interface_util.dimension_reduce(
                    cur_algo, self.all_high_dim_points_2, 2
                )
                # normalizing low dimension points within [-1, 1] sqaure
                self.all_low_dim_points_2[cur_algo] = nero_interface_util.normalize_low_dim_result(
                    low_dim_2
                )
                nero_interface_util.save_to_cache(
                    f'{self.mode}_{self.dataset_name}_{self.model_2_cache_name}_{cur_algo}_low_dim',
                    self.all_low_dim_points_2[cur_algo],
                    self.cache,
                    self.cache_path,
                )

        # data sample indices that we are plotting
        self.cur_class_indices = []
        if self.class_selection == 'all':
            self.cur_class_indices = list(range(len(self.point_cloud_paths)))
        else:
            for i in range(len(self.point_cloud_paths)):
                if self.point_cloud_paths[i][0] == self.class_selection:
                    self.cur_class_indices.append(i)

    def init_dr_plot_interface(self):
        # dr plot interface layout
        dr_plots_layout = QtWidgets.QGridLayout()
        dr_plots_layout.setAlignment(QtCore.Qt.AlignLeft)
        dr_plots_layout.setHorizontalSpacing(0)
        dr_plots_layout.setVerticalSpacing(0)

        # scatter item size for both dr plots
        self.scatter_item_size = 12
        # initialize selected index and the corresponding point cloud path
        self.point_cloud_index = 0
        self.point_cloud_path = self.point_cloud_paths[self.point_cloud_index][1]
        print(f'\nInitialized selecting point cloud at {self.point_cloud_path}')

        # dr plot for model 1
        self.low_dim_scatter_view_1 = pg.GraphicsLayoutWidget()
        self.low_dim_scatter_view_1.setBackground('white')
        self.low_dim_scatter_view_1.setFixedSize(self.plot_size * 1.35, self.plot_size * 1.35)
        self.low_dim_scatter_view_1.ci.setContentsMargins(20, 0, 0, 0)

        # dr plot for model 2
        self.low_dim_scatter_view_2 = pg.GraphicsLayoutWidget()
        self.low_dim_scatter_view_2.setBackground('white')
        self.low_dim_scatter_view_2.setFixedSize(self.plot_size * 1.35, self.plot_size * 1.35)
        self.low_dim_scatter_view_2.ci.setContentsMargins(20, 0, 0, 0)

        # sliders that rank the dimension reduction result and can select one of them
        # slider 1
        slider_1_layout = QtWidgets.QGridLayout()
        slider_1_layout.setVerticalSpacing(0)
        self.dr_result_selection_slider_1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dr_result_selection_slider_1.setFixedSize(self.plot_size + 30, 50)
        self.dr_result_selection_slider_1.setMinimum(0)
        self.dr_result_selection_slider_1.setMaximum(len(self.cur_class_indices) - 1)
        self.dr_result_selection_slider_1.setValue(0)
        self.dr_result_selection_slider_1.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.dr_result_selection_slider_1.setTickInterval(1)
        self.dr_result_selection_slider_1.valueChanged.connect(
            self._dr_result_selection_slider_1_changed
        )
        slider_1_layout.addWidget(self.dr_result_selection_slider_1, 0, 0, 1, 3)
        # left and right buttons to move the slider around, with number in the middle
        # left button
        self.slider_1_left_button = QtWidgets.QToolButton()
        self.slider_1_left_button.setArrowType(QtCore.Qt.LeftArrow)
        self.slider_1_left_button.clicked.connect(self._slider_1_left_button_clicked)
        self.slider_1_left_button.setFixedSize(30, 30)
        self.slider_1_left_button.setStyleSheet('color: black')
        slider_1_layout.addWidget(self.slider_1_left_button, 1, 0, 1, 1)
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
        slider_1_layout.addWidget(self.slider_1_text_label, 1, 1, 1, 1)
        # right button
        self.slider_1_right_button = QtWidgets.QToolButton()
        self.slider_1_right_button.setArrowType(QtCore.Qt.RightArrow)
        self.slider_1_right_button.setFixedSize(30, 30)
        self.slider_1_right_button.setStyleSheet('color: black')
        self.slider_1_right_button.clicked.connect(self._slider_1_right_button_clicked)
        slider_1_layout.addWidget(self.slider_1_right_button, 1, 2, 1, 1)
        # initialize slider selection
        self.slider_1_selected_index = None
        self.slider_1_locked = False
        slider_1_layout.setContentsMargins(0, 0, 0, 0)

        # slider 2
        slider_2_layout = QtWidgets.QGridLayout()
        slider_2_layout.setVerticalSpacing(0)
        self.dr_result_selection_slider_2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dr_result_selection_slider_2.setFixedSize(self.plot_size + 30, 50)
        self.dr_result_selection_slider_2.setMinimum(0)
        self.dr_result_selection_slider_2.setMaximum(len(self.cur_class_indices) - 1)
        self.dr_result_selection_slider_2.setValue(0)
        self.dr_result_selection_slider_2.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.dr_result_selection_slider_2.setTickInterval(1)
        self.dr_result_selection_slider_2.valueChanged.connect(
            self._dr_result_selection_slider_2_changed
        )
        slider_2_layout.addWidget(self.dr_result_selection_slider_2, 0, 0, 1, 3)
        # left and right buttons to move the slider around, with number in the middle
        # left button
        self.slider_2_left_button = QtWidgets.QToolButton()
        self.slider_2_left_button.setArrowType(QtCore.Qt.LeftArrow)
        self.slider_2_left_button.setFixedSize(30, 30)
        self.slider_2_left_button.setStyleSheet('color: black')
        self.slider_2_left_button.clicked.connect(self._slider_2_left_button_clicked)
        slider_2_layout.addWidget(self.slider_2_left_button, 1, 0, 1, 1)
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
        slider_2_layout.addWidget(self.slider_2_text_label, 1, 1, 1, 1)
        # right button
        self.slider_2_right_button = QtWidgets.QToolButton()
        self.slider_2_right_button.setArrowType(QtCore.Qt.RightArrow)
        self.slider_2_right_button.setFixedSize(30, 30)
        self.slider_2_right_button.setStyleSheet('color: black')
        self.slider_2_right_button.clicked.connect(self._slider_2_right_button_clicked)
        slider_2_layout.addWidget(self.slider_2_right_button, 1, 2, 1, 1)
        # initialize slider selection
        self.slider_2_selected_index = None
        self.slider_2_locked = False
        slider_2_layout.setContentsMargins(0, 0, 0, 0)

        # add components to layout
        dr_plots_layout.addWidget(self.low_dim_scatter_view_1, 0, 0)
        dr_plots_layout.addLayout(slider_1_layout, 1, 0)
        dr_plots_layout.addWidget(self.low_dim_scatter_view_2, 2, 0)
        dr_plots_layout.addLayout(slider_2_layout, 3, 0)
        dr_plots_layout.setRowStretch(0, 2)
        dr_plots_layout.setRowStretch(1, 0)
        dr_plots_layout.setRowStretch(2, 2)
        dr_plots_layout.setRowStretch(3, 0)
        self.layout.addLayout(dr_plots_layout, 3, 1)

    def draw_dr_plot(self):
        # high dimensional points of the current class
        self.cur_class_high_dim_1 = self.all_high_dim_points_1[self.cur_class_indices]
        self.cur_class_high_dim_2 = self.all_high_dim_points_2[self.cur_class_indices]
        # get the dimension reduced points
        self.cur_class_low_dim_1 = self.all_low_dim_points_1[self.cur_dr_algorithm][
            self.cur_class_indices
        ]
        self.cur_class_low_dim_2 = self.all_low_dim_points_2[self.cur_dr_algorithm][
            self.cur_class_indices
        ]
        # use each sample's metric average or variance across all transformations as intensity
        self.all_intensity_1 = nero_interface_util.compute_intensity(
            self.cur_class_high_dim_1, self.intensity_method
        )
        self.all_intensity_2 = nero_interface_util.compute_intensity(
            self.cur_class_high_dim_2, self.intensity_method
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
        self.cur_class_low_dim_1 = self.cur_class_low_dim_1[self.sorted_intensity_indices_1]
        self.cur_class_low_dim_2 = self.cur_class_low_dim_2[self.sorted_intensity_indices_2]
        # scatter plot 1
        # initialize plot
        self.low_dim_scatter_view_1.clear()
        self.low_dim_scatter_plot_1 = self.low_dim_scatter_view_1.addPlot()
        self.low_dim_scatter_plot_1.setContentsMargins(0, 0, 0, 0)
        self.low_dim_scatter_plot_1.hideAxis('left')
        self.low_dim_scatter_plot_1.hideAxis('bottom')
        # set axis range
        self.low_dim_scatter_plot_1.setXRange(-1.2, 1.2, padding=0)
        self.low_dim_scatter_plot_1.setYRange(-1.2, 1.2, padding=0)
        # Not letting user zoom out past axis limit
        self.low_dim_scatter_plot_1.vb.setLimits(xMin=-1.2, xMax=1.2, yMin=-1.2, yMax=1.2)
        # No auto range when adding new item (red indicator)
        self.low_dim_scatter_plot_1.vb.disableAutoRange(axis=pg.ViewBox.XYAxes)
        self._draw_scatter_plot(
            self.low_dim_scatter_plot_1,
            self.cur_class_low_dim_1,
            self.sorted_intensity_1,
            self.sorted_class_indices_1,
        )
        # scatter plot 2
        # initialize plot
        self.low_dim_scatter_view_2.clear()
        self.low_dim_scatter_plot_2 = self.low_dim_scatter_view_2.addPlot()
        self.low_dim_scatter_plot_2.hideAxis('left')
        self.low_dim_scatter_plot_2.hideAxis('bottom')
        # set axis range
        self.low_dim_scatter_plot_2.setXRange(-1.2, 1.2, padding=0)
        self.low_dim_scatter_plot_2.setYRange(-1.2, 1.2, padding=0)
        # Not letting user zoom out past axis limit
        self.low_dim_scatter_plot_2.vb.setLimits(xMin=-1.2, xMax=1.2, yMin=-1.2, yMax=1.2)
        # No auto range when adding new item (red indicator)
        self.low_dim_scatter_plot_2.vb.disableAutoRange(axis=pg.ViewBox.XYAxes)
        self._draw_scatter_plot(
            self.low_dim_scatter_plot_2,
            self.cur_class_low_dim_2,
            self.sorted_intensity_2,
            self.sorted_class_indices_2,
        )

    ################## Individual NERO Plot Related ##################
    def init_individual_plot_interface(self):
        # individual NERO plots interface layout
        individual_nero_layout = QtWidgets.QGridLayout()
        individual_nero_layout.setAlignment(QtCore.Qt.AlignLeft)
        individual_nero_layout.setHorizontalSpacing(0)
        individual_nero_layout.setVerticalSpacing(0)
        # heatmap view for model 1
        self.individual_nero_view_1 = pg.GraphicsLayoutWidget()
        self.individual_nero_view_1.ci.layout.setContentsMargins(
            0, 0, 0, 0  # left top right bottom
        )
        self.individual_nero_view_1.setFixedSize(self.plot_size * 1.35, self.plot_size * 1.35)
        # heatmap view for model 2
        self.individual_nero_view_2 = pg.GraphicsLayoutWidget()
        self.individual_nero_view_2.ci.layout.setContentsMargins(
            0, 0, 0, 0  # left top right bottom
        )
        self.individual_nero_view_2.setFixedSize(self.plot_size * 1.35, self.plot_size * 1.35)
        # add view to layout
        individual_nero_layout.addWidget(self.individual_nero_view_1, 0, 0)
        individual_nero_layout.addWidget(self.individual_nero_view_2, 1, 0)
        self.layout.addLayout(individual_nero_layout, 3, 2)

        # initialize highlighters on individual NERO plots
        self.highlighter_1 = pg.ScatterPlotItem(pxMode=False)
        self.highlighter_1.setSymbol('s')
        self.highlighter_2 = pg.ScatterPlotItem(pxMode=False)
        self.highlighter_2.setSymbol('s')

        # initialize current selected position
        self.click_image_x = len(self.all_rot_angles) - 1
        self.click_image_y = len(self.all_rot_angles) - 1
        # recover rotation axis angle and the actual rotate angle around that axis
        self.axis_angle_index, self.rot_angle_index = nero_interface_util.click_to_rotation(
            self.click_image_x,
            self.click_image_y,
            self.all_axis_angles,
            self.all_rot_angles,
        )

    def draw_point_cloud_individual_nero(self, highlighter_only=False):
        # highlighter_only is for when we only update the selection highlight
        if not highlighter_only:
            # get the results from aggregate results with respect to user selected sample
            self.ground_truth_index = self.cur_classes_names.index(
                self.point_cloud_paths[self.point_cloud_index][0]
            )
            self.cur_individual_plot_quantity_1 = self.all_outputs_1[
                self.cur_plane_index, :, :, self.point_cloud_index, self.ground_truth_index
            ]
            self.cur_individual_plot_quantity_2 = self.all_outputs_2[
                self.cur_plane_index, :, :, self.point_cloud_index, self.ground_truth_index
            ]
            # convert the result to polar plot data fit in rectangular array
            # model 1 results
            (
                self.processed_individual_quantity_1,
                self.processed_individual_quaternion_1,
            ) = nero_interface_util.process_point_cloud_result(
                self.cur_individual_plot_quantity_1,
                self.cur_plane,
                self.all_axis_angles,
                self.all_rot_angles,
                block_size=self.block_size,
            )
            # model 2 results
            (
                self.processed_individual_quantity_2,
                self.processed_individual_quaternion_2,
            ) = nero_interface_util.process_point_cloud_result(
                self.cur_individual_plot_quantity_2,
                self.cur_plane,
                self.all_axis_angles,
                self.all_rot_angles,
                block_size=self.block_size,
            )

            # initialize plot
            self.individual_nero_1 = nero_custom_plots.NEROHeatmap(
                self,
                model_index=1,
                interaction=True,
                reaction_function=self._individual_nero_clicked,
            )
            self.individual_nero_2 = nero_custom_plots.NEROHeatmap(
                self,
                model_index=2,
                interaction=True,
                reaction_function=self._individual_nero_clicked,
            )
            # draw the heatmap
            self.individual_heatmap_plot_1 = self._draw_individual_heatmap(
                self.processed_individual_quantity_1, self.individual_nero_1
            )
            self.individual_heatmap_plot_2 = self._draw_individual_heatmap(
                self.processed_individual_quantity_2, self.individual_nero_2
            )
        else:
            self.individual_heatmap_plot_1.removeItem(self.highlighter_1)
            self.individual_heatmap_plot_2.removeItem(self.highlighter_2)

        # add default selection highlighters
        rectangle_highlighter = [
            {
                'pos': (
                    self.click_image_x * self.block_size
                    + self.block_size // 2,  # top left to center pos
                    self.click_image_y * self.block_size
                    + self.block_size // 2,  # top left to center pos
                ),
                'size': self.block_size,
                'pen': {'color': 'red', 'width': 3},
                'brush': (0, 0, 0, 0),
            }
        ]
        # add highlighters to plots
        self.highlighter_1.setData(rectangle_highlighter)
        self.individual_heatmap_plot_1.addItem(self.highlighter_1)
        self.highlighter_2.setData(rectangle_highlighter)
        self.individual_heatmap_plot_2.addItem(self.highlighter_2)
        # add plots to views
        self.individual_nero_view_1.clear()
        self.individual_nero_view_1.addItem(self.individual_heatmap_plot_1)
        self.individual_nero_view_2.clear()
        self.individual_nero_view_2.addItem(self.individual_heatmap_plot_2)

    ################## Detail Plot Related ##################
    def init_detail_plot_interface(self):
        # detail plots interface layout
        detail_plot_layout = QtWidgets.QGridLayout()
        detail_plot_layout.setAlignment(QtCore.Qt.AlignLeft)
        detail_plot_layout.setHorizontalSpacing(0)
        detail_plot_layout.setVerticalSpacing(0)
        # label stype and font
        tick_font = QFont('Helvetica', 14)
        x_label_style = {'color': 'black', 'font-size': '16pt', 'text': 'Object Classes'}
        y_label_style = {'color': 'black', 'font-size': '16pt', 'text': 'Confidence'}
        self.detail_plot_bar_width = 0.5
        # prepare ticks
        ticks = []
        for i, label in enumerate(self.cur_classes_names):
            ticks.append((i + 1, label))
        # initialize plot 1
        self.bar_plot_1 = pg.plot()
        self.bar_plot_1.setBackground('w')
        self.bar_plot_1.plotItem.vb.setLimits(
            xMin=-self.detail_plot_bar_width / 2,
            xMax=self.cur_num_classes + self.detail_plot_bar_width / 2,
            yMin=0,
            yMax=1,
        )
        self.bar_plot_1.setFixedSize(self.plot_size * 3, self.plot_size * 1.35)
        self.bar_plot_1.setContentsMargins(0, 100, 0, 0)
        self.bar_plot_1.getAxis('bottom').setLabel(**x_label_style)
        self.bar_plot_1.getAxis('left').setLabel(**y_label_style)
        self.bar_plot_1.getAxis('bottom').setTickFont(tick_font)
        self.bar_plot_1.getAxis('left').setTickFont(tick_font)
        self.bar_plot_1.getAxis('bottom').setTextPen('black')
        self.bar_plot_1.getAxis('left').setTextPen('black')
        self.bar_plot_1.setMouseEnabled(x=False, y=False)
        self.bar_plot_1.getAxis('bottom').setTicks([ticks])
        # initialize plot 2
        self.bar_plot_2 = pg.plot()
        self.bar_plot_2.setBackground('w')
        self.bar_plot_2.plotItem.vb.setLimits(
            xMin=-self.detail_plot_bar_width / 2,
            xMax=self.cur_num_classes + self.detail_plot_bar_width / 2,
            yMin=0,
            yMax=1,
        )
        self.bar_plot_2.setFixedSize(self.plot_size * 3, self.plot_size * 1.35)
        self.bar_plot_2.setContentsMargins(0, 100, 0, 0)
        self.bar_plot_2.getAxis('bottom').setLabel(**x_label_style)
        self.bar_plot_2.getAxis('left').setLabel(**y_label_style)
        self.bar_plot_2.getAxis('bottom').setTickFont(tick_font)
        self.bar_plot_2.getAxis('left').setTickFont(tick_font)
        self.bar_plot_2.getAxis('bottom').setTextPen('black')
        self.bar_plot_2.getAxis('left').setTextPen('black')
        self.bar_plot_2.setMouseEnabled(x=False, y=False)
        self.bar_plot_2.getAxis('bottom').setTicks([ticks])
        # add plots to layout
        detail_plot_layout.addWidget(self.bar_plot_1, 0, 0)
        detail_plot_layout.addWidget(self.bar_plot_2, 1, 0)
        self.layout.addLayout(detail_plot_layout, 3, 3)

    def draw_point_cloud_detail_plot(self):
        # all the probabilities of current selected sample
        self.cur_individual_plot_quantity_1 = self.all_outputs_1[
            self.cur_plane_index,
            self.axis_angle_index,
            self.rot_angle_index,
            self.point_cloud_index,
            :,
        ]
        self.cur_individual_plot_quantity_2 = self.all_outputs_2[
            self.cur_plane_index,
            self.axis_angle_index,
            self.rot_angle_index,
            self.point_cloud_index,
            :,
        ]
        # make the bar plot
        graph_1 = pg.BarGraphItem(
            x=np.arange(1, len(self.cur_individual_plot_quantity_1) + 1),
            height=list(self.cur_individual_plot_quantity_1),
            width=self.detail_plot_bar_width,
            brush='blue',
        )
        graph_2 = pg.BarGraphItem(
            x=np.arange(1, len(self.cur_individual_plot_quantity_2) + 1),
            height=list(self.cur_individual_plot_quantity_2),
            width=self.detail_plot_bar_width,
            brush='magenta',
        )

        # add graph to plot
        self.bar_plot_1.clear()
        self.bar_plot_1.addItem(graph_1)
        self.bar_plot_2.clear()
        self.bar_plot_2.addItem(graph_2)

    ################## Point Cloud Sample Visualization Related ##################
    def init_point_cloud_vis_interface(self):
        # point cloud visualization layout
        point_cloud_vis_layout = QtWidgets.QGridLayout()
        point_cloud_vis_layout.setAlignment(QtCore.Qt.AlignLeft)
        point_cloud_vis_layout.setHorizontalSpacing(0)
        point_cloud_vis_layout.setVerticalSpacing(0)
        # widget
        self.point_cloud_vis_widget = gl.GLViewWidget()
        self.point_cloud_vis_widget.setBackgroundColor('black')
        self.point_cloud_vis_widget.opts['distance'] = 10
        # add to layout
        point_cloud_vis_layout.addWidget(self.point_cloud_vis_widget)
        self.layout.addLayout(point_cloud_vis_layout, 0, 3, 3, 1)

    def draw_point_cloud(self):
        # load current point clouds
        point_cloud = np.loadtxt(self.point_cloud_path, delimiter=',').astype(np.float32)
        point_cloud_pos = point_cloud[:, :3]
        # point size
        sizes = np.array([0.02] * len(point_cloud_pos))
        # assign color
        colors = np.atleast_2d([0.5, 0.5, 0.5, 0.5]).repeat(repeats=len(point_cloud_pos), axis=0)
        # set data
        self.point_cloud_vis_widget.clear()
        self.point_cloud_vis_item = gl.GLScatterPlotItem()
        self.point_cloud_vis_widget.addItem(self.point_cloud_vis_item)
        self.point_cloud_vis_item.setData(
            pos=point_cloud_pos, color=colors, size=sizes, pxMode=False
        )

    ################## All private functions ##################
    # dataset drop-down menu
    @QtCore.Slot()
    def _dataset_selection_changed(self, text):
        # filter out 0 selection signal
        if text == 'Input dataset':
            return

        # udpate dataset selection
        self.dataset_name = text
        self.dataset_index = self.aggregate_image_menu.currentIndex()
        # re-load the data
        self.load_point_cloud_data()
        # re-load the model since different model correspond to different dataset
        self.init_point_cloud_models()
        self.model_1 = self.load_point_cloud_model(self.model_1_name)
        self.model_2 = self.load_point_cloud_model(self.model_2_name)
        # update aggregate nero plot
        self.draw_point_cloud_aggregate_nero()
        # update dr plot
        self.draw_dr_plot()
        # update visualization of selected point cloud
        self.draw_point_cloud()
        # update individiaul NERO plot
        self.draw_point_cloud_individual_nero()
        # update detail plot
        self.draw_point_cloud_detail_plot()

    # dataset class selection drop-down menu
    @QtCore.Slot()
    def _class_selection_changed(self, text):
        # re-initialize the scatter plot
        self.dr_result_existed = False
        # udpate class selection
        self.class_selection = text.lower()
        # update the data indices according to class selection
        self.cur_class_indices = []
        if self.class_selection == 'all':
            self.cur_class_indices = list(range(len(self.point_cloud_paths)))
        else:
            for i in range(len(self.point_cloud_paths)):
                if self.point_cloud_paths[i][0] == self.class_selection:
                    self.cur_class_indices.append(i)

        # update aggregate nero plot
        self.draw_point_cloud_aggregate_nero()
        # update dr plot
        self.draw_dr_plot()
        # update visualization of selected point cloud
        self.draw_point_cloud()
        # update individiaul NERO plot
        self.draw_point_cloud_individual_nero()
        # update detail plot
        self.draw_point_cloud_detail_plot()

    # drop down menu that let user select model 1
    @QtCore.Slot()
    def _model_1_selection_changed(self, text):
        print('Model 1:', text)
        self.model_1_name = text
        # Original or Data
        self.model_1_cache_name = self.model_1_name.split(' ')[0]
        # load the model
        self.model_1 = self.load_point_cloud_model(self.model_1_name)
        # update aggregate nero plot
        self.draw_point_cloud_aggregate_nero()
        # update dr plot
        self.draw_dr_plot()
        # update visualization of selected point cloud
        self.draw_point_cloud()
        # update individiaul NERO plot
        self.draw_point_cloud_individual_nero()
        # update detail plot
        self.draw_point_cloud_detail_plot()

    # drop down menu that lets user select model 2
    @QtCore.Slot()
    def _model_2_selection_changed(self, text):
        print('Model 2:', text)
        self.model_2_name = text
        # Original or Data
        self.model_2_cache_name = self.model_2_name.split(' ')[0]
        # load the model
        self.model_2 = self.load_point_cloud_model(self.model_2_name)
        # update aggregate nero plot
        self.draw_point_cloud_aggregate_nero()
        # update dr plot
        self.draw_dr_plot()
        # update visualization of selected point cloud
        self.draw_point_cloud()
        # update individiaul NERO plot
        self.draw_point_cloud_individual_nero()
        # update detail plot
        self.draw_point_cloud_detail_plot()

    # drop down menu that lets user select NERO metric
    @QtCore.Slot()
    def _nero_metric_changed(self, text):
        print(f'\nNERO metric changed to {text}')
        # save the selection
        self.cur_metric = text
        # update aggregate nero plot
        self.draw_point_cloud_aggregate_nero()
        # update dr plot
        self.draw_dr_plot()
        # update visualization of selected point cloud
        self.draw_point_cloud()
        # update individiaul NERO plot
        self.draw_point_cloud_individual_nero()
        # update detail plot
        self.draw_point_cloud_detail_plot()

    @QtCore.Slot()
    def _view_plane_changed(self, text):
        print(f'\nViewing plane changed to {text}')
        self.cur_plane = text
        # update aggregate nero plot
        self.draw_point_cloud_aggregate_nero()
        # update dr plot
        self.draw_dr_plot()
        # update visualization of selected point cloud
        self.draw_point_cloud()
        # update individiaul NERO plot
        self.draw_point_cloud_individual_nero()
        # update detail plot
        self.draw_point_cloud_detail_plot()

    # @QtCore.Slot()
    # def _realtime_inference_checkbox_clicked(self):
    #     if self.realtime_inference_checkbox.isChecked():
    #         self.realtime_inference = True
    #     else:
    #         self.realtime_inference = False

    @QtCore.Slot()
    # change different dimension reduction algorithms
    def _dr_selection_changed(self, text):
        # update dimension reduction algorithm
        self.cur_dr_algorithm = text
        print(f'\nDR algorithm changed to {self.cur_dr_algorithm}')
        # update dr plot
        self.draw_dr_plot()

    # radio buttons on choosing quantity used to compute intensity
    @QtCore.Slot()
    def _mean_encoding_button_clicked(self):
        self.intensity_method = 'mean'
        print(f'\nDR plots color encoded based on {self.intensity_method}')
        # update dr plot
        self.draw_dr_plot()

    @QtCore.Slot()
    def _variance_encoding_button_clicked(self):
        self.intensity_method = 'variance'
        print(f'\nDR plots color encoded based on {self.intensity_method}')
        # update dr plot
        self.draw_dr_plot()

    # update slider 1 text
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

    # update slider 2 text
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
            # current index from slider
            self.slider_1_selected_index = self.dr_result_selection_slider_1.value()
            # get the clicked scatter item's information
            self.point_cloud_index = self.sorted_class_indices_1[self.slider_1_selected_index]
            print(
                f'\nSlider 1 image index {self.point_cloud_index}, ranked position {self.slider_1_selected_index}'
            )
            # update the text
            self._update_slider_1_text()

            # change the other slider's value
            # the other slider should be locked so that
            # _dr_result_selection_slider_2_changed is not triggered
            self.slider_2_locked = True
            self.slider_2_selected_index = self.sorted_class_indices_2.index(
                self.point_cloud_index
            )
            self.dr_result_selection_slider_2.setValue(self.slider_2_selected_index)
            # update the text
            self._update_slider_2_text()
            self.slider_2_locked = False

            # update dr plot because the highlight is changed
            self.draw_dr_plot()

            # get the corresponding point cloud data path
            self.point_cloud_path = self.point_cloud_paths[self.point_cloud_index][1]
            print(f'\nSelected point cloud at {self.point_cloud_path}')

            # visualize point cloud
            self.draw_point_cloud()
            # individiaul NERO plot
            self.draw_point_cloud_individual_nero()
            # detail plot
            self.draw_point_cloud_detail_plot()

    # slider for dr plot 2
    @QtCore.Slot()
    def _dr_result_selection_slider_2_changed(self):
        # when the slider bar is changed directly by user, it is unlocked
        # mimics that a point has been clicked
        if not self.slider_2_locked:
            # change the ranking in the other colorbar
            self.slider_2_selected_index = self.dr_result_selection_slider_2.value()
            # get the clicked scatter item's information
            self.point_cloud_index = self.sorted_class_indices_2[self.slider_2_selected_index]
            print(
                f'\nSlider 2 image index {self.point_cloud_index}, ranked position {self.slider_2_selected_index}'
            )
            # update the text
            self._update_slider_2_text()

            # change the other slider's value
            # the other slider should be locked so that
            # _dr_result_selection_slider_1_changed is not triggered
            self.slider_1_locked = True
            self.slider_1_selected_index = self.sorted_class_indices_1.index(
                self.point_cloud_index
            )
            self.dr_result_selection_slider_1.setValue(self.slider_1_selected_index)
            # update the text
            self._update_slider_1_text()
            self.slider_1_locked = False

            # update dr plot because the highlight is changed
            self.draw_dr_plot()

            # get the corresponding point cloud data path
            self.point_cloud_path = self.point_cloud_paths[self.point_cloud_index][1]
            print(f'\nSelected image at {self.point_cloud_path}')

            # visualize point cloud
            self.draw_point_cloud()
            # individiaul NERO plot
            self.draw_point_cloud_individual_nero()
            # detail plot
            self.draw_point_cloud_detail_plot()

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

    # when clicked on the scatter plot item
    @QtCore.Slot()
    def _low_dim_scatter_clicked(self, item=None, points=None):
        # get the clicked scatter item's information
        # when item is not none, it is from real click
        # if item != None:
        self.point_cloud_index = int(item.opts['name'])
        print(f'\nClicked image index {self.point_cloud_index}')

        # get the ranking in each colorbar and change its value while locking both sliders
        # slider 1
        self.slider_1_locked = True
        self.slider_2_locked = True
        self.slider_1_selected_index = self.sorted_class_indices_1.index(self.point_cloud_index)
        self.dr_result_selection_slider_1.setValue(self.slider_1_selected_index)
        # update the text
        self._update_slider_1_text()
        # slider 2
        self.slider_2_selected_index = self.sorted_class_indices_2.index(self.point_cloud_index)
        self.dr_result_selection_slider_2.setValue(self.slider_2_selected_index)
        # update the text
        self._update_slider_2_text()
        # update the indicator of current selected item
        self.draw_dr_plot()
        # unlock after changing the values
        self.slider_1_locked = False
        self.slider_2_locked = False

        # get the corresponding point cloud path
        self.point_cloud_path = self.point_cloud_paths[self.point_cloud_index][1]
        print(f'\nSelected point cloud at {self.point_cloud_path}')

        # visualize point cloud
        self.draw_point_cloud()
        # individiaul NERO plot
        self.draw_point_cloud_individual_nero()
        # detail plot
        self.draw_point_cloud_detail_plot()

    # when hovered on the scatter plot item
    @QtCore.Slot()
    def _low_dim_scatter_hovered(self, item, points):
        item.setToolTip(item.opts['hover_text'])

    # plot all the scatter items with brush color reflecting the intensity
    def _draw_scatter_plot(
        self,
        low_dim_scatter_plot,
        low_dim,
        sorted_intensity,
        sorted_class_indices,
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
        # if self.point_cloud_index != None:
        sorted_selected_index = sorted_class_indices.index(self.point_cloud_index)
        # else:
        #     sorted_selected_index = len(sorted_class_indices) - 1

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
        # add points to the item
        low_dim_scatter_item.setData(
            low_dim_point, name=str(sorted_class_indices[sorted_selected_index])
        )
        # connect click events on scatter items
        low_dim_scatter_item.sigClicked.connect(self._low_dim_scatter_clicked)
        low_dim_scatter_item.sigHovered.connect(self._low_dim_scatter_hovered)
        # add points to the plot
        low_dim_scatter_plot.addItem(low_dim_scatter_item)

    def _draw_individual_heatmap(self, data, heatmap, title=None):
        # viewbox that contains the heatmap
        view_box = pg.ViewBox(invertY=True)
        view_box.setAspectLocked(lock=True)
        # set up heatmap plot, image and viewbox
        heatmap.setImage(data)
        heatmap.setOpts(axisOrder='row-major')
        # connnect colorbar
        self.color_bar.setImageItem(heatmap)
        # add image to the viewbox
        view_box.addItem(heatmap)
        # so that displaying highlighter at the boundary does not jitter the plot
        view_box.disableAutoRange()
        # plot item that contains viewbox
        heatmap_plot = pg.PlotItem(viewBox=view_box, title=title)
        # disable being able to move plot around
        heatmap_plot.setMouseEnabled(x=False, y=False)
        # display arguments
        x_label_style = {'color': 'white'}   # white so it is not visible
        heatmap_plot.getAxis('bottom').setLabel(**x_label_style)
        heatmap_plot.getAxis('bottom').setStyle(tickLength=0, showValues=False)
        y_label_style = {'color': 'white'}   # white so it is not visible
        heatmap_plot.getAxis('left').setLabel(**y_label_style)
        heatmap_plot.getAxis('left').setStyle(tickLength=0, showValues=False)

        return heatmap_plot

    def _individual_nero_clicked(self):
        # convert to data array position
        self.click_image_x = self.click_pos_x // self.block_size
        self.click_image_y = self.click_pos_y // self.block_size
        # recover rotation axis angle and the actual rotate angle around that axis
        self.axis_angle_index, self.rot_angle_index = nero_interface_util.click_to_rotation(
            self.click_image_x,
            self.click_image_y,
            self.all_axis_angles,
            self.all_rot_angles,
        )
        print(
            f'Axis rotation: {self.all_axis_angles[self.axis_angle_index]} (index {self.axis_angle_index}), Angle rotation: {self.all_rot_angles[self.rot_angle_index]} (index {self.rot_angle_index})'
        )

        # update individual nero plot's highligher
        self.draw_point_cloud_individual_nero(highlighter_only=True)
        # plot detail plot
        self.draw_point_cloud_detail_plot()


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
