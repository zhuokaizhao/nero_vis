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
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignCenter)
        # left, top, right, and bottom margins
        self.layout.setContentsMargins(10, 10, 10, 10)
        # spacing
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)

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

        # initialize data loading and associated interface
        self.init_point_cloud_data()
        self.init_data_loading_interface()

        # initialize data loading and associated interface
        self.init_point_cloud_models()
        self.init_model_loading_interface()

        # prepare results NERO test results
        self.prepare_aggregate_results()

        # # prepare the interface and isplay
        # self.init_point_cloud_interface()

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
        self.layout.addWidget(self.model_label, 0, 0)

        # aggregate images loading drop down menu
        self.aggregate_image_menu = QtWidgets.QComboBox()
        self.aggregate_image_menu.setFixedSize(QtCore.QSize(220, 50))
        self.aggregate_image_menu.setStyleSheet(
            'color: black; font-size: 34px; font-family: Helvetica; font-style: normal;'
        )
        self.aggregate_image_menu.addItem('Input dataset')

        # load all images in the folder
        for i in range(len(self.all_data_paths)):
            self.aggregate_image_menu.addItem(self.all_data_paths[i].split('/')[-1].split('.')[0])

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
        self.layout.addLayout(aggregate_image_menu_layout, 0, 0)

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
        self.layout.addWidget(self.model_selection_label, 1, 2)
        self.layout.addLayout(model_menus_layout, 2, 2)

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

    ################## Aggregate NERO Plots Related ##################
    def prepare_aggregate_results(self):
        # TODO: create user interface for selecting planes
        self.all_planes = ['xy', 'xz', 'yz']
        self.cur_plane = 'xy'

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
