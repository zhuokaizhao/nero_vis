import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui  import QPixmap, QFont
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QFileDialog, QWidget, QLabel, QRadioButton
from PySide6.QtCharts import QBarSet, QBarSeries, QChart, QBarCategoryAxis, QValueAxis, QChartView, QPolarChart, QScatterSeries

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
        self.layout.setContentsMargins(50, 50, 50, 50)

        # individual laytout for different widgets
        # title
        self.title_layout = QtWidgets.QVBoxLayout()
        self.title_layout.setContentsMargins(0, 0, 0, 20)
        # mode selection radio buttons
        self.mode_layout = QtWidgets.QHBoxLayout()
        self.mode_layout.setContentsMargins(50, 0, 0, 50)
        # image and model loading buttons and drop down menus
        self.load_button_layout = QtWidgets.QHBoxLayout()
        self.load_button_layout.setContentsMargins(0, 0, 0, 0)

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
        self.title.setFont(QFont('Helvetica', 24))
        self.title_layout.addWidget(self.title)
        self.title_layout.setContentsMargins(0, 0, 0, 50)

        # radio buttons on mode selection (digit_recognition, object detection, PIV)
        # pixmap on text telling what this is
        # use the loaded_layout
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
        self.mode_layout.addWidget(self.mode_label)

        # radio_buttons_layout = QtWidgets.QGridLayout(self)
        self.radio_button_1 = QRadioButton('Digit recognition')
        self.radio_button_1.setChecked(True)
        self.radio_button_1.setStyleSheet('QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};')
        self.radio_button_1.pressed.connect(self.digit_reconition_button_clicked)
        self.mode_layout.addWidget(self.radio_button_1)
        # spacer item
        self.mode_layout.addSpacing(30)

        self.radio_button_2 = QRadioButton('Object detection')
        self.radio_button_2.setChecked(False)
        self.radio_button_2.setStyleSheet('QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};')
        self.radio_button_2.pressed.connect(self.object_detection_button_clicked)
        self.mode_layout.addWidget(self.radio_button_2)
        # spacer item
        self.mode_layout.addSpacing(30)

        self.radio_button_3 = QRadioButton('Particle Image Velocimetry (PIV)')
        self.radio_button_3.setChecked(False)
        self.radio_button_3.setStyleSheet('QRadioButton{font: 18pt Helvetica;} QRadioButton::indicator { width: 18px; height: 18px;};')
        self.radio_button_3.pressed.connect(self.piv_button_clicked)
        self.mode_layout.addWidget(self.radio_button_3)

        # default app mode is digit_recognition
        self.mode = 'digit_recognition'
        self.display_image_size = 150

        # load data button
        self.data_button = QtWidgets.QPushButton('Load Test Image')
        self.data_button.setStyleSheet('font-size: 18px')
        data_button_size = QtCore.QSize(500, 50)
        self.data_button.setMinimumSize(data_button_size)
        self.load_button_layout.addWidget(self.data_button)
        self.data_button.clicked.connect(self.load_image_clicked)

        # load models choices
        # model 1
        # graphic representation
        self.model_1_label = QLabel(self)
        self.model_1_label.setAlignment(QtCore.Qt.AlignCenter)
        model_1_icon = QPixmap(25, 25)
        model_1_icon.fill(QtCore.Qt.white)
        # draw model representation
        painter = QtGui.QPainter(model_1_icon)
        self.draw_circle(painter, 12, 12, 10, 'blue')

        # spacer item
        self.load_button_layout.addSpacing(30)

        model_1_menu = QtWidgets.QComboBox()
        model_1_menu.setMinimumSize(QtCore.QSize(250, 50))
        model_1_menu.setStyleSheet('font-size: 18px')
        if self.mode == 'digit_recognition':
            model_1_menu.addItem(model_1_icon, 'Simple model')
            model_1_menu.addItem(model_1_icon, 'E2CNN model')
            model_1_menu.addItem(model_1_icon, 'Data augmentation model')
            model_1_menu.setCurrentText('Simple model')

        # connect the drop down menu with actions
        model_1_menu.currentTextChanged.connect(self.model_1_text_changed)
        model_1_menu.setEditable(True)
        model_1_menu.lineEdit().setReadOnly(True)
        model_1_menu.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.load_button_layout.addWidget(model_1_menu)

        # model 2
        # graphic representation
        self.model_2_label = QLabel(self)
        self.model_2_label.setAlignment(QtCore.Qt.AlignCenter)
        model_2_icon = QPixmap(25, 25)
        model_2_icon.fill(QtCore.Qt.white)
        # draw model representation
        painter = QtGui.QPainter(model_2_icon)
        self.draw_circle(painter, 12, 12, 10, 'Green')

        # spacer item
        self.load_button_layout.addSpacing(30)

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

        # connect the drop down menu with actions
        model_2_menu.currentTextChanged.connect(self.model_2_text_changed)
        self.load_button_layout.addWidget(model_2_menu)

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
        self.model_2_name = 'Data augmentation model'
        self.model_2_path = glob.glob(os.path.join(os.getcwd(), 'example_models', self.mode, 'aug_rot_eqv', '*.pt'))[0]
        # preload model
        self.model_1 = nero_run_model.load_model('non-eqv', self.model_1_path)
        self.model_2 = nero_run_model.load_model('aug-eqv', self.model_2_path)

        # flag to check if an image has been displayed
        self.image_existed = False
        self.run_button_existed = False
        self.result_existed = False
        self.warning_existed = False

        # image (input data) modification mode
        self.translation = False
        self.rotation = False

        # rotation angles
        self.cur_rotation_angle = 0
        self.prev_rotation_angle = 0

        # result of transformed data
        self.all_quantities_1 = []
        self.all_quantities_2 = []

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
        if not self.run_button_existed:
            # run button
            self.run_button = QtWidgets.QPushButton('Analyze model')
            self.run_button_layout.addWidget(self.run_button)
            self.run_button.clicked.connect(self.run_button_clicked)

            self.run_button_existed = True

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
    def run_button_clicked(self):
        # run model once and display results
        self.run_model_once()

        # run model all and display results
        self.run_model_all()

    # run model on a single test sample
    def run_model_once(self):
        if self.mode == 'digit_recognition':
            self.output_1 = nero_run_model.run_mnist_once(self.model_1, self.cur_images_pt[0])
            self.output_2 = nero_run_model.run_mnist_once(self.model_2, self.cur_images_pt[0])

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

    # run model on all the available transformations on a single sample
    def run_model_all(self):
        if self.mode == 'digit_recognition':
            self.all_angles = []
            self.all_quantities_1 = []
            self.all_quantities_2 = []
            # run all rotation test with 5 degree increment
            for degree in range(0, 365, 5):
                self.cur_rotation_angle = -degree
                print(f'\nRotated {self.cur_rotation_angle} degrees')
                self.all_angles.append(self.cur_rotation_angle)
                # rotate the image tensor
                self.cur_images_pt[0] = nero_transform.rotate_mnist_image(self.loaded_images_pt[0], self.cur_rotation_angle)
                # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(angle), QtCore.Qt.SmoothTransformation)
                # convert image tensor to qt image and resize for display
                self.display_images[0] = nero_utilities.tensor_to_qt_image(self.cur_images_pt[0]).scaledToWidth(self.display_image_size)
                # update the pixmap and label
                image_pixmap = QPixmap(self.display_images[0])
                self.image_label.setPixmap(image_pixmap)
                # force repaint
                self.image_label.repaint()

                # sleep for 0.1 second
                # time.sleep(0.05)

                # update the model output
                self.output_1 = nero_run_model.run_mnist_once(self.model_1, self.cur_images_pt[0])
                self.output_2 = nero_run_model.run_mnist_once(self.model_2, self.cur_images_pt[0])

                # plotting the quantity regarding the correct label
                quantity_1 = self.output_1[self.loaded_image_labels[0]]
                quantity_2 = self.output_2[self.loaded_image_labels[0]]
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

    # function used when displaying model representation
    def draw_circle(self, painter, center_x, center_y, radius, color):

        # paint.begin(self)
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


    # draw arrow
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

    # draw a polar plot
    def draw_polar(self, plot):
        # plot = pg.plot()
        plot.setAspectLocked()

        # Add polar grid lines
        plot.addLine(x=0, pen=10)
        plot.addLine(y=0, pen=10)
        for r in np.arange(0, 1, 0.2):
            circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
            circle.setPen(pg.mkPen(2))
            plot.addItem(circle)

        return plot

    def display_mnist_result(self, mode, boundary_width):

        # use the loaded_layout
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
        self.loaded_layout.addWidget(self.mnist_label, 0, 1)

        # draw result using bar plot
        if mode == 'bar':
            bar_plot = pg.plot()
            bar_plot.setBackground('w')
            graph_1 = pg.BarGraphItem(x=np.arange(len(self.output_1))-0.2, height = list(self.output_1), width = 0.4, brush ='blue')
            graph_2 = pg.BarGraphItem(x=np.arange(len(self.output_1))+0.2, height = list(self.output_2), width = 0.4, brush ='green')
            bar_plot.addItem(graph_1)
            bar_plot.addItem(graph_2)

            self.loaded_layout.addWidget(bar_plot, 0, 2)

            # # bar series
            # bar_series = QBarSeries()
            # bar_series.append(model_1_set)
            # bar_series.append(model_2_set)

            # chart = QChart()
            # chart.addSeries(bar_series)
            # # animation when plotting
            # if self.repaint:
            #     chart.setAnimationOptions(QChart.NoAnimation)
            # else:
            #     chart.setAnimationOptions(QChart.SeriesAnimations)

            # # x and y axis
            # axis_x = QBarCategoryAxis()
            # axis_x.append(categories)
            # chart.setAxisX(axis_x, bar_series)
            # axis_y = QValueAxis()
            # chart.setAxisY(axis_y, bar_series)
            # axis_y.setRange(0, 1)

            # # legend
            # chart.legend().setVisible(False)
            # chart.legend().setAlignment(QtGui.Qt.AlignBottom)

            # # create chart view
            # self.chart_view = QChartView(chart)
            # self.chart_view.setRenderHint(QtGui.QPainter.Antialiasing)

            # # add to the layout
            # self.loaded_layout.addWidget(self.chart_view, 0, 2)

        elif mode == 'polar':
            polar_view = pg.GraphicsLayoutWidget()
            polar_view.setBackground('w')
            polar_plot = polar_view.addPlot()
            polar_plot = self.draw_polar(polar_plot)

            # helper function for clicking inside polar plot
            self.lastClicked = []
            def clicked(plot, points):

                # global lastClicked
                for p in self.lastClicked:
                    p.resetPen()

                # clicked point's position
                x_pos = points[0].pos().x()
                y_pos = points[0].pos().y()

                # convert back to polar coordinate
                radius = np.sqrt(x_pos**2+y_pos**2)
                theta = np.arctan2(y_pos, x_pos)*np.pi
                print(f'clicked point with r = {radius}, theta = {theta}')

                for p in points:
                    p.setPen('b', width=2)

                self.lastClicked = points

            # Set pxMode=False to allow spots to transform with the view
            # all the points to be plotted
            scatter_items = pg.ScatterPlotItem(pxMode=False)
            all_points_1 = []
            all_points_2 = []
            for i in range(len(self.all_angles)):
                degree = self.all_angles[i]
                # model 1 quantity
                cur_quantity_1 = self.all_quantities_1[i]
                # Transform to cartesian and plot
                x_1 = cur_quantity_1 * np.cos(degree)
                y_1 = cur_quantity_1 * np.sin(degree)
                all_points_1.append({'pos': (x_1, y_1),
                                    'size': 0.05,
                                    'pen': {'color': 'w', 'width': 0.1},
                                    'brush': QtGui.QColor('blue')})

                # model 2 quantity
                cur_quantity_2 = self.all_quantities_2[i]
                # Transform to cartesian and plot
                x_2 = cur_quantity_2 * np.cos(degree)
                y_2 = cur_quantity_2 * np.sin(degree)
                all_points_2.append({'pos': (x_2, y_2),
                                    'size': 0.05,
                                    'pen': {'color': 'w', 'width': 0.1},
                                    'brush': QtGui.QColor('green')})

            # add points to the item
            scatter_items.addPoints(all_points_1)
            scatter_items.addPoints(all_points_2)

            # add points to the plot and connect click events
            polar_plot.addItem(scatter_items)
            scatter_items.sigClicked.connect(clicked)

            # add to the layout
            self.loaded_layout.addWidget(polar_view, 0, 3)

            # marker_size = 8.0

            # scatter series
            # scatter_series_1 = QScatterSeries()
            # scatter_series_2 = QScatterSeries()
            # scatter_series_1.setMarkerSize(marker_size)
            # scatter_series_2.setMarkerSize(marker_size)
            # for i in range(len(self.all_quantities_1)):
            #     scatter_series_1.append(self.all_angles[i], self.all_quantities_1[i])
            #     scatter_series_2.append(self.all_angles[i], self.all_quantities_2[i])

            # chart = QPolarChart()
            # chart.addSeries(scatter_series_1)
            # chart.addSeries(scatter_series_2)

            # # create axis
            # angular_axis = QValueAxis()
            # # First and last ticks are co-located on 0/360 angle.
            # angular_axis.setTickCount(2)
            # angular_axis.setRange(0, 1)
            # angular_axis.setLabelFormat('%.1f')
            # # angular_axis.setShadesVisible(True)
            # # angular_axis.setShadesBrush(QtGui.QBrush(QtGui.QColor(249, 249, 255)))
            # chart.setAxisY(angular_axis)

            # radial_axis = QValueAxis()
            # radial_axis.setTickCount(9)
            # radial_axis.setLabelFormat('%d')
            # radial_axis.setRange(0, 360)
            # chart.setAxisX(radial_axis)

            # # animation when plotting
            # chart.setAnimationOptions(QChart.SeriesAnimations)

        else:
            raise Exception('Unsupported display mode')

        painter.end()

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
            print(f'\nRotated {self.cur_rotation_angle} degrees')
            # rotate the image tensor
            self.cur_images_pt[0] = nero_transform.rotate_mnist_image(self.loaded_images_pt[0], self.cur_rotation_angle)
            # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(angle), QtCore.Qt.SmoothTransformation)
            # convert image tensor to qt image and resize for display
            self.display_images[0] = nero_utilities.tensor_to_qt_image(self.cur_images_pt[0]).scaledToWidth(self.display_image_size)
            # update the pixmap and label
            image_pixmap = QPixmap(self.display_images[0])
            self.image_label.setPixmap(image_pixmap)

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
    widget.resize(1920, 1080)
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