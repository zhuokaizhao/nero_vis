import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui



class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.button = QtWidgets.QPushButton('Select model path')
        self.text = QtWidgets.QLabel('Non-Equivariance Revealed on Orbits',
                                     alignment=QtCore.Qt.AlignCenter)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)

        self.button.clicked.connect(self.select_file_path)



    @QtCore.Slot()
    def select_file_path(self):
        QtWidgets.QFileDialog.getOpenFileName(self)



if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())
