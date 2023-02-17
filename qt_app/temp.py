# import sys
# from PySide6 import QtCore, QtGui, QtWidgets
# from PySide6.QtCore import Qt
# import numpy as np


# class TableModel(QtCore.QAbstractTableModel):

#     def __init__(self, data):
#         super(TableModel, self).__init__()
#         self._data = data

#     def data(self, index, role):
#         if role == Qt.DisplayRole:
#             # Note: self._data[index.row()][index.column()] will also work
#             value = self._data[index.row(), index.column()]
#             return str(value)

#     def rowCount(self, index):
#         return self._data.shape[0]

#     def columnCount(self, index):
#         return self._data.shape[1]


# class MainWindow(QtWidgets.QMainWindow):

#     def __init__(self):
#         super().__init__()

#         self.table = QtWidgets.QTableView()

#         data = np.array([
#           ['aaa', 9, 2],
#           ['bb', 0, -1],
#           [3, 5, 2],
#           [3, 3, 2],
#           [5, 8, 9],
#         ])

#         self.model = TableModel(data)
#         self.table.setModel(self.model)

#         self.setCentralWidget(self.table)


# app=QtWidgets.QApplication(sys.argv)
# window=MainWindow()
# window.show()
# app.exec_()

# https://pythonprogramminglanguage.com/pyqt-checkbox/
# import sys
# from PySide6 import QtCore, QtWidgets
# from PySide6.QtWidgets import QMainWindow, QLabel, QCheckBox, QWidget
# from PySide6.QtCore import QSize


# class ExampleWindow(QMainWindow):
#     def __init__(self):
#         QMainWindow.__init__(self)

#         self.setMinimumSize(QSize(140, 40))
#         self.setWindowTitle('Checkbox')

#         self.b = QCheckBox('Awesome?', self)
#         self.b.stateChanged.connect(self.clickBox)
#         self.b.move(20, 20)
#         self.b.resize(320, 40)

#     def clickBox(self, state):

#         if self.b.isChecked():
#             print('Checked')
#         else:
#             print('Unchecked')


# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     mainWin = ExampleWindow()
#     mainWin.show()
#     sys.exit(app.exec())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

aggregate_consensus = np.load('aggregate_consensus.npy', allow_pickle=True)
aggregate_consensus_outputs = np.load('aggregate_consensus_outputs.npy', allow_pickle=True)

cur_consensus = aggregate_consensus[0]
cur_consensus = [36.37101105, 34.63551742, 87.81043541, 89.07986373, 2.0]
print(cur_consensus)
print(len(aggregate_consensus_outputs))
all_x1 = aggregate_consensus_outputs[:, 0]
all_y1 = aggregate_consensus_outputs[:, 1]
all_x2 = aggregate_consensus_outputs[:, 2]
all_y2 = aggregate_consensus_outputs[:, 3]
print(all_x1[0], all_y1[0], all_x2[0], all_y2[0])

fig, ax = plt.subplots()
ax.scatter(
    (np.array(all_x1) + np.array(all_x2)) / 2,
    (np.array(all_y1) + np.array(all_y2)) / 2,
    c='red',
    alpha=0.1,
)
ax.scatter(np.array(all_y1), np.array(all_x1), c='green', alpha=0.1)
ax.scatter(np.array(all_y2), np.array(all_x2), c='blue', alpha=0.1)
consensus = patches.Rectangle(
    (cur_consensus[0], cur_consensus[1]),
    cur_consensus[2] - cur_consensus[0],
    cur_consensus[3] - cur_consensus[1],
    fill=None,
    color='yellow',
    linewidth=5,
)
average_box = patches.Rectangle(
    (np.mean(all_x1), np.mean(all_y1)),
    np.mean(all_x2) - np.mean(all_x1),
    np.mean(all_y2) - np.mean(all_y1),
    fill=None,
    color='orange',
    linewidth=5,
)
ax.add_patch(consensus)
ax.add_patch(average_box)
ax.vlines(x=63, ymin=-100, ymax=200)
ax.hlines(y=63, xmin=-100, xmax=200)

ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_title('Bounding boxes histgram')

ax.grid(True)
fig.tight_layout()
plt.gca().invert_yaxis()
plt.axis('equal')

plt.show()
