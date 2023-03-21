import numpy as np
import pyqtgraph as pg

# subclass of ImageItem that reimplements the control methods
class NEROHeatmap(pg.ImageItem):
    def __init__(self, interface, plot_type, model_index):
        super().__init__()
        self.interface = interface
        self.plot_type = plot_type
        self.model_index = model_index

    def mouseClickEvent(self, event):
        if self.plot_type == 'individual':
            print(f'Clicked on heatmap at ({event.pos().x()}, {event.pos().y()})')
            # the position of un-repeated aggregate result
            self.interface.block_x = int(
                np.floor(event.pos().x() // self.interface.translation_step_single)
            )
            self.interface.block_y = int(
                np.floor(event.pos().y() // self.interface.translation_step_single)
            )

            # in COCO mode, clicked location indicates translation
            # draw a point(rect) that represents current selection of location
            # although in this case we are taking results from the aggregate result
            # we need these locations for input modification
            self.interface.cur_x_tran = (
                int(np.floor(event.pos().x() // self.interface.translation_step_single))
                * self.interface.translation_step_single
            )
            self.interface.cur_y_tran = (
                int(np.floor(event.pos().y() // self.interface.translation_step_single))
                * self.interface.translation_step_single
            )
            self.interface.x_tran = self.interface.cur_x_tran + self.interface.x_translation[0]
            self.interface.y_tran = self.interface.cur_y_tran + self.interface.y_translation[0]

            # udpate the correct coco label
            self.interface.update_coco_label()

            # update the input image with FOV mask and ground truth labelling
            self.interface.display_coco_image()

            # redisplay model output (result taken from the aggregate results)
            if self.interface.data_mode == 'aggregate':
                self.interface.draw_model_output(take_from_aggregate_output=True)
            else:
                self.interface.draw_model_output()

            # remove existing selection indicater from both scatter plots
            self.interface.heatmap_plot_1.removeItem(self.interface.scatter_item_1)
            self.interface.heatmap_plot_2.removeItem(self.interface.scatter_item_2)

            # new scatter points
            scatter_point = [
                {
                    'pos': (
                        self.interface.cur_x_tran + self.interface.translation_step_single // 2,
                        self.interface.cur_y_tran + self.interface.translation_step_single // 2,
                    ),
                    'size': self.interface.translation_step_single,
                    'pen': {'color': 'red', 'width': 3},
                    'brush': (0, 0, 0, 0),
                }
            ]

            # add points to both views
            self.interface.scatter_item_1.setData(scatter_point)
            self.interface.scatter_item_2.setData(scatter_point)
            self.interface.heatmap_plot_1.addItem(self.interface.scatter_item_1)
            self.interface.heatmap_plot_2.addItem(self.interface.scatter_item_2)

    def mouseDragEvent(self, event):
        if self.plot_type == 'individual':
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
            block_x = int(np.floor(event.pos().x() // self.interface.translation_step_single))
            block_y = int(np.floor(event.pos().y() // self.interface.translation_step_single))
            if self.model_index == 1:
                hover_text = str(
                    round(self.interface.cur_aggregate_plot_quantity_1[block_y][block_x], 3)
                )
            elif self.model_index == 2:
                hover_text = str(
                    round(self.interface.cur_aggregate_plot_quantity_2[block_y][block_x], 3)
                )

            self.setToolTip(hover_text)
