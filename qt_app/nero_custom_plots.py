import numpy as np
import pyqtgraph as pg

# subclass of ImageItem that reimplements the control methods
class NEROHeatmap(pg.ImageItem):
    def __init__(self, interface, model_index, interaction=False, reaction_function=None):
        super().__init__()
        self.interface = interface
        self.model_index = model_index
        self.interaction = interaction
        self.function = reaction_function

    def mouseClickEvent(self, event):
        if self.interaction:
            # print(f'Clicked on heatmap at ({event.pos().x()}, {event.pos().y()})')
            # record these positions
            self.interface.click_pos_x = event.pos().x()
            self.interface.click_pos_y = event.pos().y()
            self.function()

    def mouseDragEvent(self, event):
        if self.interaction:
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

    # def hoverEvent(self, event):
    #     if not event.isExit():
    #         block_x = int(np.floor(event.pos().x() // self.interface.translation_step_single))
    #         block_y = int(np.floor(event.pos().y() // self.interface.translation_step_single))
    #         if self.model_index == 1:
    #             hover_text = str(
    #                 round(self.interface.cur_aggregate_plot_quantity_1[block_y][block_x], 3)
    #             )
    #         elif self.model_index == 2:
    #             hover_text = str(
    #                 round(self.interface.cur_aggregate_plot_quantity_2[block_y][block_x], 3)
    #             )

    #         self.setToolTip(hover_text)
