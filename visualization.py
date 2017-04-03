# Build-in libs
import sys
# Third-part libs
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
# Local dependencies
from som import *


class SomVisualization:

    def __init__(self, som):
        """
        Initialization of class SomVisualization, build basic app and window using Qt

        :param som: Self-organizing map object
        :return None
        """
        self.som = som

        self.app = QtGui.QApplication([])
        self.win = None
        self.lol = None
        # pg.setConfigOption('background', 'w')
        # pg.setConfigOption('antialias', 'True')
        # self.win = pg.GraphicsWindow(title="Self-organizing map")
        # self.win.resize(1000, 600)

    def colormap_build(self):
        """
        Create color mapping of self-organizing map
        """
        self.win = QtGui.QMainWindow()

        graphics_widget = pg.GraphicsLayoutWidget()

        self.win.setCentralWidget(graphics_widget)
        self.win.show()
        self.win.setWindowTitle("Color map")

        lattice = self.som.image_suitable_conversion()

        if self.som.prototype_dimension > 3:
            for i in range(0, self.som.prototype_dimension):
                graphics_widget.addLabel(text="Dimension number {0}".format(i+1))

                view = graphics_widget.addViewBox()
                view.setAspectLocked(True)

                img = pg.ImageItem(border='w')

                view.addItem(img)
                view.setRange(QtCore.QRectF(0, 0, lattice.shape[0], lattice.shape[1]))

                graphics_widget.nextRow()

                img.setImage(lattice[:, :, i], autoLevels=False)

        else:
            view = graphics_widget.addViewBox(row=0, col=0, name="Self-organizing map")
            view.setAspectLocked(True)

            img = pg.ImageItem(border='w')

            view.addItem(img)
            view.setRange(QtCore.QRectF(0, 0, lattice.shape[0], lattice.shape[1]))

            img.setImage(lattice, autoLevels=False)
        # TODO: Separate setup widget parameter and image update
        # print(graphics_widget.itemIndex(item=pg.ViewBox))

        self.show_window()

    # def color_map_show(self):
    #     """
    #
    #     :return:
    #     """
    #     # TODO: separate windows and widget creation

    def show_window(self):
        """
        Run
        """
        if not sys.flags.interactive or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def topology_map_build(self):
        """
        Topology map build
        :return:
        """
        self.win = QtGui.QMainWindow()

        graphics_widget = pg.GraphicsLayoutWidget()

        self.win.setCentralWidget(graphics_widget)
        self.win.show()
        self.win.setWindowTitle("Topology map")

        view = graphics_widget.addViewBox()
        view.setAspectLocked(True)

        graph_item = pg.GraphItem()
        view.addItem(graph_item)

        l_pen = pg.mkPen(color='w', width=3)
        graph_item.setData(pos=np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
                           adj=np.array([[0, 1], [0, 2], [2, 3]]), pen=l_pen, size=20)

        self.show_window()

    # def som_show(self):
    #
    # def som_interactive(self):


if __name__ == "__main__":
    map = SOM(map_size=(10, 10), proto_dim=2)
    data = np.random.rand(100, 2)
    lol = SomVisualization(map)
    # lol.colormap_build()
    lol.topology_map_build()



