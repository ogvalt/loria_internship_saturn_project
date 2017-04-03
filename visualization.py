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
        self.color_view = None
        self.topology_view = None
        self.som = som
        # Create Qt app
        self.app = QtGui.QApplication([])
        # Create main window
        self.win = QtGui.QMainWindow()
        pg.setConfigOption('antialias', 'True')
        self.win.resize(800, 400)

    def colormap_build(self):
        """
        Create color mapping of self-organizing map

        :return Widget with color map
        """
        graphics_widget = pg.GraphicsLayout()
        image_items = []

        if self.som.prototype_dimension > 3:
            for i in range(0, self.som.prototype_dimension):
                graphics_widget.addLabel(text="Dimension number {0}".format(i+1))

                view = graphics_widget.addViewBox()
                view.setAspectLocked(True)

                img = pg.ImageItem(border='w')
                image_items.append(img)

                view.addItem(img)
                view.setRange(QtCore.QRectF(0, 0, self.som.lattice.shape[0], self.som.lattice.shape[1]))

                graphics_widget.nextRow()

        else:
            view = graphics_widget.addViewBox(row=0, col=0, name="Self-organizing map")
            view.setAspectLocked(True)

            img = pg.ImageItem(border='w')
            image_items = img

            view.addItem(img)
            view.setRange(QtCore.QRectF(0, 0, self.som.lattice.shape[0], self.som.lattice.shape[1]))

        return graphics_widget, image_items

    def topology_map_build(self):
        """
        Topology map build

        :return: Widget with topology map
        """
        graphics_widget = pg.GraphicsLayout()

        view = graphics_widget.addViewBox()
        view.setAspectLocked(True)

        graph_item = pg.GraphItem()
        view.addItem(graph_item)

        l_pen = pg.mkPen(color='w', width=1)

        graph_item.setPen(l_pen)

        return graphics_widget, graph_item

    def main_widget_construct(self, options=None):
        """
        Construct main widget
        :param options: specify what SOM representation shows: None = both, 'color' - color map,
                        'topology' - topology map
        :type options: str

        :return: None
        """
        # Add GraphicsLayoutWidget that will contain all visualization
        main_widget = pg.GraphicsLayoutWidget()
        # Set main_widget as central widget
        self.win.setCentralWidget(main_widget)
        self.win.show()
        self.win.setWindowTitle("Self-organizing map")
        # Create color map layout
        color_map_layout, self.color_view = self.colormap_build()
        # Create topology map layout
        topology_map_layout, self.topology_view = self.topology_map_build()
        # Add layout to main widget
        if options == 'color':
            main_widget.addItem(color_map_layout)
        elif options == 'topology':
            main_widget.addItem(topology_map_layout)
        elif options is None:
            main_widget.addItem(color_map_layout)
            main_widget.addItem(topology_map_layout)
        else:
            raise Exception("Unknown options value, could be: None, 'color', 'topology' ")

    def update_data(self):
        """
        Update GUI with new data from SOM

        :return: None
        """
        self.topology_view.setData(pos=self.som.lattice.reshape(self.som.map_size[0] * self.som.map_size[1],
                                                                self.som.prototype_dimension),
                                   adj=self.som.connection_array[:, 0:2].astype(np.uint32), size=10)

        lattice = self.som.image_suitable_conversion()
        if isinstance(self.color_view, list):
            i = 0
            for item in self.color_view:
                item.setImage(lattice[:, :, i], autoLevels=False)
                i += 1
        else:
            self.color_view.setImage(lattice, autoLevels=False)

    def run(self):
        self.main_widget_construct()
        QtCore.QTimer.singleShot(1, self.update_data)


if __name__ == "__main__":
    som_map = SOM(map_size=(20, 20), proto_dim=5)
    data = np.random.rand(100, 2)
    lol = SomVisualization(som_map)
    lol.run()

    if not sys.flags.interactive or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



