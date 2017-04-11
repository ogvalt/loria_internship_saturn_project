# TODO: Spiking SOM

from pyqtgraph.Qt import QtGui, QtCore

import som
import visualization
import dataset

# Obtain data from video and visualize it
data_set_instance = dataset.DataSet()
data_set_instance.visualize_data()
data = data_set_instance.merge_data_set()
# Perform SOM training

som_map = som.SOM(data, proto_dim=2, learning_rate_init=0.001, save_after=10)
visual = visualization.SomVisualization(som=som_map)
visual.run()


