# TODO: DEBUG Current version of SOM

import numpy as np


class SOM:

    epoch_counter = 0

    def __init__(self, data: np.array, map_size: tuple = (10, 10), proto_dim: int = 3,
                 number_of_epochs: int = 100,
                 learning_rate_init: float = 0.1,
                 neighbourhood_radius_init: float or None = None,
                 neighbourhood_radius_time_const: float or None = None) -> None:
        """
        Initialized Self-organizing map parameter

        :param map_size: Width and height of self-organizing map
        :param proto_dim: Dimension of prototype (weight vector of neuron)
        :param number_of_epochs: Number of learning epoch
        :param learning_rate_init: Initial learning rate
        :param neighbourhood_radius_init: Initial neighbourhood radius
        :param neighbourhood_radius_time_const: Time constant of neighbourhood radius
        """
        ''' Codebook '''
        self.data = data
        '''Initial value for dimensions of the map'''
        self.map_size = map_size
        self.prototype_dimension = proto_dim

        self.number_of_learning_epoch = number_of_epochs

        '''Initial parameter for learning rate'''
        self.learning_rate_init = learning_rate_init

        '''Initial parameter for neighbourhood radius'''
        if neighbourhood_radius_init is not None:
            self.neighbourhood_radius_init = neighbourhood_radius_init
        else:
            self.neighbourhood_radius_init = max(map_size)//2

        if neighbourhood_radius_time_const is not None:
            self.neighbourhood_radius_time_const = neighbourhood_radius_time_const
        else:
            self.neighbourhood_radius_time_const = number_of_epochs / \
                                                   np.log(self.neighbourhood_radius_init)
        '''Initial lattice configuration (only random initialization mode) '''
        self.lattice = np.random.rand(self.map_size[0], self.map_size[1], self.prototype_dimension)
        '''Initial connection matrix'''
        self.connection_array = []
        self.connection_init()

    def connection_init(self):
        """
        Connection matrix initialization

        :return: None
        """
        row_size = self.lattice.shape[0]
        for i, j in np.ndindex(self.map_size):
            if i == 0:
                if j == 0:
                    self.connection_array.extend([np.array([j, (i+1) * row_size + j, 1.0]), np.array([j, j + 1, 1.0])])
                elif j == (self.map_size[1]-1):
                    self.connection_array.extend([np.array([j, (i+1) * row_size + j, 1.0]), np.array([j, j - 1, 1.0])])
                else:
                    self.connection_array.extend([np.array([j, (i+1) * row_size + j, 1.0]), np.array([j, j + 1, 1.0]),
                                                  np.array([j, j - 1, 1.0])])
            elif i == (self.map_size[0]-1):
                if j == 0:
                    self.connection_array.extend([np.array([j, (i-1) * row_size + j, 1.0]), np.array([j, j + 1, 1.0])])
                elif j == (self.map_size[1]-1):
                    self.connection_array.extend([np.array([j, (i-1) * row_size + j, 1.0]), np.array([j, j - 1, 1.0])])
                else:
                    self.connection_array.extend([np.array([j, (i-1) * row_size + j, 1.0]), np.array([j, j + 1, 1.0]),
                                                  np.array([j, j - 1, 1.0])])
            else:
                if j == 0:
                    self.connection_array.extend([np.array([j, (i-1) * row_size + j, 1.0]), np.array([j, j + 1, 1.0]),
                                                  np.array([j, (i+1) * row_size + j, 1.0])])
                elif j == (self.map_size[1]-1):
                    self.connection_array.extend([np.array([j, (i-1) * row_size + j, 1.0]), np.array([j, j - 1, 1.0]),
                                                  np.array([j, (i+1) * row_size + j, 1.0])])
                else:
                    self.connection_array.extend([np.array([j, (i-1) * row_size + j, 1.0]), np.array([j, j + 1, 1.0]),
                                                  np.array([j, (i+1) * row_size + j, 1.0]), np.array([j, j - 1, 1.0])])

        self.connection_array = np.array(self.connection_array)

    def neighbourhood_radius(self, epoch: int) -> float:
        """
        Calculate current neighbourhood radius with respect to epoch

        :param epoch: Current epoch of learning
        :return: Return neighbourhood radius on the specific epoch
        """
        return self.neighbourhood_radius_init * np.exp(-epoch/self.neighbourhood_radius_time_const)

    def learning_rate(self, epoch: int) -> float:
        """
        Calculate current learning rate with respect to epoch

        :param epoch: Current epoch of learning
        :return: Return learning rate on the specific epoch
        """
        return self.learning_rate_init * np.exp(-epoch/self.number_of_learning_epoch)

    def recall(self, input_vector: np.array) -> tuple:
        """
        Perform RECALL phase of SOM

        :param input_vector: Vector provided form data set
        :return: Return indices of Best Matching Unit (BMU)
        """
        dist = np.linalg.norm(self.lattice - input_vector, axis=2)
        indices = np.where(dist == dist.min())
        return indices[0][0], indices[1][0]

    def update_step(self, input_vector: np.array, learning_rate: float, neighbourhood_radius: float) -> None:
        """
        Update prototypes with single input vector according to learning rate and neighbourhood radius

        :param input_vector: Vector provided from data set
        :param learning_rate: Learning rate on the specific epoch
        :param neighbourhood_radius: Neighbourhood radius on the specific epoch
        :return None
        """
        best_match = self.recall(input_vector)
        for i, j in np.ndindex(self.map_size):
            dx = i - best_match[0]
            dy = j - best_match[1]
            exp_power = np.exp(-(dx * dx + dy * dy) / (2 * neighbourhood_radius * neighbourhood_radius))
            self.lattice[i][j] += learning_rate * exp_power * (input_vector - self.lattice[i][j])

    def update_epoch(self) -> None:
        """
        Update prototype within epoch

        :return None
        """
        learning_rate_current = self.learning_rate(self.epoch_counter)
        neighbourhood_radius_current = self.neighbourhood_radius(self.epoch_counter)

        '''Order of input data vectors in which they will be processed by SOM'''
        vector_sequence = np.arange(0, len(self.data))
        np.random.shuffle(vector_sequence)

        for single in vector_sequence:
            self.update_step(self.data[single], learning_rate_current, neighbourhood_radius_current)

        self.epoch_counter += 1

    def learn(self) -> None:
        """
        Perform LEARNING phase of SOM

        :return None
        """
        for i in range(0, self.number_of_learning_epoch):
            self.update_epoch()

    def image_suitable_conversion(self):
        """
        Convert prototype vectors to uint8 representation in range of [0, 255] in order
        to visualize lattice

        :return: re-arranged prototype vectors
        :rtype: np.array, dtype = np.uint8
        """
        temp = self.lattice - self.lattice.min()
        temp /= temp.max()
        temp *= 255
        if self.prototype_dimension == 2:
            temp = np.dstack((temp, np.zeros((self.map_size[0], self.map_size[1], 1))))
        temp = temp.astype(np.uint8)
        return temp
