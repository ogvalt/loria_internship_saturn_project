from os import listdir
import concurrent.futures

# import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_video(stream, max_frame_number=100, frame_width=180):
    """
    Extract amount of movement and amount of edges from video

    :param stream: Video stream
     :type stream: cv2.VideoCapture
    :param max_frame_number: Maximal number of frame that will be read
     :type max_frame_number: int
    :param frame_width: Width to which each frame should be resized
     :type frame_width: int
    :return: Amount of edges and amount of movement
     :rtype: tuple(np.array, np.array)
    """
    counter = 0
    previous_frame_dog = None
    data_edges = np.array([])
    data_movement = np.array([])

    width = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    resize_ratio = frame_width / width

    while counter < max_frame_number:
        counter += 1

        (grabbed, frame) = stream.read()

        if not grabbed:
            break

        frame = cv2.resize(frame, dsize=(0, 0), fx=resize_ratio,
                           fy=resize_ratio, interpolation=cv2.INTER_CUBIC)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (35, 35), 0)

        g1 = cv2.GaussianBlur(frame_gray, (1, 1), 0)
        g2 = cv2.GaussianBlur(frame_gray, (3, 3), 0)
        frame_dog = g1 - g2

        amount_of_edges = cv2.countNonZero(frame_dog)
        data_edges = np.append(data_edges, amount_of_edges)

        frame_dog = cv2.threshold(frame_dog, 254, 255, cv2.THRESH_BINARY)[1]

        if previous_frame_dog is None:
            previous_frame_dog = frame_dog.copy()

        frame_temporal = cv2.subtract(frame_dog, previous_frame_dog)

        amount_of_movement = cv2.countNonZero(frame_temporal)
        data_movement = np.append(data_movement, amount_of_movement)

        previous_frame_dog = frame_dog.copy()

    data_set = np.vstack([data_edges, data_movement])
    data_set = data_set.T
    data_set = np.divide(data_set, data_set.max(axis=0))

    return data_set


def get_video_parameter(stream):
    """
    Return video parameter, such as: width, height, total number of frame
    :param stream: Video stream
    :type stream: cv2.VideoCapture
    :return: video_width, video_height, number_of_frame
    """
    video_width = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return video_width, video_height


class DataSet:
    path = "data"
    file_list = None
    data_set = None

    def __init__(self, path=None):
        """
        Initiate class instance
        :param path: Path to the directory with videos, default path = "data"
        :type path: str
        """
        if path is not None and isinstance(path, str):
            self.path = path

        self.file_list = listdir(self.path)
        self.process_files()

    def process_files(self):
        """
        Process files in order to extract amount of movement and edges

        :return None
        """
        streams = list(map(lambda file: cv2.VideoCapture(self.path+'/'+file), self.file_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            self.data_set = list(executor.map(process_video, streams))
        map(lambda stream: stream.release(), streams)

    def merge_data_set(self):
        """
        Merge all data from separate items in data_set into one large set

        :return: Merged data
        """
        return np.vstack((i for i in self.data_set))

    def visualize_data(self):
        """
        Visualization of obtained data

        :return: Display plotted data
        """
        plt.figure(figsize=(12, 6))
        number_of_plots = len(self.data_set) + 1
        for i in range(0, number_of_plots - 1):
            plt.subplot(3, np.ceil(number_of_plots/3), i+1)
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.plot(self.data_set[i][:, 0], self.data_set[i][:, 1], 'bo')

        merged_set = self.merge_data_set()
        plt.subplot(3, np.ceil(number_of_plots / 3), number_of_plots)
        plt.plot(merged_set[:, 0], merged_set[:, 1], 'bo')
        plt.savefig("result.png")
        plt.show()


class ArtificialDataSet:
    """
    Class that generate artificial data set with normal distribution
    """

    dataset = None

    def __init__(self, size_of_set, dimentionality=3):
        self.size = size_of_set
        self.dim = dimentionality

    def generate_set(self):
        set1 = np.random.normal(loc=0.35, scale=0.35, size=(self.size, self.dim))
        set2 = np.random.normal(loc=0.65, scale=0.35, size=(self.size, self.dim))
        set3 = np.random.normal(loc=0.50, scale=0.65, size=(self.size, self.dim))

        self.dataset = np.append(set1, set2, axis=0)
        self.dataset = np.append(self.dataset, set3, axis=0)
        np.random.shuffle(self.dataset)
        self.dataset = self.dataset[0:self.size]

        return self.dataset


if __name__ == "__main__":
    lol = ArtificialDataSet(100, 3)
    data = lol.generate_set()
    print(data)