
import numpy as np
# load the deepforest prebuilt crown model and run it on a single image
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt

class Grid_Layers:

    def __init__(self, point_a, point_b, image_path, rows, cols):
        '''
        Grid Layers class constructor:
        This class will generate and contain the layers of the grid for a given location / image

        Layer 1: deepforest tree density calculation grid
        Layer 2: slope calculation grid

        Args:
        point_a: tuple of x, y coordinates latitude and longitude
        point_b: tuple of x, y coordinates latitude and longitude
        image_path: image path to file tif or png
        rows: number of rows for the grid
        cols: number of columns for the grid


        Methods:
        - tree_grid: generates a rows x cols grid of tree density values between 0 and 1
        - slope_grid: generates a rows x cols grid of absolute slope values between 0 and 1
        - combine_layers: combines the two layers into a single grid using input weights


        '''

        self.point_a = point_a
        self.point_b = point_b
        self.image_path = image_path
        self.rows = rows
        self.cols = cols

    def tree_grid(self):
        '''
        Generate a grid of tree density values between 0 and 1 using deepforest model
        '''
        pass

    def slope_grid(self):
        '''
        Generate a grid of slope values between 0 and 1
        '''
        pass

    def combine_layers(self, tree_w, slope_w):
        '''
        Combine the two layers into a single grid using input weights
        '''
        pass


class Layer:

    def __init__(self, rows, cols):
        '''
        this object will contain the grid for a layer

        Args:
        rows: number of rows for the grid
        cols: number of columns for the grid

        '''
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols))





if __name__ == '__main__':

    # Create a layer object
    layer = Layer(10, 10)
    print(layer.grid)
    print(layer.rows)
    print(layer.cols)

