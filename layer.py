
import numpy as np
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt
from PIL import Image
import rasterio

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

    def tree_grid(self, plot=False):
        '''
        Generate a grid of tree density values between 0 and 1 using deepforest model

        Args:
        plot: boolean, if True, plot the image and the bounding boxes

        Returns:
        Z: numpy array of shape (rows, cols, 1) with tree density values between 0 and 1
        '''
        model = main.deepforest()
        model.use_release()

        # if .tif file:
        if self.image_path.endswith('.tif'):
            # run the model
            predictions = model.predict_tile(raster_path=self.image_path, return_plot=False, patch_size=100, patch_overlap=0.1)

            # open tif image
            image = rasterio.open(image_path)
            image = image.read()
            image = np.moveaxis(image, 0, -1)
            image = image.squeeze()

        elif self.image_path.endswith('.png') or self.image_path.endswith('.jpg'):
            # run the model
            predictions = model.predict_image(image_path=self.image_path, return_plot=False)

            # open png or jpg image
            image = plt.imread(self.image_path)
        else:
            raise ValueError('Image file must be .tif, .png or .jpg')
        
        # get the tree density grid

        if plot:
            # plot the image .tif and the predictions
            fig, ax = plt.subplots(figsize=(20, 20))
            plt.imshow(image)
            plt.show()

            # plot the bounding boxes
            for i in range(len(predictions)):
                x, y, w, h = predictions["xmin"][i], predictions["ymin"][i], predictions["xmax"][i], predictions["ymax"][i]
                rect = plt.Rectangle((x, y), w-x, h-y, fill=False, color="red")
                ax.add_patch(rect)
                # scatter points on the image, scale the scatter points fit the same scale as the image
                ax.scatter(predictions["xmin"][i] + (predictions["xmax"][i] - predictions["xmin"][i]) / 2,
                            predictions["ymin"][i] + (predictions["ymax"][i] - predictions["ymin"][i]) / 2,   
                            color="blue", linewidths=2.5, edgecolors="blue")
                
        # transform the scatter into a density map
        # create a grid of points
        x_size = np.linspace(0, image.shape[1], image.shape[1])
        y_size = np.linspace(0, image.shape[0], image.shape[0])
        X, Y = np.meshgrid(x_size, y_size)
        # create a density map
        Z = np.zeros(X.shape)
        for i in range(len(predictions)):
            x, y = (predictions["xmin"][i] + predictions["xmax"][i]) / 2, (predictions["ymin"][i] + predictions["ymax"][i]) / 2
            Z += np.exp(-((X - x)**2 + (Y - y)**2) / 10000) # the 1000 is a hyperparameter that controls the spread of the density map

        # transform the density map into an array of density values shape (m,n,1)
        Z = Z.reshape(Z.shape[0], Z.shape[1], 1)

        # resize and interpolate the density map to the same number of pixels as the image as requested
        Z = np.array(Image.fromarray(Z.squeeze()).resize((self.rows, self.cols)))

        # normalize the density map
        Z = Z / np.max(Z)

        if plot:
            fig, ax = plt.subplots()
            ax.imshow(Z, aspect="auto")
            plt.show()
        
        # create Layer object
        tree_layer = Layer(self.rows, self.cols)
        tree_layer.grid = Z

        self.tree_layer = tree_layer

        return tree_layer



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

