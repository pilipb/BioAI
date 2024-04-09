
import numpy as np
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
import ee

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

    def get_image(self):
        '''
        Get the image from point A to point B using Google Maps API


        '''

        ee.Authenticate()
        ee.Initialize()
        
        # get the bounding box of the image
        coord_A = ee.Geometry.Point(self.point_a[0], self.point_a[1])
        coord_B = ee.Geometry.Point(self.point_b[0], self.point_b[1])

        centre_coords = [ (self.point_a[0] + self.point_b[0]) / 2, (self.point_a[1] + self.point_b[1]) / 2]

        # create a bounding box that has a and b as two opposite corners of the rectangle
        # the max and min of the coordinates are used to create the rectangle
        lat_min = min(coord_A.getInfo().get("coordinates")[1], coord_B.getInfo().get("coordinates")[1])
        lat_max = max(coord_A.getInfo().get("coordinates")[1], coord_B.getInfo().get("coordinates")[1])
        lon_min = min(coord_A.getInfo().get("coordinates")[0], coord_B.getInfo().get("coordinates")[0])
        lon_max = max(coord_A.getInfo().get("coordinates")[0], coord_B.getInfo().get("coordinates")[0])

        # add a buffer to the bounding box
        buffer = 0.0005
        lat_min -= buffer
        lat_max += buffer
        lon_min -= buffer
        lon_max += buffer

        # make the bounding box a geometry object
        bounding_box = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        # extract satellite image of the area
        image_collection = ee.ImageCollection("COPERNICUS/S2")
        image_collection = image_collection.filterBounds(bounding_box)
        image_collection = image_collection.filterDate("2023-01-01", "2024-12-31")
        # image_collection = image_collection.sort()
        image = image_collection.first()

        # get the image
        image = image.visualize(**{
            "bands": ["B4", "B3", "B2"],
            "min": 1000,
            "max": 5000,
        })

        # resample the image with a resolution of 1m
        res = 0.1
        image = image.resample("bilinear").reproject(crs= image.projection(), scale=res)


        # extract values from the image
        image = image.updateMask(image.mask().reduce("min"))


        # show the image with ee
        print(ee.Image(image).getThumbUrl({
            "region": bounding_box.getInfo()["coordinates"],
            "dimensions": "1500x1500"
        }))


        elevation_clip = elevation.clip(bounding_box)

        res = 10

        resampled = elevation_clip.resample("bilinear").reproject(crs= elevation_clip.projection(), scale=res)

        opacity = 0.9

        vis_params = {
            "min": 0,
            "max": 300,
            "opacity": opacity
        }

        # get the image
        image = resampled.visualize(**vis_params)

        # show the image with ee
        print(ee.Image(image).getThumbUrl({
            "region": bounding_box.getInfo()["coordinates"],
            "dimensions": "1500x1500"
        }))
        pass

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

        dataset = ee.Image("CGIAR/SRTM90_V4")
        elevation = dataset.select("elevation")
        slope = ee.Terrain.slope(elevation)

        pass

    def river_grid(self):
        '''
        Identify where the rivers are is in the image and generate a grid of river values between 0 and 1

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

