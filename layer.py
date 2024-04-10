
import numpy as np
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
import ee
import requests
import cv2

class Grid_Layers:

    def __init__(self, point_a, point_b, rows, cols, buffer_dist=100):
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
        self.rows = rows
        self.cols = cols

        ee.Authenticate()
        ee.Initialize()
        
        # get the bounding box of the image
        self.coord_A = ee.Geometry.Point(self.point_a[0], self.point_a[1])
        self.coord_B = ee.Geometry.Point(self.point_b[0], self.point_b[1])

        # create a bounding box that has a and b as two opposite corners of the rectangle
        # the max and min of the coordinates are used to create the rectangle
        self.lat_min = min(self.coord_A.getInfo().get("coordinates")[1], self.coord_B.getInfo().get("coordinates")[1])
        self.lat_max = max(self.coord_A.getInfo().get("coordinates")[1], self.coord_B.getInfo().get("coordinates")[1])
        self.lon_min = min(self.coord_A.getInfo().get("coordinates")[0], self.coord_B.getInfo().get("coordinates")[0])
        self.lon_max = max(self.coord_A.getInfo().get("coordinates")[0], self.coord_B.getInfo().get("coordinates")[0])

        # add a buffer to the bounding box
        buffer = buffer_dist / 111000 # 1 degree of latitude is 111000 meters
        self.lat_min -= buffer
        self.lat_max += buffer
        self.lon_min -= buffer
        self.lon_max += buffer

        self.box_width = self.lon_max - self.lon_min
        self.box_height = self.lat_max - self.lat_min

        # make the bounding box a geometry object
        self.bounding_box = ee.Geometry.Rectangle([self.lon_min, self.lat_min, self.lon_max, self.lat_max])

    def get_image(self, plot=False, resolution=0.5):
        '''
        Get the image from point A to point B using Google Maps API

        Args:
        plot: boolean, if True, plot the image and the bounding boxes
        resolution: resolution of the image in meters 
                    minimise the resolution to get a higher resolution image
                    however sometimes throws an error if the resolution is too low (depending on bounding box size)
        buffer_dist: buffer distance to add to the bounding box in meters

        Returns:
        image_path: path to the image
    

        '''

        
        # get the satellite image
        dataset = ee.ImageCollection("USDA/NAIP/DOQQ").filter(ee.Filter.date('2016-01-01', '2017-01-01')).mosaic()
        true_color = dataset.select(['R', 'G', 'B'])
        clipped_image = true_color.clip(self.bounding_box)

        # Multi-band GeoTIFF file.
        url = clipped_image.getDownloadUrl({
            'name': 'download_sat_image',
            'scale': resolution,
            'format': 'GeoTIFF',
        })

        response = requests.get(url)
        self.image_path = 'test_imgs/download.tif'

        with open('test_imgs/download.tif', 'wb') as fd:
            fd.write(response.content)

        if plot:
            image = rasterio.open(self.image_path)
            image = image.read()
            image = np.moveaxis(image, 0, -1)
            image = image.squeeze()

            # flip the image
            image = np.flip(image, axis=0)

            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.scatter((self.coord_A.getInfo().get("coordinates")[0] - self.lon_min) / self.box_width * image.shape[1], 
                        (self.coord_A.getInfo().get("coordinates")[1] - self.lat_min) / self.box_height * image.shape[0], color="red")
            ax.scatter((self.coord_B.getInfo().get("coordinates")[0] - self.lon_min) / self.box_width * image.shape[1],
                        (self.coord_B.getInfo().get("coordinates")[1] - self.lat_min) / self.box_height * image.shape[0], color="blue")
            plt.show()

        # give the points as pixel coordinates based on rows and cols
        self.pixel_A = (self.coord_A.getInfo().get("coordinates")[0] - self.lon_min) / self.box_width * self.cols, (self.coord_A.getInfo().get("coordinates")[1] - self.lat_min) / self.box_height * self.rows
        self.pixel_B = (self.coord_B.getInfo().get("coordinates")[0] - self.lon_min) / self.box_width * self.cols, (self.coord_B.getInfo().get("coordinates")[1] - self.lat_min) / self.box_height * self.rows

        return self.image_path
    
    def get_image_from_path(self, path, view=False):

        '''
        Return the image from the path as an np.array
        '''

        if path.endswith('.tif'):
            image = rasterio.open(path)
            image = image.read()
            image = np.moveaxis(image, 0, -1)
            image = image.squeeze()

        elif path.endswith('.png') or path.endswith('.jpg'):
            image = plt.imread(path)
        else:
            raise ValueError('Image file must be .tif, .png or .jpg')
        
        image = np.array(image)
        
        if view:
            plt.imshow(image)
            plt.show()

        return image

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
            image = rasterio.open(self.image_path)
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
 

            # plot the bounding boxes
            for i in range(len(predictions)):
                x, y, w, h = predictions["xmin"][i], predictions["ymin"][i], predictions["xmax"][i], predictions["ymax"][i]
                rect = plt.Rectangle((x, y), w-x, h-y, fill=False, color="red")
                ax.add_patch(rect)
                # scatter points on the image, scale the scatter points fit the same scale as the image
                ax.scatter(predictions["xmin"][i] + (predictions["xmax"][i] - predictions["xmin"][i]) / 2,
                            predictions["ymin"][i] + (predictions["ymax"][i] - predictions["ymin"][i]) / 2,   
                            color="blue", linewidths=2.5, edgecolors="blue")
                
            plt.show()
                
        # transform the scatter into a density map
        # create a grid of points
        x_size = np.linspace(0, image.shape[1], image.shape[1])
        y_size = np.linspace(0, image.shape[0], image.shape[0])
        X, Y = np.meshgrid(x_size, y_size)
        # create a density map
        Z = np.zeros(X.shape)
        for i in range(len(predictions)):
            x, y = (predictions["xmin"][i] + predictions["xmax"][i]) / 2, (predictions["ymin"][i] + predictions["ymax"][i]) / 2
            Z += np.exp(-((X - x)**2 + (Y - y)**2) / 5000) # the 1000 is a hyperparameter that controls the spread of the density map

        # transform the density map into an array of density values shape (m,n,1)
        Z = Z.reshape(Z.shape[0], Z.shape[1], 1)

        # resize and interpolate the density map to the same number of pixels as the image as requested
        Z = np.array(Image.fromarray(Z.squeeze()).resize((self.rows, self.cols)))


        # create a copy of the image and resize it to the same number of pixels as the density map
        image = np.array(Image.fromarray(image).resize((self.rows, self.cols)))
        self.resized_image = image

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


    def slope_grid(self, plot=False):
        '''
        Generate a grid of slope values between 0 and 1
        '''

        dataset = ee.Image("CGIAR/SRTM90_V4")
        elevation = dataset.select("elevation")
        slope = ee.Terrain.slope(elevation)

        elevation_clip = elevation.clip(self.bounding_box)

        res = 10

        resampled = elevation_clip.resample("bilinear").reproject(crs= elevation_clip.projection(), scale=res)

        pass


    def river_grid(self, plot=False, resolution=10):
        '''
        Identify where the rivers are is in the image and generate a grid of river values between 0 and 1

        '''

        gsw = ee.Image('JRC/GSW1_1/GlobalSurfaceWater')
        occurrence = gsw.select('occurrence')

        water_mask_params = {
            'min': 0,
            'max': 1,
            'palette': ['red', 'blue'],
        }
        # Create a water mask layer, and set the image mask so that non-water areas are transparent.
        water_mask = occurrence.gt(20).selfMask()

        # clip
        water_mask_clip = water_mask.clip(self.bounding_box)

        # Visualize the water mask
        water_mask_vis = water_mask_clip.visualize(**water_mask_params)

        # get the image
        url = water_mask_vis.getDownloadUrl({
            'name': 'download_water_mask',
            'scale': resolution,
            'region': self.bounding_box,
            'format': 'GeoTIFF',
        })

        response = requests.get(url)
        water_mask_path = 'test_imgs/download_water_mask.tif'

        with open(water_mask_path, 'wb') as fd:
            fd.write(response.content)

        image = rasterio.open(water_mask_path)
        image = image.read()
        image = np.moveaxis(image, 0, -1)
        image = image.squeeze()

        # flip image
        image = np.flip(image, axis=0)

        # transform the image into a grid of river values between 0 and 1
        river_grid = np.array(image)

        # normalize the river grid
        river_grid = river_grid / np.max(river_grid)

        # resize the river grid to the same number of pixels as requested
        river_grid = cv2.resize(river_grid, (self.cols, self.rows), interpolation=cv2.INTER_LINEAR)

        if plot:
            fig, ax = plt.subplots()
            ax.imshow(river_grid)
            plt.show()

        # create Layer object
        self.river_layer = Layer(self.rows, self.cols)
        self.river_layer.grid = river_grid

        return self.river_layer

    def combine_layers(self, tree_w, river_w, slope_w):
        '''
        Combine the two layers into a single grid using input weights

        Sum each array
        '''

        self.total_grid = tree_w * self.tree_grid.grid + river_w*self.river_grid.grid


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

