# File to smooth the best path
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import h5py
import rasterio


# Load the best path .h5 file and display the path on the map
with h5py.File('density_grids/combined_density.h5', 'r') as f:
    graph = f['combined_density'][:]

with h5py.File('ACO Test/Path Iteration Files Anti Wiggle/paths_iteration_700.h5', 'r') as f:
    best_path = f['best_path'][:]

start_node = 5464
end_node = 1761

# open tif image
image_path = "/Users/aasav/OneDrive - University of Bristol/Documents/Eng Des/Year 5/BioAI/test_imgs/download.tif"
image = rasterio.open(image_path)
image = image.read()
image = np.moveaxis(image, 0, -1)

# scale image to 50x50 pixels
image = image.squeeze()

# Rescale the graph to the image size
x_size = image.shape[1] / graph.shape[1]
y_size = image.shape[0] / graph.shape[0]

# Get the x and y values of the best path
x_values = [node % graph.shape[1] * x_size for node in best_path]
y_values = [node // graph.shape[1] * y_size for node in best_path]

# Remove any points that don't increase in the x direction
x_new = [x_values[0]]
y_new = [y_values[0]]

for i in range(1, len(x_values)):
    if x_values[i] > x_new[-1]:
        x_new.append(x_values[i])
        y_new.append(y_values[i])

# Create a spline for the x and y values
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.plot(start_node % graph.shape[1] * x_size, start_node // graph.shape[1] * y_size, 'ro', markersize=10)
plt.plot(end_node % graph.shape[1] * x_size, end_node // graph.shape[1] * y_size, 'bo', markersize=10)
plt.plot(x_values, y_values, 'r', linewidth=3)

plt.savefig('ACO Test/optimal_path_plot_1.png', bbox_inches='tight')
# plt.show()

# Plot the best path
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.plot(start_node % graph.shape[1] * x_size, start_node // graph.shape[1] * y_size, 'ro', markersize=10)
plt.plot(end_node % graph.shape[1] * x_size, end_node // graph.shape[1] * y_size, 'bo', markersize=10)
plt.plot(x_new, y_new, 'r', linewidth=3)

plt.savefig('ACO Test/optimal_path_plot_2.png', bbox_inches='tight')
# plt.show()

# Create a spline for the x and y values using interp1d
x_smooth_1 = np.linspace(min(x_new), max(x_new), 20)
y_smooth_1 = interp1d(x_new, y_new, kind='cubic')(x_smooth_1)

# Plot the smoothed path
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.plot(start_node % graph.shape[1] * x_size, start_node // graph.shape[1] * y_size, 'ro', markersize=10)
plt.plot(end_node % graph.shape[1] * x_size, end_node // graph.shape[1] * y_size, 'bo', markersize=10)
plt.plot(x_smooth_1, y_smooth_1, 'r', linewidth=3)

plt.savefig('ACO Test/optimal_path_plot_3.png', bbox_inches='tight')

# Spline the smoothed path using make_interp_spline
x_smooth = np.linspace(min(x_smooth_1), max(x_smooth_1), 100)
y_smooth = make_interp_spline(x_smooth_1, y_smooth_1)(x_smooth)
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.plot(start_node % graph.shape[1] * x_size, start_node // graph.shape[1] * y_size, 'ro', markersize=10)
plt.plot(end_node % graph.shape[1] * x_size, end_node // graph.shape[1] * y_size, 'bo', markersize=10)
plt.plot(x_smooth, y_smooth, 'r', linewidth=3)

plt.savefig('ACO Test/optimal_path_plot_4.png', bbox_inches='tight')