# Load the best path .h5 file and display the path on the map
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio

with h5py.File('density_grids/combined_density.h5', 'r') as f:
    graph = f['combined_density'][:]

# open tif image
image_path = "/Users/philblecher/Desktop/Github/BioAI/test_imgs/download.tif"
image = rasterio.open(image_path)
image = image.read()
image = np.moveaxis(image, 0, -1)
# scale image to 50x50 pixels
image = image.squeeze()
# plt.imshow(image)

with h5py.File('ACO Test/Path Iteration Files/paths_iteration_30.h5', 'r') as f:
    best_path = f['best_path'][:]
#     ant_one = f['ant_path_0'][:]
#     ant_two = f['ant_path_1'][:]
#     print(f.keys())

# best_path = ant_two

# Start and end nodes
start_node = 5619
end_node = 1779 #graph.shape[0] * graph.shape[1] - 1

x_size = image.shape[1] / graph.shape[1]
y_size = image.shape[0] / graph.shape[0]

# Plotting the final graph with the optimal path
plt.figure(figsize=(8 ,8))

# plt.imshow(graph, cmap='Blues')

# for i in range(graph.shape[0]):
#     for j in range(graph.shape[1]):
#         plt.text(j, i, str(graph[i, j]), color='blue', ha='center', va='center', fontsize=10)
for i in range(len(best_path)-1):
    plt.plot([best_path[i] % graph.shape[1] * x_size, best_path[i+1] % graph.shape[1] *x_size], [best_path[i] // graph.shape[1] * y_size, best_path[i+1] // graph.shape[1]* y_size], color='red')
plt.title('Optimal Path')
plt.imshow(image)
# Plot start and end nodes
plt.plot(start_node % graph.shape[1] * x_size, start_node // graph.shape[1] * y_size, 'ro', markersize=10)
plt.plot(end_node % graph.shape[1] * x_size, end_node // graph.shape[1] * y_size, 'bo', markersize=10)
plt.savefig('ACO Test/optimal_path_plot.png', bbox_inches='tight')
# plt.colorbar(label='Weight')
# plt.grid(visible=True)
# plt.text(0, -2, f"Optimal path: {best_path}", fontsize=12)
# plt.text(0, -1.5, f"Optimal path length based on weights: {best_path_length_weight}", fontsize=12)
# plt.text(0, -1, f"Optimal path length based on distance: {ant_colony.best_path_length_distance}", fontsize=12)
plt.show()
