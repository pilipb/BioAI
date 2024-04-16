# Script to load the data for every tenth iteration and create an animation of the ants moving
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio
import matplotlib.animation as animation
from scipy.interpolate import UnivariateSpline

# Store total ants and total iterations
total_ants = 1000
total_iterations = 750

with h5py.File('density_grids/combined_density.h5', 'r') as f:
    graph = f['combined_density'][:]

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

# x_size = 1
# y_size = 1

best_paths = []
all_ants_paths = []
best_paths_weight = []
best_paths_distance = []
# Loop through every tenth iteration
for i in range(0, total_iterations, 10):

    # Open the file
    with h5py.File(f'ACO Test/Path Iteration Files Anti Wiggle/paths_iteration_{i}.h5', 'r') as f:
        best_paths.append(f['best_path'][:])
        best_paths_weight.append(f['best_path_length_weight'][()])
        best_paths_distance.append(f['best_path_length_distance'][()])

        ant_paths = []
        # Loop through every ant and store the path
        for j in range(total_ants):
            ant_paths.append(f[f'ant_path_{j}'][:])

        all_ants_paths.append(ant_paths)

# For loop for every iterations best weight and distance
for i in range(len(best_paths_weight)):
    print(f'Best path weight for iteration {i*10}: {best_paths_weight[i]}')
    print(f'Best path distance for iteration {i*10}: {best_paths_distance[i]}')

# Plot the best weights and distances for each iteration in a graph
# plt.figure(figsize=(8, 8))
# plt.plot(best_paths_weight, label='Best Path Weight')
# plt.ylabel('Weight')

# # Plot the best distances on the right y-axis
# plt.twinx()
# plt.plot(best_paths_distance, color='red', label='Best Path Distance')

# plt.title('Best Path Weight and Distance for each iteration')
# plt.xlabel('Iteration')
# plt.ylabel('Distance')
# plt.legend()
# plt.show()

# Spline interpolation of the best path
best_path_smooth = best_paths[-1]
x_values = []
y_values = []
for idx, best_path_step in enumerate(best_path_smooth):
    x_values.append(best_path_step % graph.shape[1] * x_size)
    y_values.append(best_path_step // graph.shape[1] * y_size)#

# Create a spline interpolation
spline = UnivariateSpline(x_values, y_values, k=3, s=0)

# Generate x values for plotting the spline
x_smooth = np.linspace(min(x_values), max(x_values), 1000)
y_smooth = spline(x_smooth)

# Plot the original points and the spline interpolation
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='r', label='Original Points')
plt.plot(x_smooth, y_smooth, 'b-', label='Spline Interpolation')
plt.title('Spline Interpolation of the Best Path')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()
plt.grid(True)
plt.show()


print(fjkdslfjkdl)


# Save each ant movement as an image
for iteration in range(len(all_ants_paths)):
    # Only plot every 10th iteration
    if iteration % 10 != 0:
        continue

    ant_plot = plt.figure(figsize=(8, 8))
    plt.imshow(image)
    # plt.imshow(graph, cmap='Blues', alpha=0.5)
    plt.title(f'Iteration {iteration*10}')

    # Plot start and end nodes
    plt.plot(start_node % graph.shape[1] * x_size, start_node // graph.shape[1] * y_size, 'ro', markersize=10)
    plt.plot(end_node % graph.shape[1] * x_size, end_node // graph.shape[1] * y_size, 'bo', markersize=10)
    
    # Plot ant paths
    for ant in range(len(all_ants_paths[iteration])):
        # Only plot every 10th ant
        if ant % 100 != 0:
            continue

        ant_path_iter = all_ants_paths[iteration][ant]

        # Plot ant trail
        for idx, ant_path_step in enumerate(ant_path_iter):
            if idx > 0 and ant_path_step != ant_path_iter[idx-1]:
                plt.plot([ant_path_iter[idx-1] % graph.shape[1] * x_size, ant_path_step % graph.shape[1] * x_size], [ant_path_iter[idx-1] // graph.shape[1] * y_size, ant_path_step // graph.shape[1] * y_size], color='black')
                      
                plt.savefig(f'ACO Test/GIF Frames Anti Wiggle/iteration_{iteration}_ant_{ant}_line_{idx}.png', bbox_inches='tight')

                plt.plot([ant_path_iter[idx-1] % graph.shape[1] * x_size, ant_path_step % graph.shape[1] * x_size], [ant_path_iter[idx-1] // graph.shape[1] *y_size, ant_path_step // graph.shape[1] * y_size], color='green')

    # Plot best path in red
    best_path_iter = best_paths[iteration]
    for idx, best_path_step in enumerate(best_path_iter):
        if idx > 0 and best_path_step != best_path_iter[idx-1]:
            plt.plot([best_path_iter[idx-1] % graph.shape[1] * x_size, best_path_step % graph.shape[1] * x_size], [best_path_iter[idx-1] // graph.shape[1] * y_size, best_path_step // graph.shape[1] * y_size], color='black', linewidth=3)

            plt.savefig(f'ACO Test/GIF Frames Anti Wiggle/iteration_{iteration}_ant{ant+1}_line_{idx}.png', bbox_inches='tight')

            plt.plot([best_path_iter[idx-1] % graph.shape[1] * x_size, best_path_step % graph.shape[1] * x_size], [best_path_iter[idx-1] // graph.shape[1] * y_size, best_path_step // graph.shape[1] * y_size], color='red', linewidth=3)
