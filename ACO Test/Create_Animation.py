# Script to load the data for every tenth iteration and create an animation of the ants moving
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio
import matplotlib.animation as animation

# Store total ants and total iterations
total_ants = 100
total_iterations = 500

with h5py.File('density.h5', 'r') as f:
    graph = f['density'][:]

start_node = 0
end_node = graph.shape[0] * graph.shape[1] - 1

# open tif image
image_path = "/Users/aasav/OneDrive - University of Bristol/Documents/Eng Des/Year 5/BioAI/test_imgs/example.tif"
image = rasterio.open(image_path)
image = image.read()
image = np.moveaxis(image, 0, -1)

# scale image to 50x50 pixels
image = image.squeeze()

# Rescale the graph to the image size
x_size = image.shape[1] / graph.shape[1]
y_size = image.shape[0] / graph.shape[0]

best_paths = []
all_ants_paths = []
# Loop through every tenth iteration
for i in range(0, total_iterations, 10):

    # Open the file
    with h5py.File(f'ACO Test/Path Iteration Files/paths_iteration_{i}.h5', 'r') as f:
        best_paths.append(f['best_path'][:])

        ant_paths = []
        # Loop through every ant and store the path
        for j in range(total_ants):
            ant_paths.append(f[f'ant_path_{j}'][:])

        all_ants_paths.append(ant_paths)

# Save each ant movement as an image
for iteration in range(len(all_ants_paths)):
    # Only plot every 10th iteration
    if iteration % 10 != 0:
        continue

    ant_plot = plt.figure(figsize=(8, 8))
    # plt.imshow(image)
    plt.imshow(graph, cmap='Blues', alpha=0.5)
    plt.title(f'Iteration {iteration*10}')

    # Plot start and end nodes
    plt.plot(start_node % graph.shape[1], start_node // graph.shape[1], 'ro')
    plt.plot(end_node % graph.shape[1], end_node // graph.shape[1], 'rx')
    
    # Plot ant paths
    for ant in range(len(all_ants_paths[iteration])):
        # Only plot every 10th ant
        if ant % 10 != 0:
            continue

        ant_path_iter = all_ants_paths[iteration][ant]

        # Plot ant trail
        for idx, ant_path_step in enumerate(ant_path_iter):
            if idx > 0 and ant_path_step != ant_path_iter[idx-1]:
                plt.plot([ant_path_iter[idx-1] % graph.shape[1], ant_path_step % graph.shape[1]], [ant_path_iter[idx-1] // graph.shape[1], ant_path_step // graph.shape[1]], color='black')
                      
                plt.savefig(f'ACO Test/GIF Frames/iteration_{iteration}_ant_{ant}_line_{idx}.png', bbox_inches='tight')

                plt.plot([ant_path_iter[idx-1] % graph.shape[1], ant_path_step % graph.shape[1]], [ant_path_iter[idx-1] // graph.shape[1], ant_path_step // graph.shape[1]], color='green')

    # Plot best path in red
    best_path_iter = best_paths[iteration]
    for idx, best_path_step in enumerate(best_path_iter):
        if idx > 0 and best_path_step != best_path_iter[idx-1]:
            plt.plot([best_path_iter[idx-1] % graph.shape[1], best_path_step % graph.shape[1]], [best_path_iter[idx-1] // graph.shape[1], best_path_step // graph.shape[1]], color='black', linewidth=3)

            plt.savefig(f'ACO Test/GIF Frames/iteration_{iteration}_ant{ant+1}_line_{idx}.png', bbox_inches='tight')

            plt.plot([best_path_iter[idx-1] % graph.shape[1], best_path_step % graph.shape[1]], [best_path_iter[idx-1] // graph.shape[1], best_path_step // graph.shape[1]], color='red', linewidth=3)
