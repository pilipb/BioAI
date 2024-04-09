import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py

class AntColony:
    def __init__(self, num_ants, graph, start_node, end_node, alpha=1, beta=2, rho=0.5, q=100, max_iter=100, stop_percentage=0.5):
        self.num_ants = num_ants
        self.graph = graph
        self.num_nodes = graph.shape[0] * graph.shape[1]
        self.start_node = start_node
        self.end_node = end_node
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.max_iter = max_iter
        self.stop_percentage = stop_percentage
        self.pheromone = np.ones((self.num_nodes, self.num_nodes))
        self.visibility = np.zeros((self.num_nodes, self.num_nodes))
        
        # Calculate the distance between nodes using their coordinates
        coordinates = [(i // graph.shape[1], i % graph.shape[1]) for i in range(self.num_nodes)]
        self.distance = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                self.distance[i][j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        np.fill_diagonal(self.distance, 1)  # Avoid division by zero
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    self.visibility[i][j] = 1 / self.distance[i][j]
        
        np.fill_diagonal(self.visibility, 0)
        self.best_path = []
        self.best_path_length_weight = np.inf
        self.best_path_length_distance = np.inf
        self.ants_paths = []

    def find_path(self):
        for iter in range(self.max_iter):
            print(f"Iteration {iter}")
            all_paths = self.generate_paths()
            self.update_pheromone(all_paths)
            shortest_path = min(all_paths, key=lambda x: x[1][0])  # Based on weight
            if shortest_path[1][0] < self.best_path_length_weight:
                self.best_path_length_weight = shortest_path[1][0]
                self.best_path = shortest_path[0]
                self.best_path_length_distance = shortest_path[1][1]  # Actual distance
            print(f"Iteration {iter}: Best path length based on weight = {self.best_path_length_weight}")
            print(f"Iteration {iter}: Best path length based on distance = {self.best_path_length_distance}")

            # Count the number of ants that have found the best path
            num_best_paths = sum(all(ant_node == best_node for ant_node, best_node in zip(ant_path, self.best_path)) for ant_path, _ in all_paths)
            print(f"Iteration {iter}: Number of ants on best path = {num_best_paths}")

            # Check if specified percentage of all ants have found the optimal path
            if num_best_paths >= self.stop_percentage * self.num_ants:
                break

            # Store the paths of all ants for this iteration
            self.ants_paths.append([path for path, _ in all_paths])

            # Save the best path and all paths every 10 iterations
            if iter % 10 == 0:
                with h5py.File(f'ACO Test/paths_iteration_{iter}.h5', 'w') as f:
                    f.create_dataset('best_path', data=self.best_path)
                    f.create_dataset('best_path_length_weight', data=self.best_path_length_weight)
                    f.create_dataset('best_path_length_distance', data=self.best_path_length_distance)
                    f.create_dataset('ants_paths', data=self.ants_paths)

        return self.best_path, self.best_path_length_weight

    def generate_paths(self):
        all_paths = []
        for ant in range(self.num_ants):
            current_node = self.start_node
            visited_nodes = [current_node]
            path = [current_node]
            path_length_weight = 0
            path_length_distance = 0
            while current_node != self.end_node:
                next_node = self.choose_next_node(current_node, visited_nodes)
                path_length_weight += self.graph[current_node // self.graph.shape[1], current_node % self.graph.shape[1]]
                path_length_distance += self.distance[current_node][next_node]
                visited_nodes.append(next_node)
                path.append(next_node)
                current_node = next_node
                if len(visited_nodes) == self.num_nodes:
                    break  # break if all nodes are visited
            path_length_weight += self.graph[current_node // self.graph.shape[1], current_node % self.graph.shape[1]]
            all_paths.append((path, (path_length_weight, path_length_distance)))
        return all_paths

    def choose_next_node(self, current_node, visited_nodes):
        row, col = current_node // self.graph.shape[1], current_node % self.graph.shape[1]
        neighbors = [(row-1, col-1), (row-1, col), (row-1, col+1),
                     (row, col-1),                 (row, col+1),
                     (row+1, col-1), (row+1, col), (row+1, col+1)]
        valid_neighbors = [(r, c) for r, c in neighbors if 0 <= r < self.graph.shape[0] and 0 <= c < self.graph.shape[1] and (r*self.graph.shape[1]+c) not in visited_nodes]
        
        probabilities = []
        total_probability = 0
        for r, c in valid_neighbors:
            node = r*self.graph.shape[1] + c
            if node not in visited_nodes:
                pheromone = self.pheromone[current_node][node] ** self.alpha
                visibility = self.visibility[current_node][node] ** self.beta
                total_probability += pheromone * visibility
                probabilities.append((node, pheromone * visibility))
        
        probabilities = [(node, prob/total_probability) for node, prob in probabilities]
        
        if len(probabilities) == 0:
            return current_node

        # Select the next node based on the probabilities
        p = np.random.rand()
        cumulative_probability = 0
        for node, probability in probabilities:
            cumulative_probability += probability
            if p <= cumulative_probability:
                return node

        # If for some reason we don't pick any node, return the last valid node
        return probabilities[-1][0]

    def update_pheromone(self, all_paths):
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Pheromone update
        for path, (path_length_weight, _) in all_paths:
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += self.q / path_length_weight
            self.pheromone[path[-1]][path[0]] += self.q / path_length_weight

def animate(frame):
    plt.clf()
    # plt.figure(figsize=(10, 10))
    plt.imshow(graph, cmap='Blues')
    plt.title(f'Iteration {frame+1}')
    plt.colorbar(label='Weight')
    # plt.grid(visible=True)
    # Plot the start and end nodes
    plt.text(start_node % graph.shape[1], start_node // graph.shape[1], '*', color='red', ha='center', va='center', fontsize=20)
    plt.text(end_node % graph.shape[1], end_node // graph.shape[1], '*', color='red', ha='center', va='center', fontsize=20)
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            plt.text(j, i, str(graph[i, j]), color='blue', ha='center', va='center', fontsize=10)
    
    colors = ['red', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'pink']
    iteration_path_weights = []
    for idx_iter, ant_path_iter in enumerate(all_ants_paths[frame]):
        ant_weight = 0
        for idx, ant_path in enumerate(ant_path_iter):
            # Plot the path of each ant as a line
            if idx > 0 and ant_path != ant_path_iter[idx-1]:
                plt.plot([ant_path_iter[idx-1] % graph.shape[1], ant_path % graph.shape[1]], [ant_path_iter[idx-1] // graph.shape[1], ant_path // graph.shape[1]], color=colors[idx_iter % len(colors)])
                plt.savefig(f'ACO Test/GIF Frames/iteration_{frame+1}_ant_{idx_iter+1}_line_{idx+1}.png', bbox_inches='tight')

            row, col = ant_path // graph.shape[1], ant_path % graph.shape[1]
            ant_weight += graph[row, col]

        iteration_path_weights.append(ant_weight)

    # Highlight the best path
    best_path_weight = min(iteration_path_weights)
    best_path_idx = iteration_path_weights.index(best_path_weight)
    best_path = all_ants_paths[frame][best_path_idx]
    
    for i in range(len(best_path)-1):
        plt.plot([best_path[i] % graph.shape[1], best_path[i+1] % graph.shape[1]], [best_path[i] // graph.shape[1], best_path[i+1] // graph.shape[1]], color='red', linewidth=3)
        plt.savefig(f'ACO Test/GIF Frames/iteration_{frame+1}_best_path.png', bbox_inches='tight')

# Define the custom graph (grid)
# graph = np.array([
#     [3, 2, 3, 4, 5, 2, 3],
#     [1, 3, 1, 4, 2, 3, 4],
#     [1, 1, 3, 5, 3, 1, 2],
#     [1, 4, 5, 3, 4, 2, 1],
#     [5, 1, 3, 4, 3, 4, 5],
#     [1, 1, 3, 4, 2, 3, 4],
#     [2, 1, 1, 1, 3, 2, 1]
# ])

# Load a .h5 file for the graph
with h5py.File('density.h5', 'r') as f:
    graph = f['density'][:]

# graph = np.random.randint(1, 6, (30, 30))

# Start and end nodes
start_node = 0
end_node = graph.shape[0] * graph.shape[1] - 1

# Initialize and run the Ant Colony Optimization algorithm
ant_colony = AntColony(num_ants=100, graph=graph, start_node=start_node, end_node=end_node, alpha=1, beta=2, rho=0.5, q=100, max_iter=1000, stop_percentage=0.5)
best_path, best_path_length_weight = ant_colony.find_path()

# Save best path as a .h5 file
with h5py.File(f'ACO Test/best_path.h5', 'w') as f:
    f.create_dataset('best_path', data=best_path)

# Plotting the final graph with the optimal path
plt.figure(figsize=(10, 10))
plt.imshow(graph, cmap='Blues')
for i in range(graph.shape[0]):
    for j in range(graph.shape[1]):
        plt.text(j, i, str(graph[i, j]), color='blue', ha='center', va='center', fontsize=10)
for i in range(len(best_path)-1):
    plt.plot([best_path[i] % graph.shape[1], best_path[i+1] % graph.shape[1]], [best_path[i] // graph.shape[1], best_path[i+1] // graph.shape[1]], color='red')
plt.title('Optimal Path')
plt.colorbar(label='Weight')
plt.grid(visible=True)
plt.text(0, -2, f"Optimal path: {best_path}", fontsize=12)
plt.text(0, -1.5, f"Optimal path length based on weights: {best_path_length_weight}", fontsize=12)
plt.text(0, -1, f"Optimal path length based on distance: {ant_colony.best_path_length_distance}", fontsize=12)
plt.savefig('ACO Test/optimal_path_plot.png', bbox_inches='tight')

print(f"Optimal path: {best_path}")
print(f"Optimal path length based on weights: {best_path_length_weight}")
print(f"Optimal path length based on distance: {ant_colony.best_path_length_distance}")

# Create and save the animation
# all_ants_paths = ant_colony.ants_paths
# ani = FuncAnimation(plt.gcf(), animate, frames=len(all_ants_paths), interval=1000, repeat=False)
# ani.save('ACO Test/ant_colony_optimization.gif', writer='pillow', fps=1)
