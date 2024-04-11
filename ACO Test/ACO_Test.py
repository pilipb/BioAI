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
                with h5py.File(f'ACO Test/Path Iteration Files/paths_iteration_{iter}.h5', 'w') as f:
                    f.create_dataset('best_path', data=self.best_path)
                    f.create_dataset('best_path_length_weight', data=self.best_path_length_weight)
                    f.create_dataset('best_path_length_distance', data=self.best_path_length_distance)

                    for idx, path in enumerate(self.ants_paths[-1]):
                        f.create_dataset(f'ant_path_{idx}', data=np.array(path))

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

if __name__ == '__main__':

    # Load a .h5 file for the graph
    with h5py.File('density_grids/combined_density.h5', 'r') as f:
        graph = f['combined_density'][:]

    # graph = np.random.randint(1, 6, (30, 30))

    # Start and end nodes
    start_node = 1409
    end_node = 439

    # Get the row and column of the start and end nodes
    start_row, start_col = start_node // graph.shape[1], start_node % graph.shape[1]
    end_row, end_col = end_node // graph.shape[1], end_node % graph.shape[1]

    # Initialize and run the Ant Colony Optimization algorithm
    ant_colony = AntColony(num_ants=1000, graph=graph, start_node=start_node, end_node=end_node, alpha=0.5, beta=2, rho=0.5, q=100, max_iter=1000, stop_percentage=0.5)
    best_path, best_path_length_weight = ant_colony.find_path()

    # Save best path as a .h5 file
    with h5py.File(f'ACO Test/best_path.h5', 'w') as f:
        f.create_dataset('best_path', data=best_path)

    # Plotting the final graph with the optimal path
    plt.figure(figsize=(10, 10))
    plt.imshow(graph, cmap='Blues')
    for i in range(len(best_path)-1):
        plt.plot([best_path[i] % graph.shape[1], best_path[i+1] % graph.shape[1]], [best_path[i] // graph.shape[1], best_path[i+1] // graph.shape[1]], color='red')
    plt.title('Optimal Path')

    # Plot the start and end nodes
    plt.plot(start_col, start_row, 'ro')
    plt.plot(end_col, end_row, 'rx')
    plt.savefig('ACO Test/optimal_path_plot.png', bbox_inches='tight')

    print(f"Optimal path: {best_path}")
    print(f"Optimal path length based on weights: {best_path_length_weight}")
    print(f"Optimal path length based on distance: {ant_colony.best_path_length_distance}")
