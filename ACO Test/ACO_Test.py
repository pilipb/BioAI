import numpy as np
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, num_ants, graph, start_node, end_node, alpha=1, beta=2, rho=0.5, q=100, max_iter=100):
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
        self.pheromone = np.ones((self.num_nodes, self.num_nodes))
        self.visibility = np.zeros((self.num_nodes, self.num_nodes))
        
        # Calculate the distance between nodes using their coordinates
        coordinates = [(i // 5, i % 5) for i in range(self.num_nodes)]
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
        self.best_path_length = np.inf

    def find_path(self):
        for iter in range(self.max_iter):
            print(f"Iteration {iter}")
            all_paths = self.generate_paths()
            self.update_pheromone(all_paths)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < self.best_path_length:
                self.best_path_length = shortest_path[1]
                self.best_path = shortest_path[0]
            print(f"Iteration {iter}: Best path length = {self.best_path_length}")

            # Check if all ants have reached the end node
            if all(ant_path[-1] == self.end_node for ant_path, _ in all_paths):
                break

        return self.best_path, self.best_path_length

    def generate_paths(self):
        all_paths = []
        for ant in range(self.num_ants):
            current_node = self.start_node
            visited_nodes = [current_node]
            path = [current_node]
            path_length = 0
            while current_node != self.end_node:
                next_node = self.choose_next_node(current_node, visited_nodes)
                path_length += self.graph[current_node // 5, current_node % 5]
                visited_nodes.append(next_node)
                path.append(next_node)
                current_node = next_node
                if len(visited_nodes) == self.num_nodes:
                    break  # break if all nodes are visited
            path_length += self.graph[current_node // 5, current_node % 5]
            all_paths.append((path, path_length))
        return all_paths

    def choose_next_node(self, current_node, visited_nodes):
        row, col = current_node // 5, current_node % 5
        neighbors = [(row-1, col-1), (row-1, col), (row-1, col+1),
                     (row, col-1),                 (row, col+1),
                     (row+1, col-1), (row+1, col), (row+1, col+1)]
        valid_neighbors = [(r, c) for r, c in neighbors if 0 <= r < 5 and 0 <= c < 5 and (r*5+c) not in visited_nodes]
        
        probabilities = []
        total_probability = 0
        for r, c in valid_neighbors:
            node = r*5 + c
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
        for path, path_length in all_paths:
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += self.q / path_length
            self.pheromone[path[-1]][path[0]] += self.q / path_length


# Define the custom graph (grid)
graph = np.array([
    [3, 2, 3, 4, 5],
    [1, 3, 1, 4, 2],
    [1, 1, 3, 5, 3],
    [4, 4, 1, 3, 4],
    [5, 2, 3, 1, 3]
])

# Start and end nodes
start_node = 0
end_node = 24

# Initialize and run the Ant Colony Optimization algorithm
ant_colony = AntColony(num_ants=100, graph=graph, start_node=start_node, end_node=end_node)
best_path, best_path_length = ant_colony.find_path()

# Plotting the graph
plt.figure(figsize=(8, 8))
plt.imshow(graph, cmap='Blues')
for i in range(5):
    for j in range(5):
        plt.text(j, i, str(graph[i, j]), color='blue', ha='center', va='center', fontsize=12)
for i in range(len(best_path)-1):
    plt.plot([best_path[i] % 5, best_path[i+1] % 5], [best_path[i] // 5, best_path[i+1] // 5], color='red')
plt.title('Optimal Path')
plt.colorbar(label='Weight')
plt.grid(visible=True)
plt.show()
