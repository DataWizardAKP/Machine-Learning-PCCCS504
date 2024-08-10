import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
G = nx.read_edgelist('facebook_combined.txt')

# Basic info about the graph
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Is the graph directed: {G.is_directed()}")
print(f"Graph density: {nx.density(G)}")

# Define start and goal nodes
start_node = '0'
goal_node = '100'

# Ensure start_node and goal_node are in the graph
if start_node not in G or goal_node not in G:
    raise ValueError("Start or goal node not in graph")

# BFS to find the shortest path
def bfs_shortest_path(graph, start, goal):
    try:
        return nx.shortest_path(graph, source=start, target=goal)
    except nx.NetworkXNoPath:
        return "No path found between the nodes."

# Iterative DFS to explore connections
def dfs_paths(graph, start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(set(graph[node]) - visited)
            yield node

# Perform BFS
bfs_path = bfs_shortest_path(G, start_node, goal_node)
print("BFS Shortest Path:", bfs_path)

# Perform DFS
dfs_connections = list(dfs_paths(G, start_node))
print("DFS Connections:", dfs_connections)


pos = nx.spring_layout(G)


plt.figure(figsize=(6,6))
nx.draw(G, pos, with_labels=True, node_size=20, font_size=8)
plt.title("Facebook Social Network Graph")
plt.show()


if bfs_path != "No path found between the nodes.":
    plt.figure(figsize=(6, 6))
    path_edges = list(zip(bfs_path, bfs_path[1:]))
    nx.draw(G, pos, with_labels=True, node_size=20, font_size=8)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)
    plt.title("BFS Shortest Path")
    plt.show()


plt.figure(figsize=(6,6))
nx.draw(G, pos, with_labels=True, node_size=20, font_size=8)
nx.draw_networkx_nodes(G, pos, nodelist=dfs_connections, node_color='b', node_size=50)
plt.title("DFS Exploration from Start Node")
plt.show()
