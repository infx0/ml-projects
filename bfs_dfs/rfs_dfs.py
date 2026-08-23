"""Demonstrate breadth-first and depth-first traversal on adjacency-list graphs.

Breadth-first search uses a queue to visit nodes level by level, while depth-first
search uses a nested recursive visitor. Both algorithms track visited vertices and
print their traversal order for comparison.
"""

from collections import deque


def bfs(graph: dict, start: str):
    node_queue = deque()

    # set up a dict to track the visited nodes
    visited = {vertex: False for vertex in graph.keys()}
    visited[start] = True
    node_queue.append(start)

    while node_queue:
        # get the next node in the queue
        current_node = node_queue.popleft()
        print(current_node)
        for neighbor in graph[current_node]:
            if not visited[neighbor]:
                # if we haven't already visited the node, add it to the queue
                visited[neighbor] = True
                node_queue.append(neighbor)


def dfs(graph, start):
    # set up a dict to track the visited nodes
    visited = {vertex: False for vertex in graph.keys()}
    # store traversal order
    X = []

    # if we define dfs_visit() within dfs, we only need to pass the node
    def dfs_visit(vertex):
        visited[vertex] = True
        X.append(vertex)
        print(vertex)
        for neighbor in graph[vertex]:
            if not visited[neighbor]:
                dfs_visit(neighbor)

    dfs_visit(start)


# Define the warehouse as a graph (Adjacency List)
warehouse_graph = {
    "A": ["B", "D"],
    "B": ["A", "C", "E"],
    "C": ["B", "F"],
    "D": ["A", "E", "G"],
    "E": ["B", "D", "F", "H"],
    "F": ["C", "E", "I"],
    "G": ["D", "H"],
    "H": ["E", "G", "I"],
    "I": ["F", "H"],
}

print("Graph Traversal with BFS")
bfs(warehouse_graph, "A")

print("Graph Traversal with DFS")
dfs(warehouse_graph, "A")
