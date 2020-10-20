from networkx.algorithms.clique import number_of_cliques
import pygame
import graphUI
from node_color import white, yellow, black, red, blue, purple, orange, green

"""
Feel free print graph, edges to console to get more understand input.
Do not change input parameters
Create new function/file if necessary
"""


def BFS(graph, edges, edge_id, start, goal):
    """
    BFS search
    """
    # TODO: your code
    print("Implement BFS algorithm.")
    if start not in range (0, len(graph)) or goal not in range (0, len(graph)):
        return

    for i in range(len(graph)):
        graph[i][3] = black
    graphUI.updateUI()

    # keep track of explored nodes
    explored = set()

    # keep track of all the paths to be checked
    queue = [[start]]
    explored.add(start)

    # Set color orange for start node
    graph[start][3] = orange
    graphUI.updateUI()

    if start == goal:
        print("Algorithm finished - your path is: {} -> {}".format(start, goal))
        # Set color yellow for the goal node
        graph[goal][3] = purple
        graphUI.updateUI()

    while queue:
        path = queue.pop(0)
        node = path[-1]
        graph[node][3] = yellow
        graphUI.updateUI()
        if node == goal:
            graph[start][3] = orange
            for i in range(len(path) - 1):
                edges[edge_id(path[i], path[i + 1])][1] = green
            graphUI.updateUI()
            graph[goal][3] = purple
            graphUI.updateUI()
            print(path)
            return

        for adjacency in graph[node][1]:
            edges[edge_id(node, adjacency)][1] = white
            graph[adjacency][3] = red
            graphUI.updateUI()
            if adjacency not in explored:
                new_path = list(path)
                new_path.append(adjacency)
                queue.append(new_path)
                explored.add(adjacency)
                graph[adjacency][3] = blue
                graphUI.updateUI()
        


def find_path_dfs(graph, edges, edge_id, current, goal, visited):
    if current == goal:
        graph[goal][3] = purple
        graphUI.updateUI()
        return [current]

    graph[current][3] = yellow
    graphUI.updateUI()
    for neighbor in graph[current][1]:
        graph[neighbor][3] = red
        edges[edge_id(current, neighbor)][1] = white
        graphUI.updateUI()
        if neighbor not in visited:
            visited.add(neighbor)
            graph[neighbor][3] = blue
            graphUI.updateUI()
            path = find_path_dfs(graph, edges, edge_id, neighbor, goal, visited)

            if path is not None:
                path.insert(0, current)
                return path

def DFS(graph, edges, edge_id, start, goal):
    """
    DFS search
    """
    # TODO: your code
    print("Implement DFS algorithm.")
    if start not in range (0, len(graph)) or goal not in range (0, len(graph)):
        return

    for i in range(len(graph)):
        graph[i][3] = black
    graphUI.updateUI()

    # keep track of explored nodes
    explored = set()
    explored.add(start)
    # Set color orange for start node
    graph[start][3] = orange
    graphUI.updateUI()

    if start == goal:
        print("Algorithm finished - your path is: {} -> {}".format(start, goal))
        # Set color yellow for the goal node
        graph[goal][3] = purple
        graphUI.updateUI()

    path = find_path_dfs(graph, edges, edge_id, start, goal, explored)    
    graph[start][3] = orange
    for i in range(len(path) - 1):
        edges[edge_id(path[i], path[i + 1])][1] = green
    graph[goal][3] = purple
    graphUI.updateUI()


def UCS(graph, edges, edge_id, start, goal):
    """
    Uniform Cost Search search
    """
    # TODO: your code
    print("Implement Uniform Cost Search algorithm.")
    if start not in range (0, len(graph)) or goal not in range (0, len(graph)):
        return
    
    if start == goal:
        print("Algorithm finished - your path is: {} -> {}".format(start, goal))
        graph[start][3] = orange
        graph[goal][3] = purple
        edges[edge_id(start, goal)][1] = green
        return
    
    
    pass


def AStar(graph, edges, edge_id, start, goal):
    """
    A star search
    """
    # TODO: your code
    print("Implement A* algorithm.")
    pass


def example_func(graph, edges, edge_id, start, goal):
    """
    This function is just show some basic feature that you can use your project.
    @param graph: list - contain information of graph (same value as global_graph)
                    list of object:
                     [0] : (x,y) coordinate in UI
                     [1] : adjacent node indexes
                     [2] : node edge color
                     [3] : node fill color
                Ex: graph = [
                                [
                                    (139, 140),             # position of node when draw on UI
                                    [1, 2],                 # list of adjacent node
                                    (100, 100, 100),        # grey - node edged color
                                    (0, 0, 0)               # black - node fill color
                                ],
                                [(312, 224), [0, 4, 2, 3], (100, 100, 100), (0, 0, 0)],
                                ...
                            ]
                It means this graph has Node 0 links to Node 1 and Node 2.
                Node 1 links to Node 0,2,3 and 4.
    @param edges: dict - dictionary of edge_id: [(n1,n2), color]. Ex: edges[edge_id(0,1)] = [(0,1), (0,0,0)] : set color
                    of edge from Node 0 to Node 1 is black.
    @param edge_id: id of each edge between two nodes. Ex: edge_id(0, 1) : id edge of two Node 0 and Node 1
    @param start: int - start vertices/node
    @param goal: int - vertices/node to search
    @return:
    """

    # Ex1: Set all edge from Node 1 to Adjacency node of Node 1 is green edges.
    node_1 = graph[1]
    for adjacency_node in node_1[1]:
        edges[edge_id(1, adjacency_node)][1] = green
    graphUI.updateUI()

    # Ex2: Set color of Node 2 is Red
    graph[2][3] = red
    graphUI.updateUI()

    # Ex3: Set all edge between node in a array.
    path = [4, 7, 9]  # -> set edge from 4-7, 7-9 is blue
    for i in range(len(path) - 1):
        edges[edge_id(path[i], path[i + 1])][1] = blue
    graphUI.updateUI()
