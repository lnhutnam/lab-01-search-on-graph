from math import cos, dist
from networkx.algorithms.clique import number_of_cliques
import pygame
import graphUI
from node_color import white, yellow, black, red, blue, purple, orange, green, grey

import math
from queue import PriorityQueue
DFS_LIMIT = 40

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
    # Nếu tham số start và goal không tồn tại, tức là out range len(graph)
    if start not in range(0, len(graph)) or goal not in range(0, len(graph)):
        print("Error: Input invaild range.")
        return

    # Tô màu lại toàn bộ bản đồ đồ là màu đen
    for i in range(len(graph)):
        graph[i][3] = black
    graphUI.updateUI()

    # keep track of explored nodes
    explored = set()

    # keep track of all the paths to be checked
    queue = [[start]]
    explored.add(start)

    # Nếu trạng thái start trùng với trạng thái goal, tô màu tím (purple)
    if start == goal:
        print("Algorithm finished - your path is: {}".format(start))
        # Set color yellow for the goal node
        graph[goal][3] = purple
        graphUI.updateUI()

    # Trong khi hàm đợi chưa rỗng
    while queue:
        pygame.event.get()  # Dòng này đặt sau while để tránh việc not responding của frame PyGame
        path = queue.pop(0)
        node = path[-1]
        neighbors = graph[node][1]
        if not queue and not neighbors:
            graph[start][3] = orange
            for i in range(len(path) - 1):
                edges[edge_id(path[i], path[i + 1])][1] = green
            graphUI.updateUI()
            print(path)
            print("There is no way to get the goal.")
            return
        if node == goal:
            graph[start][3] = orange
            for i in range(len(path) - 1):
                edges[edge_id(path[i], path[i + 1])][1] = green
            graph[goal][3] = purple
            graphUI.updateUI()
            print(path)
            return

        graph[node][3] = yellow
        graphUI.updateUI()
        for adjacency in neighbors:
            if adjacency not in explored:
                edges[edge_id(node, adjacency)][1] = white
                graph[adjacency][3] = red
                graphUI.updateUI()
                new_path = list(path)
                new_path.append(adjacency)
                if adjacency == goal:
                    graph[start][3] = orange
                    for i in range(len(new_path) - 1):
                        edges[edge_id(new_path[i], new_path[i + 1])][1] = green
                    graph[goal][3] = purple
                    graphUI.updateUI()
                    print(new_path)
                    return
                queue.append(new_path)
                explored.add(adjacency)

        graph[node][3] = blue
        graphUI.updateUI()
    print("There is no way to reach the goal.")


def find_path_dfs(graph, edges, edge_id, current, goal, visited, limit):
    if current == goal:
        graph[goal][3] = purple
        graphUI.updateUI()
        return [current]

    if limit > DFS_LIMIT:
        graph[goal][3] = purple
        graphUI.updateUI()
        return [current]
    limit = limit + 1
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
            path = find_path_dfs(graph, edges, edge_id,
                                 neighbor, goal, visited, limit)
            if path is not None:
                path.insert(0, current)
                return path
    graph[current][3] = blue
    graphUI.updateUI()


def DFS(graph, edges, edge_id, start, goal):
    """
    DFS search
    """
    # TODO: your code
    print("Implement DFS algorithm.")
    if start not in range(0, len(graph)) or goal not in range(0, len(graph)):
        print("Error: Input invaild range.")
        return

    for i in range(len(graph)):
        graph[i][3] = black
    graphUI.updateUI()

    # keep track of explored nodes
    explored = set()
    explored.add(start)
    if start == goal:
        print("Algorithm finished - your path is: {} -> {}".format(start, goal))
        # Set color yellow for the goal node
        graph[goal][3] = purple
        graphUI.updateUI()

    count = 0
    path = find_path_dfs(graph, edges, edge_id, start, goal, explored, count)
    if not path:
        print("There is no way to reach the goal.")
    print(path)
    graph[start][3] = orange
    for i in range(len(path) - 1):
        edges[edge_id(path[i], path[i + 1])][1] = green
    graph[goal][3] = purple
    graphUI.updateUI()


def get_cost(first_node, second_node):
    # Function calculate distance between two node
    # first_node at (x_1, y_1)
    # second_node at (x_2, y_2)
    # dist = sqrt((x_2 - x_1)^2 + (y_2 - y_1)^2)
    dist = math.sqrt((second_node[0] - first_node[0])
                     ** 2 + (second_node[1] - first_node[1])**2)
    return dist


def UCS(graph, edges, edge_id, start, goal):
    """
    Uniform Cost Search search
    """
    # TODO: your code
    print("Implement Uniform Cost Search algorithm.")
    if start not in range(0, len(graph)) or goal not in range(0, len(graph)):
        print("Error: Input invaild range.")
        return

    if start == goal:
        print("Algorithm finished - your path is: {}".format(goal))
        graph[goal][3] = purple
        graphUI.updateUI()
        return

    print(graph)
    for i in range(len(graph)):
        graph[i][3] = black
    graphUI.updateUI()

    # keep track of explored nodes
    explored = set()
    # containter is priority queue
    container = PriorityQueue()
    predecessor_set = set()
    # predecessor, cost, current node
    container.put((0, -1, start))

    while container:
        pygame.event.get()
        if container.empty():
            graph[start][3] = orange
            graph[goal][3] = purple
            graphUI.updateUI()
            break

        cost, predecessor, current_node = container.get()
        if current_node not in explored:
            graph[current_node][3] = yellow
            graphUI.updateUI()
            predecessor_set.add((predecessor, current_node))
            explored.add(current_node)
            if current_node == goal:
                path = [current_node]
                print(predecessor_set)
                # Trace path
                while True:
                    for item in predecessor_set:
                        if item[1] == current_node:
                            predecessor = item[0]
                    path.append(predecessor)
                    current_node = predecessor
                    if current_node == start:
                        break
                print("Algorithm finshed, your path is:")
                print(path)
                graph[start][3] = orange
                graph[goal][3] = purple
                for i in range(len(path) - 1):
                    edges[edge_id(path[i], path[i + 1])][1] = green
                graphUI.updateUI()
                return   # End algorithm
            for neighbor in graph[current_node][1]:
                check = neighbor in container.queue
                total_cost = cost + \
                    get_cost(graph[current_node][0], graph[neighbor][0])
                if neighbor not in explored or not check:
                    graph[neighbor][3] = red
                    edges[edge_id(current_node, neighbor)][1] = white
                    graphUI.updateUI()
                    container.put((total_cost, current_node,  neighbor))
                elif check:
                    container.get()
                    container.put((total_cost, current_node, neighbor))
            graph[current_node][3] = blue
            graphUI.updateUI()

    print("There is no way to reach the goal.")


# Heuristic function
# Euclidean Heuristic Distance:
def euclidean_distance(current_x, current_y, goal_x, goal_y):
    """
    Euclidean distance
    current_node(current_x, current_y)
    goal_node(goal_x, goal_y)
    """
    distance = math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
    return distance

# Manhattan Heuristic Distance:


def manhattan_distance(current_x, current_y, goal_x, goal_y):
    """
    Manhattan distance
    current_node(current_x, current_y)
    goal_node(goal_x, goal_y)
    """
    return abs(current_x - goal_x) + abs(current_y - goal_y)

# Diagonal Heuristic Distance:


def diagonal_distance(current_x, current_y, goal_x, goal_y):
    """
    Diagonal distance
    current_node(current_x, current_y)
    goal_node(goal_x, goal_y)
    """
    return max(abs(current_x - goal_x),  abs(current_y - goal_y))


def AStar(graph, edges, edge_id, start, goal):
    """
    A star search
    """
    # TODO: your code
    print("Implement A* algorithm.")

    def pop_frontier_cost(frontier):
        if len(frontier) == 0:
            return None
        frontier.sort()
        out = frontier[0]
        frontier.remove(out)
        return out

    def in_frontier_Cost(frontier, node):
        for i in range(len(frontier)):
            cost, temp, path = frontier[i]
            if node == str(temp):
                return i, True, temp, cost
        return None, False, None, None

    if (start or goal) not in range(len(graph) + 1):
        print(len(graph))
        print(start)
        print("Error: Input invaild range.")
        return

    if start == goal:
        print("Algorithm finished - your path is: {} -> {}".format(start, goal))
        graph[start][3] = orange
        graph[goal][3] = purple
        edges[edge_id(start, goal)][1] = green
        return

    for i in range(len(graph)):
        graph[i][3] = black
    graphUI.updateUI()
    # Create a open set
    open_set = []
    # Create a explored set
    explored_set = []

    # Create a path
    path = []
    # path_cost = euclidean_distance(graph[start][0][0], graph[start][0][1], graph[goal][0][0], graph[goal][0][1])
    path_cost = manhattan_distance(
        graph[start][0][0], graph[start][0][1], graph[goal][0][0], graph[goal][0][1])
    # path_cost = diagonal_distance(graph[start][0][0], graph[start][0][1], graph[goal][0][0], graph[goal][0][1])
    path.append(start)
    open_set = [(path_cost, start, path)]

    while len(open_set) > 0:
        pygame.event.get()
        current_cost, current_node, current_path = pop_frontier_cost(open_set)
        graph[current_node][3] = yellow
        graphUI.updateUI()
        explored_set.append(current_node)
        # Vì trên map ban đầu không có thông tin heuristic sẵn nên ta phải tính như thế, nó chỉ tương đối
        # Nếu ta có một ma trận heuristic sẵn thì hàm g ta sẽ tính chính xác hơn
        # g function = current cost - ước lượng bằng manhattan distance từ current node đến goal
        # current_cost = current_cost - euclidean_distance(graph[current_node][0][0], graph[current_node][0][1], graph[start][0][0], graph[start][0][1])
        current_cost = current_cost - manhattan_distance(
            graph[current_node][0][0], graph[current_node][0][1], graph[goal][0][0], graph[goal][0][1])
        # current_cost = current_cost - diagonal_distance(graph[current_node][0][0], graph[current_node][0][1], graph[start][0][0], graph[start][0][1])
        if current_node == goal:
            print(current_path)
            print(explored_set)
            graph[start][3] = orange
            graph[goal][3] = purple
            for i in range(len(current_path) - 1):
                edges[edge_id(current_path[i], current_path[i + 1])][1] = green
            graphUI.updateUI()
            return
        neighbors = graph[current_node][1]

        for neighbor in neighbors:
            pos, check, old_node, old_cost = in_frontier_Cost(
                open_set, neighbor)
            if neighbor not in explored_set and not check:
                graph[neighbor][3] = red
                edges[edge_id(current_node, neighbor)][1] = white
                graphUI.updateUI()
                new_path = list(current_path)
                new_path.append(neighbor)
                # f = g + h = g + ước lượng manhattan distance từ neighbor node đến goal
                # new_cost = current_cost + euclidean_distance(graph[neighbor][0][0], graph[neighbor][0][1], graph[goal][0][0], graph[goal][0][1]) + 1
                new_cost = current_cost + manhattan_distance(
                    graph[neighbor][0][0], graph[neighbor][0][1], graph[goal][0][0], graph[goal][0][1]) + 1
                # new_cost = current_cost + diagonal_distance(graph[neighbor][0][0], graph[neighbor][0][1], graph[goal][0][0], graph[goal][0][1]) + 1
                open_set.append((new_cost, int(neighbor), new_path))
            elif check:
                # new_cost = current_cost + euclidean_distance(graph[neighbor][0][0], graph[neighbor][0][1], graph[goal][0][0], graph[goal][0][1]) + 1
                new_cost = current_cost + manhattan_distance(
                    graph[neighbor][0][0], graph[neighbor][0][1], graph[goal][0][0], graph[goal][0][1]) + 1
                # new_cost = current_cost + diagonal_distance(graph[neighbor][0][0], graph[neighbor][0][1], graph[goal][0][0], graph[goal][0][1]) + 1
                new_path = list(current_path)
                new_path.append(neighbor)
                if(old_cost > new_cost):
                    open_set.pop(pos)
                    open_set.append((new_cost, int(neighbor), new_path))

        graph[current_node][3] = blue
        graphUI.updateUI()
    print("There is no way to reach the goal.")


def GreedyHeuristic(graph, edges, edge_id, start, goal):
    """
    Greedy Best-First Search
    f(n) = g(n) + h(n)
    with g(n) = 0
    """
    # TODO: your code
    print("Implement Greedy (Heuristic) Best-First Search.")

    def pop_frontier_cost(frontier):
        if len(frontier) == 0:
            return None
        frontier.sort()
        out = frontier[0]
        frontier.remove(out)
        return out

    def in_frontier_Cost(frontier, node):
        for i in range(len(frontier)):
            cost, temp, path = frontier[i]
            if node == str(temp):
                return i, True, temp, cost
        return None, False, None, None

    if (start or goal) not in range(len(graph)):
        print("Error: Input invaild range.")
        return

    if start == goal:
        print("Algorithm finished - your path is: {}".format(goal))
        graph[start][3] = purple
        graphUI.updateUI()
        return

    # Tô lại toàn bô các đỉnh màu đen
    for i in range(len(graph)):
        graph[i][3] = black
    graphUI.updateUI()

    # Create a open set
    open_set = []
    # Create a explored set
    explored_set = []

    # Create a path
    path = []
    # path_cost = euclidean_distance(graph[start][0][0], graph[start][0][1], graph[goal][0][0], graph[goal][0][1])
    path_cost = manhattan_distance(
        graph[start][0][0], graph[start][0][1], graph[goal][0][0], graph[goal][0][1])
    # path_cost = diagonal_distance(graph[start][0][0], graph[start][0][1], graph[goal][0][0], graph[goal][0][1])
    path.append(start)
    open_set = [(path_cost, start, path)]
    while len(open_set) > 0:
        pygame.event.get()
        current_cost, current_node, current_path = pop_frontier_cost(open_set)
        graph[current_node][3] = yellow
        graphUI.updateUI()
        if current_node == goal:
            print("Your path is: ")
            print(current_path)
            print("Explored vertex set: ")
            print(explored_set)
            graph[start][3] = orange
            graph[goal][3] = purple
            for i in range(len(current_path) - 1):
                edges[edge_id(current_path[i], current_path[i + 1])][1] = green
            graphUI.updateUI()
            return
        neighbors = graph[current_node][1]
        for neighbor in neighbors:
            graph[neighbor][3] = red
            edges[edge_id(current_node, neighbor)][1] = white
            graphUI.updateUI()
            pos, check, old_node, old_cost = in_frontier_Cost(
                open_set, neighbor)
            new_path = list(current_path)
            new_path.append(neighbor)
            # f = h
            # new_cost = current_cost + euclidean_distance(graph[neighbor][0][0], graph[neighbor][0][1], graph[goal][0][0], graph[goal][0][1])
            # Hàm f trung với hàm h nên giờ chỉ cần tính ước lượng manhattan từ neighbor đến goal thôi
            new_cost = current_cost + manhattan_distance(
                graph[neighbor][0][0], graph[neighbor][0][1], graph[goal][0][0], graph[goal][0][1])
            # new_cost = current_cost + diagonal_distance(graph[neighbor][0][0], graph[neighbor][0][1], graph[goal][0][0], graph[goal][0][1])
            if neighbor not in explored_set and not check:
                open_set.append((new_cost, neighbor, new_path))
            elif check:
                if(old_cost > new_cost):
                    open_set.pop(pos)
                    open_set.append((new_cost, int(neighbor), new_path))
        graph[current_node][3] = blue
        explored_set.append(current_node)
        graphUI.updateUI()
    print("There is no way to reach the goal.")

"""
def a_star(graph, edges, edge_id, start, goal):
    open_set = set([start])
    closed_set = set()
    g = {}  # store distance from starting node
    parents = {}  # parents contains an adjacency map of all nodes

     # ditance of starting node from itself is zero
    g[start] = 0
      # start_node is root node i.e it has no parent nodes
      # so start_node is set to its own parent node
    parents[start] = start

    while len(open_set) > 0:
            pygame.event.get()
            n = None

            # node with lowest f() is found
            for v in open_set:
                if n == None or g[v] + manhattan_distance(graph[v][0][0], graph[v][0][1], graph[goal][0][0], graph[goal][0][1]) < g[n] +manhattan_distance(graph[n][0][0], graph[n][0][1], graph[goal][0][0], graph[goal][0][1]):
                    n = v

            if n == goal or graph[n][1] == None:
                pass
            else:
                lst = list()
                for neighbor in graph[n][1]:
                    lst.append((neighbor, euclidean_distance(graph[neighbor][0][0], graph[neighbor][0][1], graph[n][0][0], graph[n][0][1])))
                for (m, weight) in lst:
                    # nodes 'm' not in first and last set are added to first
                    # n is set its parent
                    if m not in open_set and m not in closed_set:
                        open_set.add(m)
                        parents[m] = n
                        g[m] = g[n] + weight

                    # for each node m,compare its distance from start i.e g(m) to the
                    # from start through n node
                    else:
                        if g[m] > g[n] + weight:
                            # update g(m)
                            g[m] = g[n] + weight
                            # change parent of m to n
                            parents[m] = n

                            # if m in closed set,remove and add to open
                            if m in closed_set:
                                closed_set.remove(m)
                                open_set.add(m)

            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == goal:
                path = []

                while parents[n] != n:
                    path.append(n)
                    n = parents[n]

                path.append(start)

                path.reverse()

                print('Path found: {}'.format(path))
                return path

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_set.remove(n)
            closed_set.add(n)
    print('Path does not exist!')
    return None
"""

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
