# now add color state of each node and edge
# change color states in breadth first search (BFS)
import pygame

from constant_variable import display_width, display_height, fps_speed, delay_start_search_time

# constants
from node_color import grey, black, white
from search_algorithm import BFS, example_func, DFS, AStar, UCS
from utils import draw_graph, quit_event

global_graph = None
screen = None
edges = None
clock: pygame.time
font: pygame.font

'''
We will create a global_graph as this follow structure:
 [0] : xy 
 [1] : adjacent node indexes
 [2] : node edge color 
 [3] : node fill color
 
Ex: global_graph = [
    [
        (139, 140),             # position of node when draw on UI
        [1, 2],                 # list of adjacent node
        (100, 100, 100),        # grey - node edged color
        (0, 0, 0)               # black - node fill color 
    ],
    ...
]
'''


def run(graph, start, goal, search_alg=None):
    global screen, edges, clock, font, global_graph
    global_graph = graph

    # add start colors to graph
    for element in global_graph:
        element.extend([grey, black])

    # print(global_graph)
    build_edges()
    pygame.init()

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((display_width, display_height))
    font = pygame.font.Font(pygame.font.get_default_font(), 25)

    pygame.display.set_caption('Search on Graph - ' + str(search_alg).upper())

    draw_graph(screen, font, global_graph, edges)  # initial
    updateUI()
    pygame.time.delay(delay_start_search_time)  # wait 5 sec to start

    if search_alg == 'bfs':
        BFS(global_graph, edges, edge_id, start, goal)
    elif search_alg == 'dfs':
        DFS(global_graph, edges, edge_id, start, goal)
    elif search_alg == 'ucs':
        UCS(global_graph, edges, edge_id, start, goal)
    elif search_alg == 'a_star':
        AStar(global_graph, edges, edge_id, start, goal)
    # elif search_alg == 'other_algorithm':
    #     pass
    else:
        print("Pass a search algorithm to run program.")
        example_func(global_graph, edges, edge_id, start, goal)

    while True:
        quit_event()


def updateUI():
    global screen, edges, clock, font

    draw_graph(screen, font, global_graph, edges)
    pygame.display.update()
    clock.tick(fps_speed)


# normalize id for either order
def edge_id(n1, n2):
    return tuple(sorted((n1, n2)))


def build_edges():
    global edges
    edges = {}  # edge_id: [(n1,n2), color]
    for n1, (_, adjacency, _, _) in enumerate(global_graph):
        for n2 in adjacency:
            eid = edge_id(n1, n2)
            if eid not in edges:
                edges[eid] = [(n1, n2), grey]
