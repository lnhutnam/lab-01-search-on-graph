import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from constant_variable import display_height, display_width

#
# graph = [
#     # Node 0
#     [(70, 210),  # x,y position
#      (1, 2, 4, 6, 7)  # adjacent nodes
#      ],
#     # Node 1
#     [(70, 350),  # Node 1
#      (0, 2)
#      ],
#     [(140, 420),
#      (0, 1, 5, 9, 10)
#      ],
#     [(210, 70),
#      (11, 8, 6)
#      ],
#     [(210, 210),
#      (0, 6, 7, 11, 12)
#      ],
#     [(210, 490),
#      (2, 10)
#      ],
#     [(280, 140),
#      (0, 3, 4, 11)
#      ],
#     [(280, 280),
#      (0, 4, 9, 12)
#      ],
#     [(350, 70),
#      (11, 3)
#      ],
#     [(350, 350),
#      (2, 7, 10, 12, 13, 15)
#      ],
#     [(350, 490),
#      (2, 5, 9, 13, 14, 15)
#      ],
#     [(420, 140),
#      (3, 4, 6, 8, 12, 16, 17)
#      ],
#     [(420, 280),
#      (4, 7, 9, 11, 17)
#      ],
#     [(420, 420),
#      (9, 10, 15)
#      ],
#     [(490, 490),
#      (10, 18, 15)
#      ],
#     [(560, 420),
#      (9, 10, 13, 14, 17, 18)
#      ],
#     [(630, 70),
#      (17, 11)
#      ],
#     [(630, 210),
#      (11, 12, 15, 16, 18)
#      ],
#     [(700, 420),
#      (17, 14, 15)
#      ],
# ]


# Graph with
#    xy coord in pygame UI
#    adjacent node list (list indexes)

threshold_bound = 50


def get_pos(xy):
    x, y = int((xy[0] + 1) / 2 * display_width), int((xy[1] + 1) / 2 * display_height)
    x = min(display_width - threshold_bound, x)
    x = max(threshold_bound, x)
    y = min(display_height - threshold_bound, y)
    y = max(threshold_bound, y)

    return x, y


def get_graph(fi):
    graph = nx.read_edgelist(fi, nodetype=int)
    # print(graph.edges())
    pos = nx.spring_layout(graph, seed=6)
    # nx.draw_networkx(graph, pos=pos)
    # plt.show()

    structure_graph = []
    for node in sorted(graph.nodes()):
        coord = get_pos(pos[node])
        adj_node = [v for _, v in sorted(graph.edges(node))]
        struct_node = [coord, adj_node]
        structure_graph.append(struct_node)

    return structure_graph
