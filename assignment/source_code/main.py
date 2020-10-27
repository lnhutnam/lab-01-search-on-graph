import sys
import graphUI

import graph_data


def read_graph(filename):
    fi = open(filename, 'r')
    start = int(fi.readline())
    goal = int(fi.readline())
    graph = graph_data.get_graph(fi)
    fi.close()
    return start, goal, graph


if __name__ == '__main__':
    """
        Argument from command line. To run: `python main.py input_file search_algorithm`
        search_algorithm must be one of ['bfs', 'dfs', 'ucs', 'a_star']
    """
    print('Number of arguments:', len(sys.argv), 'arguments. Argument List:', str(sys.argv))

    if len(sys.argv) < 3:
        print("To run: `python main.py input_file search_algorithm`")
        print("WARMING: Must be provided enough argument!\n")

    input_file_name = str(sys.argv[1])
    if len(sys.argv) == 3:
        search_alg = str(sys.argv[2])
    else:
        search_alg = None

    start, goal, graph = read_graph(input_file_name)
    graphUI.run(graph, start, goal, search_alg)
