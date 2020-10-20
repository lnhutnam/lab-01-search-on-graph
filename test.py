graph = {
    '0': ['1', '2'],
    '1': ['0', '2', '3', '4', '10'],
    '2': ['0', '1', '3', '4', '7', '9'],
    '3': ['1', '2', '4', '5'],
    '4': ['1', '2', '3', '5', '6', '7', '8'],
    '5': ['3', '4', '6', '8'],
    '6': ['4', '5'],
    '7': ['2', '4', '9', '10'],
    '8': ['4', '5'],
    '9': ['2', '7'],
    '10': ['1', '7']
}


def bfs(graph, start, end):
    visited = set()
    # maintain a queue of paths
    queue = []
    # push the first path into the queue
    queue.append([start])
    visited.add(start)
    while queue:
        # get the first path from the queue
        path = queue.pop(0)
        # print(path)
        # get the last node from the path
        node = path[-1]
        # print(node)
        # path found
        if node == end:
            return path
        # enumerate all adjacent nodes, construct a new path and push it into the queue
        for adjacent in graph.get(node, []):
            if adjacent not in visited:
                new_path = list(path)
                new_path.append(adjacent)
                queue.append(new_path)
                visited.add(adjacent)
                print(visited)


print(bfs(graph, '0', '6'))

from queue import PriorityQueue

q = PriorityQueue()

q.put((4, 'Read'))
q.put((2, 'Play'))
q.put((5, 'Write'))
q.put((1, 'Code'))
q.put((3, 'Study'))

while not q.empty():
    next_item = q.get()
    print(next_item)
