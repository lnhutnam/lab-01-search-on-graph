import pygame

from constant_variable import node_radius
from node_color import white


def quit_event():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.display.quit()
            pygame.quit()
            quit()


def draw_graph(screen, font, graph, edges):
    screen.fill((0, 0, 0,))

    for e in edges.values():  # draw edges
        (n1, n2), color = e
        pygame.draw.line(screen, color, graph[n1][0], graph[n2][0], 2)

    for idx, (xy, _, lcolor, fcolor) in enumerate(graph):  # draw nodes
        circle_fill(screen, xy, lcolor, fcolor, node_radius, thickness=1)
        circle_label(screen, font, xy, label=str(idx))


def circle_label(screen, font: pygame.font, xy, label):
    node_label = font.render(label, True, white)
    screen.blit(node_label, (xy[0] - 10, xy[1] - 10))


def circle_fill(screen, xy, line_color, fill_color, radius, thickness):
    pygame.draw.circle(screen, line_color, xy, radius)
    pygame.draw.circle(screen, fill_color, xy, radius - thickness)
