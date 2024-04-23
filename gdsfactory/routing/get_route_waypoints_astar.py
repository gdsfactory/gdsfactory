from __future__ import annotations

import uuid
import warnings
from collections.abc import Callable
from functools import partial

import gdstk
import numpy as np
from numpy import bool_, ndarray

import gdsfactory as gf
from gdsfactory.port import Port, select_ports_list

# Node class -- each node is just a coordinate that is legal to search when routing
class Node:
    def __init__(self, coords, visited=False, parent=None):
        self.coords = coords
        self.visited = visited
        self.parent = parent
        self.cost = 0
        self.dist_till_legal_turn = 0

    def __str__(self):
        return "Node:"+str(self.coords[0]) +","+ str(self.coords[1])
    
# priority q for A*
class PriorityQueue(object):
    def __init__(self):
        self.queue_dict = {}
        self.queue = []

    def __str__(self):
        return '|'.join([str(node) for node in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for sorting the Q based on the (heuristic value)
    def sortQueue(self):
        self.queue = sorted(self.queue_dict.items(), key=lambda x:x[1], reverse=True) # x[1] because we want to sort by values
        self.queue = [x[0] for x in self.queue]

    # for inserting an element in the queue
    def insert(self, data, value):
        self.queue_dict[data] = value
        self.sortQueue()

    # for popping an element based on Priority
    def priorityPop(self):
        node = self.queue[-1]
        self.queue_dict.pop(self.queue[-1])
        self.sortQueue()

        return node

# to transform dims (so we can account for decimals up to 0.1 increments)
def dims_to_ints(dims, factor):
    return int(dims[0]*factor), int(dims[1]*factor)

def ints_to_dims(dims, factor):
    return np.array([round(dims[0]*factor, 1), round(dims[1]*factor, 1)])

# helper to check if we are turning
def is_turn(node):

    # check parent and grandparent
    parent_coords = node.parent.coords
    grandparent_coords = node.parent.parent.coords

    # travelling in the x-direction, then turning
    if (parent_coords[0] == grandparent_coords[0]) and grandparent_coords[0] != node.coords[0]:
        return True
    
    # travelling in the y-direction, then turning
    elif (parent_coords[1] == grandparent_coords[1]) and grandparent_coords[1] != node.coords[1]:
        return True

    return False

# to get the waypoints from the path
def waypoints_from_path(path):
    prev_dir = ''
    curr_dir = ''
    waypoints = []
    prev_coords = path[0].coords

    for node in path[1:]:
        if node.coords[0] == prev_coords[0]:
            curr_dir = 'hor'
        if node.coords[1] == prev_coords[1]:
            curr_dir = 'vert'

        if prev_dir != curr_dir:
            waypoints.append(ints_to_dims(prev_coords, 0.1))

        prev_dir = curr_dir
        prev_coords = node.coords
    waypoints.append(ints_to_dims(path[-1].coords, 0.1))
    return waypoints

# initialize a pretend search space - returns a
def init_search_space(dims, restricted_areas):
    """
    dims: tuple(x_dim, y_dim)
    restricted_areas: list(tuples) where each tuple is an illegal (x,y)

    returns: a list of nodes in areas that are legal to search, 0 in areas that are restricted/illegal to search
    """
    search_space = []
    width, height = dims_to_ints(dims, 10)

    for i in range(width): # iterate thru all x vals
        row = []
        for j in range(height): # iterate thru all y vals
            if (i,j) not in restricted_areas: # add Node objects in legal places
                node = Node(coords=(i,j))
                row.append(node)
            else: row.append(0) # add 0 in illegal places
        search_space.append(row)

    return search_space

# initialize a pretend search space
def init_search_space_rectangles(dims, restricted_rectangles):
    """
    similar to init_search_space, but its passed rectangles instead of individual points (so it fills in the rectangles with zeros)
    inputs:
    - dims: tuple(x_dim, y_dim)
    - restricted_rectangles: list of rectangles, each with the following format [top_left, bottom_left, top_right, bottom_right] -- each coordinate is a tuple of (x,y)

    returns:
    - a list of nodes in areas that are legal to search, 0 in areas that are restricted/illegal to search.
    - a list of restricted points, a list(tuples) where each tuple is an illegal (x,y) in the grid.
    """
    search_space = []
    restricted_areas = []
    width, height = dims_to_ints(dims,10)

    if len(restricted_rectangles) > 0:
        bottom_left, top_left, top_right, bottom_right = restricted_rectangles[0]
        top_left, bottom_left, top_right, bottom_right = dims_to_ints(top_left,10), dims_to_ints(bottom_left,10), dims_to_ints(top_right,10), dims_to_ints(bottom_right,10)
        for i in range(width):
            row = []
            for j in range(height):
                if not ((i>=bottom_left[0] and i<=bottom_right[0]) and (j>=bottom_left[1] and j<=top_left[1])):
                    node = Node(coords=(i,j))
                    row.append(node)
                else:
                    row.append(0)
                    restricted_areas.append((i,j))
            search_space.append(row)
        for i in range(1, len(restricted_rectangles)): # iterate thru rest of rectangles
            bottom_left, top_left, top_right, bottom_right = restricted_rectangles[i]
            top_left, bottom_left, top_right, bottom_right = dims_to_ints(top_left,10), dims_to_ints(bottom_left,10), dims_to_ints(top_right,10), dims_to_ints(bottom_right,10)
            for i in range(width):
                for j in range(height):
                    if ((i>=bottom_left[0] and i<=bottom_right[0]) and (j>=bottom_left[1] and j<=top_left[1])):
                        search_space[i][j] = 0
                        restricted_areas.append((i,j))
    else:
        for i in range(width):
            row = []
            for j in range(height):
                node = Node(coords=(i, j))
                row.append(node)
            search_space.append(row)
    return search_space, restricted_areas

# manhattan heuristic
def manhattan_heur(curr_x, curr_y, end_x, end_y):
    return abs(curr_x-end_x)+abs(curr_y-end_y)

# get neighbours for A* - more efficient than DFS, no heuristic (this comes later)
def get_neighbour_coords(curr_node_coords, search_space):
    """
    Finds coordinate of neighbouring nodes that can be searched in A*

    inputs:
    - curr_node_coords: tuple(x, y) of the current node's coordinates
    - search_space: list of list of nodes (in legal coordinates) and zeros
                    (in illegal coordinates), as returned by init_search_space
                    or init_search_space_rectangles

    returns:
    - neighbour_coords: list of tuples of legal neighbouring coordinates
    """

    neighbour_coords = [] # to store the neighbor coordinates
    x,y = curr_node_coords # current node
    x_max, y_max = len(search_space)-1, len(search_space[0])-1 # search boundaries

    if x-1 >= 0: neighbour_coords.append((x-1,y))
    if x+1 <= x_max: neighbour_coords.append((x+1,y))
    if y-1 >= 0: neighbour_coords.append((x,y-1))
    if y+1 <= y_max: neighbour_coords.append((x,y+1))

    return neighbour_coords

# helper to retrace the path
def retrieve_path(start_node, final_node):
    """
    Reconstructs the path once A* reaches the terminal node

    inputs:
    - start_node: Node object of the start node
    - start_node: Node object of the final node

    returns:
    - path: list of Node objects representing nodes along the path
    """

    path = []
    #transformed_path_coordinates = []
    curr_node = final_node

    while curr_node.parent:
        path.append(curr_node)
        #transformed_path_coordinates.append(ints_to_dims(curr_node.coords, 0.1))
        curr_node = curr_node.parent

    path.append(start_node)
    path.reverse()
    #transformed_path_coordinates.reverse()

    return path

# def A_star(input_port, output_port, search_space):
def A_star(start_pos, end_pos, search_space):
    """
    Routes multiple wires, until no more wires can be routed. Prints out the
    visualization once all wires are routed.

    inputs:
    - input_port: Port object representing input port
    - output_port: Port object representing output port
    - start_pos: starting position
    - end_pos: ending position
    - search_space: list of list of nodes (in legal coordinates) and zeros
                    (in illegal coordinates), as returned by init_search_space
                    or init_search_space_rectangles
    - restricted_area: list(tuples) where each tuple is an illegal (x,y)
    - bs: float representing bend size requirement

    returns:
    - path: list(Node) where each node is along the path of the routed wire
    """

    # get the starting node
    start_x, start_y = start_pos[0], start_pos[1]
    # start_x, start_y = dims_to_ints(input_port.center,10)
    curr_node = 0

    # get the final node
    end_x, end_y = end_pos[0], end_pos[1]
    # end_x, end_y = dims_to_ints(output_port.center,10)

    # make a Q
    q = PriorityQueue()
    q.insert(search_space[start_x][start_y], manhattan_heur(start_x, start_y, end_x, end_y))
    # prev_node = None

    #while stack:
    while not q.isEmpty():
        curr_node = q.priorityPop()
        curr_node.visited = True
        x, y = curr_node.coords

        # check if we have reached the final node
        if (x,y) == (end_x, end_y): break

        # add neighbours to stack
        neighbour_coords = get_neighbour_coords((x,y), search_space)
        for x_n, y_n in neighbour_coords:

            # if neighbour exists (ie is not in restricted area)
            if search_space[x_n][y_n]:

                if not search_space[x_n][y_n].visited and search_space[x_n][y_n] not in q.queue:
                    search_space[x_n][y_n].parent = curr_node
                    search_space[x_n][y_n].cost = curr_node.cost + 1
                    
                    # check if its a turn
                    if search_space[x_n][y_n].parent and search_space[x_n][y_n].parent.parent and is_turn(search_space[x_n][y_n]):
                        
                        # if turn is legal (if parent's distance till turn <= 1 node), add to queue
                        if not (search_space[x_n][y_n].parent.dist_till_legal_turn > 1):
                            search_space[x_n][y_n].dist_till_legal_turn = 1000 # dist (in nodes) until next legal turn
                            f = manhattan_heur(x, y, end_x, end_y) + search_space[x_n][y_n].cost
                            q.insert(search_space[x_n][y_n], f)
                            print("turning")
                    
                    # its not a turn
                    else:
                        # update distance until legal turn
                        if search_space[x_n][y_n].parent.dist_till_legal_turn > 1:
                            search_space[x_n][y_n].dist_till_legal_turn = search_space[x_n][y_n].parent.dist_till_legal_turn - 1
                        else:
                            search_space[x_n][y_n].dist_till_legal_turn = 0
                        
                        # add to queue
                        f = manhattan_heur(x, y, end_x, end_y) + search_space[x_n][y_n].cost
                        q.insert(search_space[x_n][y_n], f)


    # retrieve the path
    if curr_node != search_space[end_x][end_y]:
        print("no path found")
        return 0

    path = retrieve_path(search_space[start_x][start_y], search_space[end_x][end_y])

    return path

def generate_route_astar_points(
    input_port: Port,
    output_port: Port,
    bs: float,
    start_straight_length: float = 0.01,
    end_straight_length: float = 0.01,
    min_straight_length: float = 0.01,
    restricted_area: list[ndarray[float]] = [],
) -> ndarray:
    search_space_dims = (150,150)
    search_space, restricted_areas = init_search_space_rectangles(search_space_dims, restricted_area)

    start_pos = np.array(input_port.center)
    end_pos = np.array(output_port.center)
    
    if input_port.orientation == 0.0:
        start_pos[0] += max(bs, start_straight_length, min_straight_length)
    elif input_port.orientation == 180.0:
        start_pos[0] -= max(bs, start_straight_length, min_straight_length)
    
    if output_port.orientation == 0.0:
        end_pos[0] += max(bs, end_straight_length, min_straight_length)
    elif output_port.orientation == 180.0:
        end_pos[0] -= max(bs, end_straight_length, min_straight_length)

    
    A_star_path = A_star(dims_to_ints(start_pos, 10), dims_to_ints(end_pos, 10), search_space)
    waypoints = waypoints_from_path(A_star_path)

    # Check if waypoints[0] -> waypoints[1] is going in the same direction as i_port_pos -> waypoints[0]
    # then replace waypoints[0] with the input position
    # otherwise add the input position to waypoints
    if ((input_port.center[0] == waypoints[0][0] and waypoints[0][0] == waypoints[1][0])
        or
        (input_port.center[1] == waypoints[0][1] and waypoints[0][1] == waypoints[1][1])):
        waypoints[0] = np.array(input_port.center)
    else:
        waypoints.insert(0, np.array(input_port.center))

    # Same logic with the output position
    if ((output_port.center[0] == waypoints[-1][0] and waypoints[-1][0] == waypoints[-2][0])
        or
        (output_port.center[1] == waypoints[-1][1] and waypoints[-1][1] == waypoints[-2][1])):
        waypoints[-1] = np.array(output_port.center)
    else:
        waypoints.append(np.array(output_port.center))
    
    print("waypoints", np.array(waypoints))
    return np.array(waypoints)

"""
# misc todos
- calculate turn distance
"""
turn_dist = 2 # nodes needed straight before turning