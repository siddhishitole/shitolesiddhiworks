import math
from collections import deque
import os

from main import accessible_locations

if os.path.exists('/Users/kjindal/Downloads/Code/HW1_FOAI/Run grading_HW01_cases/output.txt'):
    os.remove('/Users/kjindal/Downloads/Code/HW1_FOAI/Run grading_HW01_cases/output.txt')

import collections
from main import *



def ucs(current_i, current_j, final):
    root = (current_i, current_j)
    explored = []
    q = collections.deque()
    q.append([0, root])
    # explored.append(root)
    answer = []

    # explored = []
    # stack = deque()
    # stack.append([0, (current_i, current_j)])

    # answer = []

    while q:
        stack_new = list(q).copy()
        stack_new.sort()
        q = deque(stack_new)
        path_required = q.popleft()
        node = path_required[-1]
        cost = path_required[0]

        if node in explored:
            continue
        else:
            explored.append(node)
        

            if node[0] == final[0] and node[1] == final[1]:
                print("GOT ANSWER ------------ ", path_required)
                answer.append(path_required)

            for neighbour in accessible_locations(node[0], node[1],
                                            rows_in_grid, columns_in_grid,
                                            selected_algo):
                # for neighbour in n:
                neighbour_ = (neighbour[0], neighbour[1])
                if neighbour_ not in explored:
                    current_height = (0 if grid[node[0]][node[1]] >= 0 else grid[node[0]][node[1]])
                    jump_height = (0 if neighbour[3] >= 0 else neighbour[3])
                    difference_in_height = abs(abs(current_height) - abs(jump_height))
                    if int(difference_in_height) <= int(max_height_to_climb):
                        # difference_in_height = abs(abs(neighbour[3]) - abs(grid[current_i][current_j]))
                        # if difference_in_height <= max_height_to_climb:
                        added_to_path = list(path_required)

                        added_to_path[0] = cost + neighbour[2]
                        added_to_path.append(neighbour_)

                        q.append(added_to_path)
                        # explored.add(neighbour_)
                        # if neighbour[0] == final[0] and neighbour[1] == final[1]:
                        #     answer.append(added_to_path)

                    # elif (neighbour[0], neighbour[1]) != node:
                    #     # add to the path
                    #     added_to_path = list(path_required)
                    #     added_to_path[0] = cost + neighbour[2]
                    #     added_to_path.append((neighbour[0], neighbour[1]))

                        # stack.append(added_to_path)
                        # neighbour_now = (neighbour[0], neighbour[1])
                        # if tuple(neighbour_now) == tuple(final):
                        #     answer.append(added_to_path)

    if len(answer) == 0:
        return "FAIL"
    else:
        answer.sort()
        print("------------ANSWER-----------", answer[0][1:])
        print("------------COST-------------", answer[0][0])
        return answer[0][1:]

