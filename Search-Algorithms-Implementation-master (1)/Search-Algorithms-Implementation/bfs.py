import math
from collections import deque
import os
from main import accessible_locations

if os.path.exists('/Users/kjindal/Downloads/Code/HW1_FOAI/Run grading_HW01_cases/output.txt'):
    os.remove('/Users/kjindal/Downloads/Code/HW1_FOAI/Run grading_HW01_cases/output.txt')

import collections

def bfs(current_i, current_j, final):
    root = (current_i, current_j)
    explored = set()
    q = collections.deque()
    q.append([root])
    explored.add(root)
    answer = []

    while q:

        path_required = q.popleft()

        node = path_required[-1]

        if node[0] == final[0] and node[1] == final[1]:
            print("GOT ANSWER ------------ ", path_required)
            answer.append(path_required)
            break

        for neighbour in accessible_locations(node[0], node[1],
                                        rows_in_grid, columns_in_grid,
                                        selected_algo):

            neighbour_ = (neighbour[0], neighbour[1])
            if neighbour_ not in explored:
                current_height = (0 if grid[node[0]][node[1]] >= 0 else grid[node[0]][node[1]]) 
                jump_height = (0 if neighbour[3] >= 0 else neighbour[3])                    
                difference_in_height = abs(abs(current_height) - abs(jump_height))
                if int(difference_in_height) <= int(max_height_to_climb):
                    added_to_path = list(path_required)
                    added_to_path.append(neighbour_)

                    q.append(added_to_path)
                    explored.add(neighbour_)
                    if neighbour[0] == final[0] and neighbour[1] == final[1]:
                        answer.append(added_to_path)
                        return (added_to_path)
        
    
    if len(answer) == 0:
        return "FAIL"

