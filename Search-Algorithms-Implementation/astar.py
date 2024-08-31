import math
from collections import deque
import os

from main import accessible_locations

if os.path.exists('/Users/kjindal/Downloads/Code/HW1_FOAI/Run grading_HW01_cases/output.txt'):
    os.remove('/Users/kjindal/Downloads/Code/HW1_FOAI/Run grading_HW01_cases/output.txt')

import collections 
from main import *
 
def calculate_eucledian(initial_node, final_node):
    initial_node_x = initial_node[0]
    initial_node_y = initial_node[1]
    final_node_x = final_node[0]
    final_node_y = final_node[1]
    distance = abs(final_node_y - initial_node_y) + abs(final_node_x - initial_node_x)
    
    return abs(distance)


def astar(current_i, current_j, final):

    root = (current_i, current_j)
    explored = set()
    q = collections.deque()
    q.append([calculate_eucledian(root, final), root])
    # explored.add(root)
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
        cost = (abs(path_required[0])) # calculate_eucledian(node, final)

        # print("path_required", path_required)
        # print("node", node)
        # print("cost", cost)

        if node in explored:
            continue

        explored.add(node)
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
                if (difference_in_height) <= (max_height_to_climb):
            # if (neighbour[3] <= 0 or grid[node[0]][node[1]] <= 0) and (neighbour[0], neighbour[1]) != node:
            # difference_in_height = abs(abs(neighbour[3]) - abs(grid[current_i][current_j]))

            # if difference_in_height <= max_height_to_climb:
                    added_to_path = list(path_required)
            # print("neighbour", neighbour)
            # print("neighbour[2]", neighbour[2])

                    g = neighbour[2]
                    mud = (0 if neighbour[3] <= 0 else neighbour[3])
                    # current_height = (0 if grid[node[0]][node[1]] else grid[node[0]][node[1]])
                    height = (abs(abs(current_height) - abs(jump_height)))
                    h = calculate_eucledian(neighbour_, final)

                    added_to_path[0] = cost + g + mud + height + h - calculate_eucledian(node, final) # - h from cur to finish node
                    # added_to_path[1] = cost_o + g + mud + height

                    added_to_path.append((neighbour[0], neighbour[1]))

                    q.append(added_to_path)
                    # explored.add(neighbour_)
                    # neighbour_now = (neighbour[0], neighbour[1])
                    if neighbour[0] == final[0] and neighbour[1] == final[1]:
                        answer.append(added_to_path)

                # elif (neighbour[0], neighbour[1]) != node:
                #     # add to the path
                #     added_to_path = list(path_required)

                #     g = neighbour[2]
                #     mud = (0 if neighbour[3] <= 0 else neighbour[3])
                #     current_height = (0 if grid[node[0]][node[1]] else grid[node[0]][node[1]])
                #     height = (abs(neighbour[3] - current_height) if neighbour[3] <= 0 else 0)
                #     h = calculate_eucledian(node, (neighbour[0], neighbour[1]))

                #     added_to_path[0] = cost + g + mud + height + h
                #     # added_to_path[0] = cost + neighbour[2] + calculate_eucledian(node, (neighbour[0], neighbour[1])) + neighbour[3] + abs()
                #     added_to_path.append((neighbour[0], neighbour[1]))

                #     stack.append(added_to_path)
                #     neighbour_now = (neighbour[0], neighbour[1])
                #     if tuple(neighbour_now) == tuple(final):
                #         answer.append(added_to_path)

    if len(answer) == 0:
        return "FAIL"
    else:
        answer.sort()
        print("------------ANSWER-----------", answer[0][1:])
        print("------------COST-------------", answer[0][0])
        # print("------------COST w/o HEURISTIC-------------", answer[0][1])
        return answer[0][1:]


def throw_output(starting_loc_i, starting_loc_j, final_sites):
    print("Final Sites: ", final_sites)
    try:
        for destinations in final_sites:
            print("-----------DESTINATION-------------", destinations)
            if selected_algo == "BFS":
                output = bfs(starting_loc_i, starting_loc_j, destinations)
                print("-----------------BFS RETURNED ANSWER---------------", output)
                formatted_output = ''
                if output == "FAIL":
                    formatted_output = 'FAIL'

                for lines in output:
                    if type(lines) is not tuple:
                        continue
                    else:
                        formatted_output = formatted_output + str(lines[1]) + ',' + str(lines[0]) + ' '
            elif selected_algo == "UCS":
                output = ucs(starting_loc_i, starting_loc_j, destinations)
                print("-----------------UCS RETURNED ANSWER---------------", output)
                formatted_output = ''
                if output == "FAIL":
                    formatted_output = 'FAIL'
                for lines in output:
                    if type(lines) is not tuple:
                        continue
                    else:
                        formatted_output = formatted_output + str(lines[1]) + ',' + str(lines[0]) + ' '

            elif selected_algo == "A*":
                output = astar(starting_loc_i, starting_loc_j, destinations)
                print("------------------A* RETURNED ANSWER-----------------")
                formatted_output = ''
                if output == "FAIL":
                    formatted_output = 'FAIL'

                for lines in output:
                    if type(lines) is not tuple:
                        continue
                    else:
                        formatted_output = formatted_output + str(lines[1]) + ',' + str(lines[0]) + ' '
            else:
                formatted_output = 'FAIL'

            file1 = open('/Users/kjindal/Downloads/Code/HW1_FOAI/Run grading_HW01_cases/output.txt', 'a+')

            # writing newline character

            file1.write(formatted_output)
            file1.write("\n")
    except Exception as E:
        print (E)

        formatted_output = 'FAIL'

        file1 = open('output.txt', 'a+')

        # writing newline character

        file1.write(formatted_output)
        file1.write("\n")
    return formatted_output


def get_main():
    return

try:
    
    # FINAL FUNCTION CALLING
    
    throw_output(starting_loc_i, starting_loc_j, final_sites)  
    
except Exception as E:
    formatted_output = 'FAIL'

    file1 = open('/Users/kjindal/Downloads/Code/HW1_FOAI/Run grading_HW01_cases/output.txt', "a+")

    # writing newline character

    file1.write(formatted_output)
    file1.write("\n")

