import math
from collections import deque
import os

if os.path.exists('/Users/kjindal/Downloads/Code/HW1_FOAI/Run grading_HW01_cases/output.txt'):
    os.remove('/Users/kjindal/Downloads/Code/HW1_FOAI/Run grading_HW01_cases/output.txt')

def read_input(input_file):
    with open(str(input_file), 'r') as file:
        input_line = [line.strip() for line in file]
    return input_line

try:
    input_lines = read_input('/Users/kjindal/Downloads/Code/HW1_FOAI/Run grading_HW01_cases/input22.txt')

    selected_algo = str(input_lines[0])

    columns_in_grid = int(input_lines[1].split(" ")[0])

    rows_in_grid = int(input_lines[1].split(" ")[1])

    starting_loc_j = int(input_lines[2].split(" ")[0])

    starting_loc_i = int(input_lines[2].split(" ")[1])

    max_height_to_climb = int(input_lines[3])

    number_of_final_loc = int(input_lines[4])

    final_sites = list()

    for loc in range(5, 5 + int(number_of_final_loc)):
        final_sites.append(input_lines[loc])

    for locations in range(len(final_sites)):
        modified_loc = final_sites[locations].split(" ")
        modified_loc = [int(i) for i in modified_loc if i != '']
        modified_loc[-1], modified_loc[0] = modified_loc[0], modified_loc[-1]
        final_sites[locations] = modified_loc

    grid = input_lines[5+int(number_of_final_loc):]

    for locations in range(len(grid)):
        modified_loc = grid[locations].split(" ")
        modified_loc = [int(i) for i in modified_loc if i != '']
        grid[locations] = modified_loc

except:
    formatted_output = 'FAIL'

def accessible_locations(starting_loc_i,
                         starting_loc_j,
                         rows_in_grid,
                         columns_in_grid,
                         selected_algo):
    accessible_node_list = []
    current_node = [[starting_loc_i, starting_loc_j, 0, grid[starting_loc_i][starting_loc_j]]]

    for i in range(starting_loc_i - 1, starting_loc_i + 2):
        if i != -1 and i < rows_in_grid:
            # current_row = i
            # right and left
            if i != starting_loc_i:
                if selected_algo == "BFS":
                    if [i, starting_loc_j, 1, grid[i][starting_loc_j]] not in accessible_node_list:
                        accessible_node_list.append([i, starting_loc_j, 1, grid[i][starting_loc_j]])

                elif selected_algo == "UCS":
                    if [i, starting_loc_j, 10, grid[i][starting_loc_j]] not in accessible_node_list:
                        accessible_node_list.append([i, starting_loc_j, 10, grid[i][starting_loc_j]])

                elif selected_algo == 'A*':
                    if [i, starting_loc_j, 10, grid[i][starting_loc_j]] not in accessible_node_list:
                        accessible_node_list.append([i, starting_loc_j, 10, grid[i][starting_loc_j]])

                else:
                    return "FAIL"

    for j in range(starting_loc_j - 1, starting_loc_j + 2):
        if j != -1 and j < columns_in_grid:
            current_col = j

            # up and down

            if j != starting_loc_j:
                if selected_algo == "BFS":
                    if [starting_loc_i, j, 1, grid[starting_loc_i][j]] not in accessible_node_list:
                        accessible_node_list.append([starting_loc_i, j, 1, grid[starting_loc_i][j]])

                elif selected_algo == "UCS":
                    if [starting_loc_i, j, 10, grid[starting_loc_i][j]] not in accessible_node_list:
                        accessible_node_list.append([starting_loc_i, j, 10, grid[starting_loc_i][j]])

                elif selected_algo == 'A*':
                    if [starting_loc_i, j, 10, grid[starting_loc_i][j]] not in accessible_node_list:
                        accessible_node_list.append([starting_loc_i, j, 10, grid[starting_loc_i][j]])

                else:
                    return "FAIL: SELECTED ALGO IS INCORRECT"

                # diagonal

                for i in range(starting_loc_i - 1, starting_loc_i + 2):
                    if i != -1 and i < rows_in_grid and i != starting_loc_i and j != starting_loc_j:
                        current_row = i
                        if selected_algo == "BFS":
                            if [i, j, 1, grid[i][j]] not in accessible_node_list:
                                accessible_node_list.append([i, j, 1, grid[i][j]])

                        elif selected_algo == "UCS":
                            if [i, j, 14, grid[i][j]] not in accessible_node_list:
                                accessible_node_list.append([i, j, 14, grid[i][j]])


                        elif selected_algo == 'A*':
                            if [i, j, 14, grid[i][j]] not in accessible_node_list:
                                accessible_node_list.append([i, j, 14, grid[i][j]])

                        else:
                            return "FAIL: SELECTED ALGO IS INCORRECT"

    return (accessible_node_list)

import collections
from ucs import ucs
from astar import astar
from bfs import bfs

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

