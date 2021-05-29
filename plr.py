from fractions import Fraction
import numpy as np
import xml.etree.cElementTree as ET
from sympy import *
import os
import sys
import copy


def reverse_matrix(fraction_matrix):
    sympy_matrix = []
    for row in fraction_matrix:
        row_buf = []
        for element in row:
            row_buf.append(sympify(element))
        sympy_matrix.append(row_buf)
    sympy_matrix = Matrix(sympy_matrix)
    reversed_sympy_matrix = sympy_matrix.inv()
    reversed_fraction_matrix = []
    position = 0
    for i in range(0, int(np.sqrt(len(reversed_sympy_matrix)))):
        row = []
        for j in range(0, int(np.sqrt(len(reversed_sympy_matrix)))):
            row.append(Fraction(str(reversed_sympy_matrix[position])))
            position += 1
        reversed_fraction_matrix.append(row)
    return np.transpose(reversed_fraction_matrix)


def print_simplex_table(simplex_table, simplexes):
    print("Basis indexes:", end=' ')
    for index in simplex_table[0]:
        print(index, end=' ')
    print()
    print("C basis:", end=' ')
    for index in simplex_table[1]:
        print(index, end=' ')
    print()
    print("Optimal resolution vector:", end=' ')
    for component in simplex_table[2]:
        print(component, end=' ')
    print()
    for i in range(3, len(simplex_table)):
        print("A{}".format(i - 2), ":", end=' ')
        for component in simplex_table[i]:
            print(component, end=' ')
        print()
    print("Simplexes:", end=' ')
    for component in simplexes:
        print(component, end=' ')
    print()


def print_parametric_solution(argument_range, basis_indexes, solution_vector,
                              optimal_resolution, file_name):
    with open(file_name, 'a') as f:
        f.write("Argument range: ({},{})\n".format(argument_range[0], argument_range[1]))
        if basis_indexes is None:
            f.write("No solution\n")
            return
        f.write("Basis variables: ")
        for i in range(0, len(basis_indexes)):
            f.write("X{} ".format(basis_indexes[i]))
        f.write("\n")
        f.write("Solution vector: (")
        for i in range(0, len(solution_vector)):
            f.write(str(solution_vector[i]))
            if i != len(solution_vector) - 1:
                f.write(" ")
        f.write(")\n")
        f.write("Optimal resolution: {}\n".format(optimal_resolution))
        f.write("\n\n")


def parse_simplex_table_xml(file_name):
    simplex_table = []
    simplexes = []
    try:
        if not os.path.exists(file_name):
            print(file_name)
            raise FileNotFoundError("File doesn't exist")
        tree = ET.parse(file_name)
        initial_data = tree.getroot()
        task_type = initial_data.find('TaskType')
        dimensionality = initial_data.find('Dimensionality')
        goal_func_vector = initial_data.find('GoalFunctionVector')
        parametric_vector = initial_data.find('ParametricVector')
        b_vector = initial_data.find('BVector')
        if dimensionality or goal_func_vector or parametric_vector or b_vector or task_type is None:
            raise AttributeError("Wrong file structure")
        goal_func_vector = list(map(Fraction, goal_func_vector.text.split()))
        b_vector = list(map(Fraction, b_vector.text.split()))
        parametric_vector = list(map(Fraction, parametric_vector.text.split()))
        dimensionality = list(map(int, dimensionality.text.split()))
        task_type = task_type.text
        basis_indexes = [i for i in range(dimensionality[1] + 1, dimensionality[0] + dimensionality[1] + 1)]
        basis_goal_function = [Fraction("0/1") for _ in range(len(basis_indexes))]
        simplex_table.append(basis_indexes)
        simplex_table.append(basis_goal_function)
        simplex_table.append(b_vector)
        for i in range(0, dimensionality[1]):
            A_vector = initial_data.find('X{}'.format(i + 1))
            if A_vector is not None:
                simplex_table.append(list(map(Fraction, A_vector.text.split())))
            else:
                raise AttributeError("Wrong file structure")
        identity_matrix = np.eye(len(basis_indexes))
        for i in range(len(basis_indexes)):
            column = map(Fraction, identity_matrix[i])
            column_to_list = list(column)
            simplex_table.append(column_to_list)
        goal_func_vector += basis_goal_function
        parametric_vector += basis_goal_function
        for i in range(3, len(simplex_table)):
            simplexes.append(np.dot(basis_goal_function, simplex_table[i]) - goal_func_vector[i - 3])
    except AttributeError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    return [task_type, goal_func_vector, parametric_vector, simplex_table, simplexes, b_vector]


def simplex_method(*simplex_problem):
    goal_function_vector, simplex_table, simplexes = simplex_problem
    min_ratio = 10000
    min_simplex = 10000
    old_basis_position = -1
    new_basis_index = -1
    non_basis_variable_indexes = [i for i in range(1, len(simplex_table) - 2) if i not in simplex_table[0]]
    for i in range(0, len(simplexes)):
        if i + 1 in non_basis_variable_indexes and simplexes[i] < min_simplex:
            min_simplex = simplexes[i]
            new_basis_index = i + 1
    for i in range(len(simplex_table[2])):
        if simplex_table[2][i] / simplex_table[2 + new_basis_index][i] < min_ratio and \
                simplex_table[2 + new_basis_index][i] >= 0:
            min_ratio = simplex_table[2][i] / simplex_table[2 + new_basis_index][i]
            old_basis_position = i
    if old_basis_position == -1:
        return -1
    norm = simplex_table[2 + new_basis_index][old_basis_position]
    for j in range(2, len(simplex_table)):
        simplex_table[j][old_basis_position] = simplex_table[j][old_basis_position] / norm
    for i in range(0, len(simplex_table[0])):
        above_elem = simplex_table[2 + new_basis_index][i]
        for j in range(2, len(simplex_table)):
            if i != old_basis_position:
                simplex_table[j][i] = simplex_table[j][i] - simplex_table[j][old_basis_position] * above_elem
    simplex_table[1][old_basis_position] = goal_function_vector[new_basis_index - 1]
    simplex_table[0][old_basis_position] = new_basis_index
    simplexes.clear()
    for i in range(3, len(simplex_table)):
        simplexes.append(np.dot(simplex_table[1], simplex_table[i]) - goal_function_vector[i - 3])
    # print_simplex_table(simplex_table, simplexes)
    if min(simplexes) < 0:
        simplex_method(*simplex_problem)


def dual_simplex_method(*simplex_problem):
    goal_function_vector, simplex_table, simplexes = simplex_problem
    old_basis_index = simplex_table[2].index((min(simplex_table[2]))) + 1
    # print(old_basis_index)
    min_ratio = 1000
    new_basis_position = -1
    for i in range(3, len(simplex_table)):
        if simplex_table[i][old_basis_index - 1] < 0 and abs(
                simplexes[i - 3] / simplex_table[i][old_basis_index - 1]) < min_ratio:
            min_ratio = abs(simplexes[i - 3] / simplex_table[i][old_basis_index - 1])
            # print("min ratio = ", min_ratio)
            new_basis_position = i
    if new_basis_position == -1:
        return -1
    # print(new_basis_position)
    norm = simplex_table[new_basis_position][old_basis_index - 1]
    # print("norm = ", norm)
    for j in range(2, len(simplex_table)):
        simplex_table[j][old_basis_index - 1] = simplex_table[j][old_basis_index - 1] / norm
    for i in range(0, len(simplex_table[0])):
        above_elem = simplex_table[new_basis_position][i]
        for j in range(2, len(simplex_table)):
            if i != old_basis_index - 1:
                simplex_table[j][i] = simplex_table[j][i] - simplex_table[j][old_basis_index - 1] * above_elem
    simplex_table[1][old_basis_index - 1] = goal_function_vector[new_basis_position - 3]
    simplex_table[0][old_basis_index - 1] = new_basis_position - 2
    simplexes.clear()
    for i in range(3, len(simplex_table)):
        simplexes.append(np.dot(simplex_table[1], simplex_table[i]) - goal_function_vector[i - 3])
    if min(simplex_table[2]) < 0:
        dual_simplex_method(*simplex_problem)


def b_vector_variation(*simplex_problem, initial_conditions, initial_param_value=0, output):
    # argument_range, basis_indexes, solution_vector,optimal_resolution, file_name
    goal_function_vector, parametric_vector, simplex_table, simplexes, b_vector = simplex_problem
    # dual_simplex_method(goal_function_vector, simplex_table, simplexes)
    # print_simplex_table(simplex_table, simplexes)
    left_border = -10000
    right_border = 10000
    # print("initial param value = ", initial_param_value)
    print_simplex_table(simplex_table, simplexes)
    basis_matrix = []
    for i in range(0, len(simplex_table) - 3):
        if i + 1 in simplex_table[0]:
            basis_matrix.append(initial_conditions[i])
    print("basis matrix = ")
    print(basis_matrix)
    reversed_basis_matrix = reverse_matrix(basis_matrix)
    print("reversed basis matrix = ")
    print(reversed_basis_matrix)
    parametric_matrix = np.dot(reversed_basis_matrix, parametric_vector[0:len(simplex_table[3])])
    # print(parametric_matrix)
    for i in range(0, len(parametric_matrix)):
        if parametric_matrix[i] > 0:
            if -simplex_table[2][i] / parametric_matrix[i] > left_border:
                left_border = -simplex_table[2][i] / parametric_matrix[i]
        elif parametric_matrix[i] < 0:
            if -simplex_table[2][i] / parametric_matrix[i] < right_border:
                right_border = -simplex_table[2][i] / parametric_matrix[i]
    print("left border = ", left_border, "right border = ", right_border)
    argument_range = []
    if left_border == -1000:
        argument_range.append('negative infinity')
    else:
        argument_range.append(left_border + initial_param_value)
    if right_border == 1000:
        argument_range.append('positive infinity')
    else:
        argument_range.append(right_border + initial_param_value)
    # solution vector creation
    solution_vector = [0] * len(goal_function_vector)
    for i in range(0, len(simplex_table[0])):
        solution_vector[simplex_table[0][i] - 1] = simplex_table[2][i]
    solution_vector = solution_vector[0:len(simplex_table[0])]
    print_parametric_solution(argument_range, simplex_table[0], solution_vector,
                              np.dot(simplex_table[1], simplex_table[2]), output)
    if right_border != 1000:
        # print("b vector = ", b_vector)
        # print(parametric_vector[0:len(simplex_table[3])])
        b_vector = b_vector + right_border * np.array(parametric_vector[0:len(simplex_table[3])])
        # print("new b vector = ", b_vector)
        simplex_table[2] = np.dot(reversed_basis_matrix, b_vector).tolist()
        # print_simplex_table(simplex_table, simplexes)
        solution_existence = dual_simplex_method(goal_function_vector, simplex_table, simplexes)
        print_simplex_table(simplex_table, simplexes)
        right_border += initial_param_value
        if solution_existence != -1:
            b_vector_variation(goal_function_vector, parametric_vector, simplex_table, simplexes, b_vector,
                               initial_conditions=initial_conditions, initial_param_value=right_border,
                               output=output)
        '''else:
            print_parametric_solution([initial_param_value, "positive infinity"],)'''


def objective_function_variation(*simplex_problem, initial_param_value=0, output):
    """

    :param output:
    :param initial_param_value:
    :param simplex_problem:
    :return:
    non_basis_variable_indexes -- indexes of variables that are not in basis
    """
    goal_function_vector, parametric_vector, simplex_table, simplexes = simplex_problem
    solution_existence = simplex_method(goal_function_vector, simplex_table, simplexes)
    if solution_existence == -1:
        print_parametric_solution([initial_param_value, "positive infinity"])
        return
    non_basis_variable_indexes = [i for i in range(1, len(simplex_table) - 2) if i not in simplex_table[0]]
    left_border = -1000
    right_border = 1000
    for i in range(0, len(non_basis_variable_indexes)):
        buf_parametric_vector = []
        for j in range(0, len(simplex_table[0])):
            buf_parametric_vector.append(parametric_vector[simplex_table[0][j] - 1])
        pi = np.dot(buf_parametric_vector, simplex_table[non_basis_variable_indexes[i] + 2]) - parametric_vector[
            non_basis_variable_indexes[i] - 1]
        if pi < 0:
            if -simplexes[non_basis_variable_indexes[i] - 1] / pi < right_border:
                right_border = -simplexes[non_basis_variable_indexes[i] - 1] / pi
        elif pi > 0:
            if -simplexes[non_basis_variable_indexes[i] - 1] / pi > left_border:
                left_border = -simplexes[non_basis_variable_indexes[i] - 1] / pi
    goal_function_vector = (np.array(goal_function_vector) + right_border * np.array(parametric_vector)).tolist()
    for i in range(0, len(simplex_table[0])):
        simplex_table[1][i] = goal_function_vector[simplex_table[0][i] - 1]
    simplexes.clear()
    for i in range(3, len(simplex_table)):
        simplexes.append(np.dot(simplex_table[1], simplex_table[i]) - goal_function_vector[i - 3])
    argument_range = []
    if left_border == -1000:
        argument_range.append('negative infinity')
    else:
        argument_range.append(left_border + initial_param_value)
    if right_border == 1000:
        argument_range.append('positive infinity')
    else:
        argument_range.append(right_border + initial_param_value)
    # solution vector creation
    solution_vector = [0] * len(goal_function_vector)
    for i in range(0, len(simplex_table[0])):
        solution_vector[simplex_table[0][i] - 1] = simplex_table[2][i]
    solution_vector = solution_vector[0:len(simplex_table[0])]
    print_parametric_solution(argument_range, simplex_table[0], solution_vector,
                              np.dot(simplex_table[1], simplex_table[2]), output)
    if right_border != 1000:
        right_border += initial_param_value
        objective_function_variation(goal_function_vector, parametric_vector, simplex_table, simplexes,
                                     initial_param_value=right_border, output=output)


def parametric_programming(input_file_name, output_file_name):
    try:
        parse_data = parse_simplex_table_xml(input_file_name)
        task_type = parse_data[0]
        if task_type not in ["c variation", "b variation"]:
            raise ValueError("Wrong task type")
        elif task_type == "c variation":
            # print_simplex_table(parse_data[2], parse_data[3])
            # goal_function_vector, parametric_vector, simplex_table, simplexes
            # return [goal_func_vector, parametric_vector, simplex_table, simplexes, b_vector]
            objective_function_variation(parse_data[1],
                                         parse_data[2],
                                         parse_data[3],
                                         parse_data[4],
                                         initial_param_value=0,
                                         output=output_file_name)
        else:
            initial_cond = copy.deepcopy(parse_data[3][3:len(parse_data[3])])
            initial_b_vector = copy.deepcopy(parse_data[5])
            simplex_method(parse_data[1], parse_data[3], parse_data[4])
            b_vector_variation(parse_data[1],
                               parse_data[2],
                               parse_data[3],
                               parse_data[4],
                               initial_b_vector,
                               initial_conditions=initial_cond,
                               initial_param_value=0,
                               output=output_file_name)
    except ValueError as e:
        print(e)
        return
