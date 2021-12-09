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


def print_simplex_table(simplex_table, simplexes, logger_file='log_simplex_table.txt'):
    with open(logger_file, 'a') as f:
        f.write('Basis indexes: ')
        for index in simplex_table[0]:
            f.write(str(index) + ' ')
        f.write('\n')
        f.write('C basis: ')
        for index in simplex_table[1]:
            f.write(str(index) + ' ')
        f.write('\n')
        f.write('Optimal resolution vector: ')
        for component in simplex_table[2]:
            f.write(str(component) + ' ')
        f.write('\n')
        for i in range(3, len(simplex_table)):
            f.write('A{}'.format(i - 2) + ': ')
            for component in simplex_table[i]:
                f.write(str(component) + ' ')
            f.write('\n')
        f.write('Simplexes: ', )
        for component in simplexes:
            f.write(str(component) + ' ')
        f.write('\n         \n')


def print_parametric_solution(argument_range, basis_indexes=None, solution_vector=None,
                              optimal_resolution=None, file_name=None):
    with open(file_name, 'a') as f:
        f.write('Argument range: ({},{})\n'.format(argument_range[0], argument_range[1]))
        if basis_indexes is None:
            f.write('No solution\n\n')
            return
        f.write('Basis variables: ')
        for i in range(0, len(basis_indexes)):
            f.write('X{} '.format(basis_indexes[i]))
        f.write('\n')
        f.write('Solution vector: (')
        for i in range(0, len(solution_vector)):
            f.write(str(solution_vector[i]))
            if i != len(solution_vector) - 1:
                f.write(' ')
        f.write(')\n')
        f.write('Optimal resolution: ' + str(optimal_resolution[0]) + ' ')
        if float(optimal_resolution[1]) > 0:
            f.write('+ ')
        f.write(str(optimal_resolution[1]))
        f.write('*t\n\n')


def parse_fractional_problem(file_name):
    try:
        if not os.path.exists(file_name):
            print(file_name)
            raise FileNotFoundError("File doesn't exist")
        tree = ET.parse(file_name)
        initial_data = tree.getroot()
        task_type = initial_data.find('TaskType')
        dimensionality = initial_data.find('Dimensionality')
        numerator_vector = initial_data.find('NumeratorVector')
        denominator_vector = initial_data.find('DenominatorVector')
        b_vector = initial_data.find('BVector')
        if dimensionality or numerator_vector or denominator_vector or b_vector or task_type is None:
            raise AttributeError('Wrong file structure')
        numerator_vector = list(map(Fraction, numerator_vector.text.split()))
        denominator_vector = list(map(Fraction, denominator_vector.text.split()))
        b_vector = list(map(Fraction, b_vector.text.split()))
        dimensionality = list(map(int, dimensionality.text.split()))
        dimensionality = {'number of constraints': dimensionality[0], 'number of variables': dimensionality[1]}
        task_type = task_type.text
        condition_vectors = []
        for i in range(0, dimensionality['number of variables']):
            condition_vector = initial_data.find('X{}'.format(i + 1))
            if condition_vector is not None:
                condition_vectors.append(list(map(Fraction, condition_vector.text.split())))
            else:
                raise AttributeError('Wrong file structure')
    except AttributeError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    return {'task_type': task_type, 'numerator_vector': numerator_vector,
            'denominator_vector': denominator_vector, 'condition_vectors': condition_vectors,
            'b_vector': b_vector, 'dimensionality': dimensionality}


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
            raise AttributeError('Wrong file structure')
        goal_func_vector = list(map(Fraction, goal_func_vector.text.split()))
        b_vector = list(map(Fraction, b_vector.text.split()))
        parametric_vector = list(map(Fraction, parametric_vector.text.split()))
        dimensionality = list(map(int, dimensionality.text.split()))
        dimensionality = {'number of constraints': dimensionality[0], 'number of variables': dimensionality[1]}
        task_type = task_type.text
        basis_indexes = [i for i in range(dimensionality['number of variables'] + 1,
                                          dimensionality['number of constraints'] + dimensionality[
                                              'number of variables'] + 1)]
        basis_goal_function = [Fraction("0/1") for _ in range(len(basis_indexes))]
        simplex_table.append(basis_indexes)
        simplex_table.append(basis_goal_function)
        simplex_table.append(b_vector)
        for i in range(0, dimensionality['number of variables']):
            A_vector = initial_data.find('X{}'.format(i + 1))
            if A_vector is not None:
                simplex_table.append(list(map(Fraction, A_vector.text.split())))
            else:
                raise AttributeError('Wrong file structure')
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
    return {'task_type': task_type, 'goal_function_vector': goal_func_vector,
            'parametric_vector': parametric_vector, 'simplex_table': simplex_table,
            'simplexes': simplexes, 'b_vector': b_vector, 'dimensionality': dimensionality}


def simplex_method(*simplex_problem, output='log_simplex_table.txt', include_logging=False):
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
    if include_logging:
        print_simplex_table(simplex_table, simplexes, logger_file=output)
    if min(simplexes) < 0:
        simplex_method(*simplex_problem, output=output, include_logging=include_logging)


def dual_simplex_method(*simplex_problem, output='log_simplex_table.txt', include_logging=False):
    goal_function_vector, simplex_table, simplexes = simplex_problem
    old_basis_index = simplex_table[2].index((min(simplex_table[2]))) + 1
    min_ratio = 1000
    new_basis_position = -1
    for i in range(3, len(simplex_table)):
        if simplex_table[i][old_basis_index - 1] < 0 and abs(
                simplexes[i - 3] / simplex_table[i][old_basis_index - 1]) < min_ratio:
            min_ratio = abs(simplexes[i - 3] / simplex_table[i][old_basis_index - 1])
            new_basis_position = i
    if new_basis_position == -1:
        return -1
    norm = simplex_table[new_basis_position][old_basis_index - 1]
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
    if include_logging:
        print_simplex_table(simplex_table, simplexes, logger_file=output)
    if min(simplex_table[2]) < 0:
        dual_simplex_method(*simplex_problem, output=output, include_logging=include_logging)


def dinkelbach_method(*fractional_simplex_method, output, include_logging=False):
    numerator_vector, denominator_vector, condition_vectors, b_vector, dimensionality = fractional_simplex_method
    print(numerator_vector)
    print(denominator_vector)
    print(condition_vectors)
    print(b_vector)
    print(dimensionality)
    optimal_solution = [Fraction("0/1") for _ in range(dimensionality['number of variables'])]
    optimal_solution.append(Fraction("1/1"))
    while True:
        lam = np.dot(numerator_vector, optimal_solution) / np.dot(denominator_vector, optimal_solution)
        print("lam: ", lam)
        goal_func_vector = numerator_vector - np.dot(lam, denominator_vector)
        goal_func_vector = goal_func_vector[:-1]
        print("goal_func_vector: ", goal_func_vector)
        simplex_table = []
        simplexes = []
        basis_indexes = [i for i in range(dimensionality['number of variables'] + 1,
                                          dimensionality['number of constraints'] + dimensionality[
                                              'number of variables'] + 1)]
        basis_goal_function = [Fraction("0/1") for _ in range(len(basis_indexes))]
        simplex_table.append(basis_indexes)
        simplex_table.append(basis_goal_function)
        simplex_table.append(copy.deepcopy(b_vector))
        for i in range(0, dimensionality['number of variables']):
            simplex_table.append(copy.deepcopy(condition_vectors[i]))
        identity_matrix = np.eye(len(basis_indexes))
        for i in range(len(basis_indexes)):
            column = map(Fraction, identity_matrix[i])
            column_to_list = list(column)
            simplex_table.append(column_to_list)
        goal_func_vector = np.concatenate((goal_func_vector, basis_goal_function))
        # goal_func_vector += basis_goal_function
        # print("basis_goal_function: ", basis_goal_function)
        print("changed: ", goal_func_vector)
        for i in range(3, len(simplex_table)):
            simplexes.append(np.dot(basis_goal_function, simplex_table[i]) - goal_func_vector[i - 3])
        print_simplex_table(simplex_table, simplexes, logger_file='log_simplex_table2.txt')
        simplex_method(goal_func_vector,
                       simplex_table,
                       simplexes,
                       output=output,
                       include_logging=True)
        optimal_solution = [Fraction("0/1") for _ in range(dimensionality['number of variables'])]
        optimal_solution.append(Fraction("1/1"))
        for j in range(0, dimensionality['number of variables']):
            if basis_indexes[j] <= dimensionality['number of variables']:
                optimal_solution[simplex_table[0][j]-1] = simplex_table[2][j]
        print("new optimal: ", optimal_solution)
        print("--------------------------------")
        goal_func_vector = numerator_vector - np.dot(lam, denominator_vector)
        if np.dot(goal_func_vector,optimal_solution)==0:
            break


def b_vector_variation(*simplex_problem, initial_conditions, initial_param_value=0, output, dimensionality,
                       side='center', include_logging=True):
    goal_function_vector, parametric_vector, simplex_table, simplexes, b_vector = simplex_problem
    left_border = -10000
    right_border = 10000
    basis_matrix = []
    optimal_resolution = np.dot(simplex_table[1], simplex_table[2])
    for i in range(0, len(simplex_table) - 3):
        if i + 1 in simplex_table[0]:
            basis_matrix.append(initial_conditions[i])
    reversed_basis_matrix = reverse_matrix(basis_matrix)
    parametric_matrix = np.dot(reversed_basis_matrix, parametric_vector[0:len(simplex_table[3])])
    for i in range(0, len(parametric_matrix)):
        if parametric_matrix[i] > 0:
            if -simplex_table[2][i] / parametric_matrix[i] > left_border:
                left_border = -simplex_table[2][i] / parametric_matrix[i]
        elif parametric_matrix[i] < 0:
            if -simplex_table[2][i] / parametric_matrix[i] < right_border:
                right_border = -simplex_table[2][i] / parametric_matrix[i]
    argument_range = []
    if left_border == -1000:
        argument_range.append('negative infinity')
    else:
        argument_range.append(left_border + initial_param_value)
    if right_border == 1000:
        argument_range.append('positive infinity')
    else:
        argument_range.append(right_border + initial_param_value)
    solution_vector = [0] * len(goal_function_vector)
    for i in range(0, len(simplex_table[0])):
        solution_vector[simplex_table[0][i] - 1] = simplex_table[2][i]
    solution_vector = solution_vector[0:dimensionality['number of variables']]
    coefficient = np.dot(simplex_table[1], reversed_basis_matrix)
    coefficient = np.dot(coefficient, parametric_vector[0:dimensionality['number of constraints']])
    parametric_optimal_resolution = [optimal_resolution, coefficient]
    print_parametric_solution(argument_range,
                              basis_indexes=simplex_table[0],
                              solution_vector=solution_vector,
                              optimal_resolution=parametric_optimal_resolution,
                              file_name=output)
    if right_border != 1000 and side != 'left':
        b_vector = b_vector + right_border * np.array(parametric_vector[0:len(simplex_table[3])])
        simplex_table[2] = np.dot(reversed_basis_matrix, b_vector).tolist()
        solution_existence = dual_simplex_method(goal_function_vector, simplex_table, simplexes,
                                                 include_logging=include_logging)
        right_border += initial_param_value
        if solution_existence != -1:
            b_vector_variation(goal_function_vector, parametric_vector, simplex_table, simplexes, b_vector,
                               initial_conditions=initial_conditions, initial_param_value=right_border,
                               output=output, dimensionality=dimensionality, side='right',
                               include_logging=include_logging)
        else:
            print_parametric_solution([right_border, 'positive infinity'], file_name=output)

    if left_border != -1000 and side != 'right':
        b_vector = b_vector + left_border * np.array(parametric_vector[0:len(simplex_table[3])])
        simplex_table[2] = np.dot(reversed_basis_matrix, b_vector).tolist()
        solution_existence = dual_simplex_method(goal_function_vector, simplex_table, simplexes,
                                                 include_logging=include_logging)
        left_border += initial_param_value
        if solution_existence != -1:
            b_vector_variation(goal_function_vector, parametric_vector, simplex_table, simplexes, b_vector,
                               initial_conditions=initial_conditions, initial_param_value=left_border,
                               output=output, dimensionality=dimensionality, side='left')
        else:
            print_parametric_solution(['negative infinity', left_border], file_name=output)


def objective_function_variation(*simplex_problem, initial_param_value=0, output,
                                 dimensionality, side='center', include_logging=True):
    goal_function_vector, parametric_vector, simplex_table, simplexes = simplex_problem
    solution_existence = simplex_method(goal_function_vector, simplex_table,
                                        simplexes, include_logging=include_logging)
    optimal_resolution = np.dot(simplex_table[1], simplex_table[2])
    if solution_existence == -1:
        print_parametric_solution([initial_param_value, 'positive infinity'], file_name=output)
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
    if left_border == -1000 and side:
        argument_range.append('negative infinity')
    else:
        argument_range.append(left_border + initial_param_value)
    if right_border == 1000:
        argument_range.append('positive infinity')
    else:
        argument_range.append(right_border + initial_param_value)
    solution_vector = [0] * len(goal_function_vector)
    for i in range(0, len(simplex_table[0])):
        solution_vector[simplex_table[0][i] - 1] = simplex_table[2][i]
    solution_vector = solution_vector[0:dimensionality['number of variables']]
    parametric_optimal_resolution = [optimal_resolution, np.dot(solution_vector,
                                                                parametric_vector[
                                                                0:dimensionality['number of variables']])]
    print_parametric_solution(argument_range,
                              basis_indexes=simplex_table[0],
                              solution_vector=solution_vector,
                              optimal_resolution=parametric_optimal_resolution,
                              file_name=output)
    if right_border != 1000 and side != 'left':
        right_border += initial_param_value
        objective_function_variation(goal_function_vector, parametric_vector, simplex_table, simplexes,
                                     initial_param_value=right_border, output=output,
                                     dimensionality=dimensionality,
                                     side='right',
                                     include_logging=include_logging)
    if left_border != -1000 and side != 'right':
        left_border += initial_param_value
        objective_function_variation(goal_function_vector, parametric_vector, simplex_table, simplexes,
                                     initial_param_value=left_border,
                                     output=output,
                                     dimensionality=dimensionality,
                                     side='left',
                                     include_logging=include_logging)


def linear_programming(input_file_name, output_file_name):
    try:
        parsed_data = parse_simplex_table_xml(input_file_name)
        task_type = parsed_data['task_type']
        match task_type:
            case 'simplex method':
                print_simplex_table(parsed_data['simplex_table'], parsed_data['simplexes'],
                                    logger_file=output_file_name)
                simplex_method(parsed_data['goal_function_vector'],
                               parsed_data['simplex_table'],
                               parsed_data['simplexes'],
                               output=output_file_name,
                               include_logging=True)
            case 'dual simplex method':
                print_simplex_table(parsed_data['simplex_table'], parsed_data['simplexes'],
                                    logger_file=output_file_name)
                dual_simplex_method(parsed_data['goal_function_vector'],
                                    parsed_data['simplex_table'],
                                    parsed_data['simplexes'],
                                    output=output_file_name,
                                    include_logging=True)
            case _:
                raise ValueError('Wrong task type')
    except ValueError as e:
        print(e)
        return


def linear_fractional_programming(input_file_name, output_file_name, include_logging=True):
    try:
        parsed_data = parse_fractional_problem(input_file_name)
        task_type = parsed_data['task_type']
        match task_type:
            case 'dinkelbach method':
                dinkelbach_method(parsed_data['numerator_vector'],
                                  parsed_data['denominator_vector'],
                                  parsed_data['condition_vectors'],
                                  parsed_data['b_vector'],
                                  parsed_data['dimensionality'],
                                  output=output_file_name,
                                  include_logging=include_logging)
            case _:
                raise ValueError('Wrong task type')
    except ValueError as e:
        print(e)
        return


def parametric_programming(input_file_name, output_file_name, include_logging=True):
    try:
        parsed_data = parse_simplex_table_xml(input_file_name)
        task_type = parsed_data['task_type']
        match task_type:
            case 'c variation':
                objective_function_variation(parsed_data['goal_function_vector'],
                                             parsed_data['parametric_vector'],
                                             parsed_data['simplex_table'],
                                             parsed_data['simplexes'],
                                             initial_param_value=0,
                                             output=output_file_name,
                                             dimensionality=parsed_data['dimensionality'],
                                             include_logging=include_logging)
            case 'b variation':
                initial_cond = copy.deepcopy(parsed_data['simplex_table'][3:len(parsed_data['simplex_table'])])
                initial_b_vector = copy.deepcopy(parsed_data['b_vector'])
                simplex_method(parsed_data['goal_function_vector'], parsed_data['simplex_table'],
                               parsed_data['simplexes'])
                b_vector_variation(parsed_data['goal_function_vector'],
                                   parsed_data['parametric_vector'],
                                   parsed_data['simplex_table'],
                                   parsed_data['simplexes'],
                                   initial_b_vector,
                                   initial_conditions=initial_cond,
                                   initial_param_value=0,
                                   output=output_file_name,
                                   dimensionality=parsed_data['dimensionality'],
                                   include_logging=include_logging)
            case _:
                raise ValueError('Wrong task type')
    except ValueError as e:
        print(e)
        return
