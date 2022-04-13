from plr import *

parsed_data = parse_fractional_problem('fractional_problem_test.xml')

# print("parsed data: ")
# print(parsed_data)


def create_fractional_table(numerator_vector, denominator_vector, condition_vectors, b_vector, dimensionality):
    simplex_table = []
    simplexes = {'numerator_simplexes': [],
                 'denominator_simplexes': [],
                 'ratio': []}
    basis_indexes = [i for i in range(dimensionality['number of variables'] + 1,
                                      dimensionality['number of constraints'] + dimensionality[
                                          'number of variables'] + 1)]
    basis_numerator_vector = [Fraction("0/1") for _ in range(len(basis_indexes))]
    basis_denominator_vector = [Fraction("0/1") for _ in range(len(basis_indexes))]
    simplex_table.append(basis_indexes)
    simplex_table.append(basis_numerator_vector)
    simplex_table.append(basis_denominator_vector)
    simplex_table.append(b_vector)
    for i in range(0, dimensionality['number of variables']):
        simplex_table.append(condition_vectors[i])
    identity_matrix = np.eye(len(basis_indexes))
    for i in range(len(basis_indexes)):
        column = map(Fraction, identity_matrix[i])
        column_to_list = list(column)
        simplex_table.append(column_to_list)
    elem = numerator_vector.pop()
    numerator_vector += basis_numerator_vector
    numerator_vector.append(elem)
    elem = denominator_vector.pop()
    denominator_vector += basis_denominator_vector
    denominator_vector.append(elem)
    q_x = (np.dot(basis_numerator_vector, b_vector) + numerator_vector[-1]) / \
          (np.dot(basis_denominator_vector, b_vector) + denominator_vector[-1])
    for i in range(4, len(simplex_table)):
        simplexes['numerator_simplexes'].append(np.dot(basis_numerator_vector, simplex_table[i])
                                                - numerator_vector[i - 4])
        simplexes['denominator_simplexes'].append(np.dot(basis_denominator_vector, simplex_table[i])
                                                  - denominator_vector[i - 4])
        simplexes['ratio'].append(
            simplexes['numerator_simplexes'][i - 4] - q_x * simplexes['denominator_simplexes'][i - 4])

    return {'numerator_vector': numerator_vector,
            'denominator_vector': denominator_vector,
            'simplex_table': simplex_table,
            'simplexes': simplexes, 'b_vector': b_vector, 'dimensionality': dimensionality}


'''data = create_fractional_table(
    parsed_data['numerator_vector'],
    parsed_data['denominator_vector'],
    parsed_data['condition_vectors'],
    parsed_data['b_vector'],
    parsed_data['dimensionality'])'''

'''print("simplex_table: ")
print(data['simplex_table'])

print("simplexes: ")
print(data['simplexes'])

print("numerator_vector: ")
print(data['numerator_vector'])'''


def fractional_simplex_method(numerator_vector, denominator_vector, condition_vectors,
                              b_vector, dimensionality, output, include_logging=False):
    # numerator_vector, denominator_vector, condition_vectors, b_vector, dimensionality = fractional_simplex_method
    data = create_fractional_table(numerator_vector, denominator_vector, condition_vectors,
                                   b_vector, dimensionality)
    simplex_table = data['simplex_table']
    simplexes = data['simplexes']
    min_ratio = 10000
    min_simplex = 10000
    old_basis_position = -1
    new_basis_index = -1
    non_basis_variable_indexes = [i for i in range(1, len(simplex_table) - 3) if i not in simplex_table[0]]
    for i in range(0, len(simplexes['ratio'])):
        if i + 1 in non_basis_variable_indexes and simplexes['ratio'][i] < min_simplex:
            min_simplex = simplexes['ratio'][i]
            new_basis_index = i + 1
    for i in range(len(simplex_table[3])):
        if simplex_table[3][i] / simplex_table[3 + new_basis_index][i] < min_ratio and \
                simplex_table[3 + new_basis_index][i] >= 0:
            min_ratio = simplex_table[3][i] / simplex_table[3 + new_basis_index][i]
            old_basis_position = i
    if old_basis_position == -1:
        return -1
    norm = simplex_table[3 + new_basis_index][old_basis_position]
    for j in range(3, len(simplex_table)):
        simplex_table[j][old_basis_position] = simplex_table[j][old_basis_position] / norm
    for i in range(0, len(simplex_table[0])):
        above_elem = simplex_table[3 + new_basis_index][i]
        for j in range(3, len(simplex_table)):
            if i != old_basis_position:
                simplex_table[j][i] = simplex_table[j][i] - simplex_table[j][old_basis_position] * above_elem
    simplex_table[1][old_basis_position] = numerator_vector[new_basis_index - 1]
    simplex_table[2][old_basis_position] = denominator_vector[new_basis_index - 1]
    simplex_table[0][old_basis_position] = new_basis_index
    simplexes['numerator_simplexes'].clear()
    simplexes['denominator_simplexes'].clear()
    simplexes['ratio'].clear()
    q_x = (np.dot(simplex_table[1], simplex_table[3]) + numerator_vector[-1]) / \
          (np.dot(simplex_table[2], simplex_table[3]) + denominator_vector[-1])
    for i in range(4, len(simplex_table)):
        simplexes['numerator_simplexes'].append(np.dot(simplex_table[1], simplex_table[i])
                                                - numerator_vector[i - 4])
        simplexes['denominator_simplexes'].append(np.dot(simplex_table[2], simplex_table[i])
                                                  - denominator_vector[i - 4])
        simplexes['ratio'].append(
            simplexes['numerator_simplexes'][i - 4] - q_x * simplexes['denominator_simplexes'][i - 4])
    if include_logging:
        print_simplex_table(simplex_table, simplexes, fractional=True, logger_file=output)
    if min(simplexes['ratio']) < 0:
        fractional_simplex_method(numerator_vector,denominator_vector,
                                  condition_vectors, b_vector, dimensionality,
                                  output=output, include_logging=include_logging)


fractional_simplex_method(parsed_data['numerator_vector'], parsed_data['denominator_vector'],
                          parsed_data['condition_vectors'], parsed_data['b_vector'],
                          parsed_data['dimensionality'], output='test_output.txt', include_logging=True)
