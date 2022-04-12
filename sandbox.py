from plr import *

parsed_data = parse_fractional_problem('fractional_problem_test.xml')

print("parsed data: ")
print(parsed_data)


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


def fractional_simplex_method(*fractional_simplex_method, output, include_logging=False):
    numerator_vector, denominator_vector, condition_vectors, b_vector, dimensionality = fractional_simplex_method


data = create_fractional_table(
    parsed_data['numerator_vector'],
    parsed_data['denominator_vector'],
    parsed_data['condition_vectors'],
    parsed_data['b_vector'],
    parsed_data['dimensionality'])

print("simplex_table: ")
print(data['simplex_table'])

print("simplexes: ")
print(data['simplexes'])