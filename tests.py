import unittest
from fractions import Fraction
from plr import parse_simplex_table_xml, simplex_method, dual_simplex_method, parametric_programming,modified_simplex_method


class MyTestCase(unittest.TestCase):
    def test_simplex_method1(self):
        result_simplexes = [Fraction(0, 1), Fraction(0, 1), Fraction(1, 5), Fraction(2, 5)]
        result_simplex_table = [[1, 2], [1, 2], [Fraction(1, 1), Fraction(1, 1)],
                                [Fraction(1, 1), Fraction(0, 1)], [Fraction(0, 1), Fraction(1, 1)],
                                [Fraction(2, 5), Fraction(-1, 10)], [Fraction(-1, 5), Fraction(3, 10)]]
        file_name = "input data/simplex_table1.xml"
        simplex_problem = parse_simplex_table_xml(file_name)
        modified_simplex_method(simplex_problem[0], simplex_problem[1], simplex_problem[2], simplex_problem[3])
        self.assertEqual(simplex_problem[2], result_simplex_table)
        self.assertEqual(simplex_problem[3], result_simplexes)

    def test_simplex_method2(self):
        result_simplexes = [Fraction(0, 1), Fraction(4, 1), Fraction(1, 1), Fraction(0, 1)]
        result_simplex_table = [[1, 4], [1, 0], [Fraction(11, 1), Fraction(39, 1)],
                                [Fraction(1, 1), Fraction(0, 1)], [Fraction(7, 1), Fraction(25, 1)],
                                [Fraction(1, 1), Fraction(3, 1)], [Fraction(0, 1), Fraction(1, 1)]]
        file_name = "input data/simplex_table2.xml"
        simplex_problem = parse_simplex_table_xml(file_name)
        modified_simplex_method(simplex_problem[0], simplex_problem[1], simplex_problem[2], simplex_problem[3])
        self.assertEqual(simplex_problem[2], result_simplex_table)
        self.assertEqual(simplex_problem[3], result_simplexes)

    def test_simplex_method3(self):
        result_simplexes = [Fraction(11, 1), Fraction(0, 1), Fraction(3, 1), Fraction(0, 1), Fraction(2, 1)]
        result_simplex_table = [[4, 2], [0, 2], [Fraction(34, 1), Fraction(13, 1)],
                                [Fraction(15, 1), Fraction(7, 1)], [Fraction(0, 1), Fraction(1, 1)],
                                [Fraction(8, 1), Fraction(2, 1)], [Fraction(1, 1), Fraction(0, 1)],
                                [Fraction(2, 1), Fraction(1, 1)]]
        file_name = "input data/simplex_table3.xml"
        simplex_problem = parse_simplex_table_xml(file_name)
        modified_simplex_method(simplex_problem[0], simplex_problem[1], simplex_problem[2], simplex_problem[3])
        self.assertEqual(simplex_problem[2], result_simplex_table)
        self.assertEqual(simplex_problem[3], result_simplexes)

    def test_simplex_method4(self):
        result_simplexes = [Fraction(0, 1), Fraction(107, 22), Fraction(0, 1), Fraction(13, 22), Fraction(5, 11)]
        result_simplex_table = [[1, 3], [1, 3], [Fraction(1, 1), Fraction(3, 2)],
                                [Fraction(1, 1), Fraction(0, 1)], [Fraction(-3, 11), Fraction(23, 22)],
                                [Fraction(0, 1), Fraction(1, 1)], [Fraction(2, 11), Fraction(3, 22)],
                                [Fraction(-1, 11), Fraction(2, 11)]]
        file_name = "input data/simplex_table4.xml"
        simplex_problem = parse_simplex_table_xml(file_name)
        modified_simplex_method(simplex_problem[0], simplex_problem[1], simplex_problem[2], simplex_problem[3])
        self.assertEqual(simplex_problem[2], result_simplex_table)
        self.assertEqual(simplex_problem[3], result_simplexes)

    def test_simplex_method5(self):
        result_simplexes = [Fraction(24, 19), Fraction(0, 1), Fraction(0, 1), Fraction(10, 19), Fraction(13, 19)]
        result_simplex_table = [[3, 2], [4, 1], [Fraction(27, 19), Fraction(1, 19)],
                                [Fraction(11, 19), Fraction(18, 19)], [Fraction(0, 1), Fraction(1, 1)],
                                [Fraction(1, 1), Fraction(0, 1)], [Fraction(3, 19), Fraction(-2, 19)],
                                [Fraction(2, 19), Fraction(5, 19)]]
        file_name = "input data/simplex_table5.xml"
        simplex_problem = parse_simplex_table_xml(file_name)
        modified_simplex_method(simplex_problem[0], simplex_problem[1], simplex_problem[2], simplex_problem[3])
        self.assertEqual(simplex_problem[2], result_simplex_table)
        self.assertEqual(simplex_problem[3], result_simplexes)

    '''def test_dual_simplex_method1(self):
        result_simplexes = [Fraction(0, 1), Fraction(0, 1), Fraction(1, 5), Fraction(2, 5)]
        result_simplex_table = [[1, 2], [1, 2], [Fraction(1, 1), Fraction(1, 1)],
                                [Fraction(1, 1), Fraction(0, 1)], [Fraction(0, 1), Fraction(1, 1)],
                                [Fraction(2, 5), Fraction(-1, 10)], [Fraction(-1, 5), Fraction(3, 10)]]
        file_name = "input data/dual_simplex_table1.xml"
        simplex_problem = parse_simplex_table_xml(file_name)
        dual_simplex_method(simplex_problem[0], simplex_problem[1], simplex_problem[2], simplex_problem[3])
        self.assertEqual(simplex_problem[2], result_simplex_table)
        self.assertEqual(simplex_problem[3], result_simplexes)

    def test_dual_simplex_method2(self):
        result_simplexes = [Fraction(0, 1), Fraction(4, 1), Fraction(1, 1), Fraction(0, 1)]
        result_simplex_table = [[1, 4], [1, 0], [Fraction(11, 1), Fraction(39, 1)],
                                [Fraction(1, 1), Fraction(0, 1)], [Fraction(7, 1), Fraction(25, 1)],
                                [Fraction(1, 1), Fraction(3, 1)], [Fraction(0, 1), Fraction(1, 1)]]
        file_name = "input data/dual_simplex_table2.xml"
        simplex_problem = parse_simplex_table_xml(file_name)
        dual_simplex_method(simplex_problem[0], simplex_problem[1], simplex_problem[2], simplex_problem[3])
        self.assertEqual(simplex_problem[2], result_simplex_table)
        self.assertEqual(simplex_problem[3], result_simplexes)

    def test_dual_simplex_method3(self):
        result_simplexes = [Fraction(11, 1), Fraction(0, 1), Fraction(3, 1), Fraction(0, 1), Fraction(2, 1)]
        result_simplex_table = [[4, 2], [0, 2], [Fraction(34, 1), Fraction(13, 1)],
                                [Fraction(15, 1), Fraction(7, 1)], [Fraction(0, 1), Fraction(1, 1)],
                                [Fraction(8, 1), Fraction(2, 1)], [Fraction(1, 1), Fraction(0, 1)],
                                [Fraction(2, 1), Fraction(1, 1)]]
        file_name = "input data/dual_simplex_table3.xml"
        simplex_problem = parse_simplex_table_xml(file_name)
        dual_simplex_method(simplex_problem[0], simplex_problem[1], simplex_problem[2], simplex_problem[3])
        self.assertEqual(simplex_problem[2], result_simplex_table)
        self.assertEqual(simplex_problem[3], result_simplexes)

    def test_dual_simplex_method4(self):
        result_simplexes = [Fraction(0, 1), Fraction(107, 22), Fraction(0, 1), Fraction(13, 22), Fraction(5, 11)]
        result_simplex_table = [[1, 3], [1, 3], [Fraction(1, 1), Fraction(3, 2)],
                                [Fraction(1, 1), Fraction(0, 1)], [Fraction(-3, 11), Fraction(23, 22)],
                                [Fraction(0, 1), Fraction(1, 1)], [Fraction(2, 11), Fraction(3, 22)],
                                [Fraction(-1, 11), Fraction(2, 11)]]
        file_name = "input data/dual_simplex_table4.xml"
        simplex_problem = parse_simplex_table_xml(file_name)
        dual_simplex_method(simplex_problem[0], simplex_problem[1], simplex_problem[2], simplex_problem[3])
        self.assertEqual(simplex_problem[2], result_simplex_table)
        self.assertEqual(simplex_problem[3], result_simplexes)

    def test_dual_simplex_method5(self):
        result_simplexes = [Fraction(24, 19), Fraction(0, 1), Fraction(0, 1), Fraction(10, 19), Fraction(13, 19)]
        result_simplex_table = [[3, 2], [4, 1], [Fraction(27, 19), Fraction(1, 19)],
                                [Fraction(11, 19), Fraction(18, 19)], [Fraction(0, 1), Fraction(1, 1)],
                                [Fraction(1, 1), Fraction(0, 1)], [Fraction(3, 19), Fraction(-2, 19)],
                                [Fraction(2, 19), Fraction(5, 19)]]
        file_name = "input data/dual_simplex_table5.xml"
        simplex_problem = parse_simplex_table_xml(file_name)
        dual_simplex_method(simplex_problem[0], simplex_problem[1], simplex_problem[2], simplex_problem[3])
        self.assertEqual(simplex_problem[2], result_simplex_table)
        self.assertEqual(simplex_problem[3], result_simplexes)'''


if __name__ == '__main__':
    unittest.main()
