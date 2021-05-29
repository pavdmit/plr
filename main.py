from plr import parametric_programming
import sys


if __name__ == '__main__':
    try:
        if len(sys.argv) != 3:
            raise TypeError("Incorrect number of parameters")
    except TypeError as e:
        print(e)
        sys.exit(1)
    parametric_programming(input_file_name=sys.argv[1], output_file_name=sys.argv[2])
