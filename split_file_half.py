
def split_file_in_half(input_filepath, output_filepath1, output_filepath2):
    """Splits a text file in half by lines.

    Args:
        input_filepath: Path to the input text file.
        output_filepath1: Path to the first output file (first half).
        output_filepath2: Path to the second output file (second half).
    """
    with open(input_filepath, 'r') as f:
        lines = f.readlines()
        midpoint = len(lines) // 2

        with open(output_filepath1, 'w') as f1:
            f1.writelines(lines[:midpoint])

        with open(output_filepath2, 'w') as f2:
            f2.writelines(lines[midpoint:])



if __name__ == '__main__':
    input_file = 'data/encoded_all_lines/10my_input_file.txt'
    output_file_1 = 'output_part1.txt'
    output_file_2 = 'output_part2.txt'

    split_file_in_half(input_file, output_file_1, output_file_2)


