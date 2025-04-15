import os

def trim_file(file_path, num_lines=1600):
    """Trims a text file to a specified number of lines.

    Args:
        file_path (str): The path to the text file.
        num_lines (int): The number of lines to keep (default: 200).
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return

    with open(file_path, 'w') as file:
        file.writelines(lines[:num_lines])



if __name__ == "__main__":
    directory = "data/unused_data/test_train_trimmed/"

    for in_file in os.scandir(directory):
        with open(in_file, 'r') as file:
            # Example usage
            trim_file(in_file) # Trims 'my_text_file.txt' to 250 lines