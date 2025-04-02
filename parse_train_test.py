import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    gold_directory = "data/unused_data/test_train/"
    input_directory = "data/encoded_all_lines/"

    column_names = ['role', 'input', 'gold']
    df = pd.DataFrame(columns=column_names)

    for gold_file in os.scandir(gold_directory):
        with open(gold_file, 'r') as file:
            gold_text = file.read()

        n_grams_gold = gold_text.split('\n')

        with open(input_directory + gold_file.name, 'r') as file:
            input_text = file.read()

        n_grams_input = input_text.split('\n')

        # Add rows to the DataFrame
        for i, g in zip(n_grams_input, n_grams_gold):
            new_row = {'role': 'user', 'input': i, 'gold': g}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Print the DataFrame
    # print(df)

    # Split the dataframe into training and testing sets (e.g., 80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the training and testing dataframes to separate CSV files
    train_df.to_csv('data/unused_data/test_train/train_data.csv', index=False)
    test_df.to_csv('data/unused_data/test_train/test_data.csv', index=False)
import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    gold_directory = "data/unused_data/test_train/"
    input_directory = "data/encoded_all_lines/"

    column_names = ['role', 'input', 'gold']
    df = pd.DataFrame(columns=column_names)

    for gold_file in os.scandir(gold_directory):
        with open(gold_file, 'r') as file:
            gold_text = file.read()

        n_grams_gold = gold_text.split('\n')

        with open(input_directory + gold_file.name, 'r') as file:
            input_text = file.read()

        n_grams_input = input_text.split('\n')

        # Add rows to the DataFrame
        for i, g in zip(n_grams_input, n_grams_gold):
            new_row = {'role': 'user', 'input': i, 'gold': g}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Print the DataFrame
    # print(df)

    # Split the dataframe into training and testing sets (e.g., 80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the training and testing dataframes to separate CSV files
    train_df.to_csv('data/unused_data/test_train/train_data.csv', index=False)
    test_df.to_csv('data/unused_data/test_train/test_data.csv', index=False)