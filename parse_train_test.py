import os
import pandas as pd
# from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    gold_directory = "data/unused_data/final_test/plain_text/"
    input_directory = "data/unused_data/final_test/mixed_text/"

    gold_file_name = 'final_eval.txt'

    column_names = ['input', 'gold', 'category']
    df = pd.DataFrame(columns=column_names)

    for input_file in os.scandir(input_directory):
        with open(input_file, 'r') as file:
            input_text = file.read()

        n_grams_input = input_text.split('\n')

        with open(gold_directory + gold_file_name, 'r') as file:
            gold_text = file.read()

        n_grams_gold = gold_text.split('\n')

        # Add rows to the DataFrame
        for i, g in zip(n_grams_input, n_grams_gold):
            new_row = { 'input': i, 'gold': g, 'category': 'final_eval'}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Print the DataFrame
    # print(df)
    
    # save final evaluation data
    df.to_csv('data/unused_data/final_test/final_eval.csv', index=False)

    # # Split the dataframe into training and testing sets (e.g., 80% train, 20% test)
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # # Save the training and testing dataframes to separate CSV files
    # train_df.to_csv('data/unused_data/test_train_trimmed/emoji/train_data.csv', index=False)
    # test_df.to_csv('data/unused_data/test_train_trimmed/emoji/test_data.csv', index=False)
