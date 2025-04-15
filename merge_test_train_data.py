import demoji
import emoji
import pandas as pd

if __name__ == '__main__':
    df_test_emoji = pd.read_csv('data/unused_data/test_train_trimmed/emoji/test_data.csv')
    df_test_emoji = df_test_emoji.dropna()
    df_train_emoji = pd.read_csv('data/unused_data/test_train_trimmed/emoji/train_data.csv')
    df_train_emoji = df_train_emoji.dropna()
    df_test_encrypted = pd.read_csv('data/unused_data/test_train_trimmed/encrypted/test_data.csv')
    df_test_encrypted = df_test_encrypted.dropna()
    df_train_encrypted = pd.read_csv('data/unused_data/test_train_trimmed/encrypted/train_data.csv')
    df_train_encrypted = df_train_encrypted.dropna()
    df_test_low_resource = pd.read_csv('data/unused_data/test_train_trimmed/low_resource/test_data.csv')
    df_test_low_resource = df_test_low_resource.dropna()
    df_train_low_resource = pd.read_csv('data/unused_data/test_train_trimmed/low_resource/train_data.csv')
    df_train_low_resource = df_train_low_resource.dropna()

    # cut dataframes to a similar size
    df_test_emoji = df_test_emoji.sample(frac=1).reset_index(drop=True)
    df_train_emoji = df_train_emoji.sample(frac=1).reset_index(drop=True)
    df_test_emoji = df_test_emoji.sample(n=700)
    df_train_emoji = df_train_emoji.sample(n=2500)
    df_test_encrypted = df_test_encrypted.sample(frac=1).reset_index(drop=True)
    df_train_encrypted = df_train_encrypted.sample(frac=1).reset_index(drop=True)
    df_test_encrypted = df_test_encrypted.sample(n=700)
    df_train_encrypted = df_train_encrypted.sample(n=2500)
    df_test_low_resource = df_test_low_resource.sample(frac=1).reset_index(drop=True)
    df_train_low_resource = df_train_low_resource.sample(frac=1).reset_index(drop=True)
    df_test_low_resource = df_test_low_resource.sample(n=700)
    df_train_encrypted = df_train_encrypted.sample(n=2500)

    df_combined_test = pd.concat([df_test_emoji, df_test_encrypted, df_test_low_resource])
    df_combined_test = df_combined_test.sample(frac=1).reset_index(drop=True)
    df_combined_test.to_csv('data/unused_data/combined_test_data.csv', index=False)

    df_combined_train = pd.concat([df_train_emoji, df_train_encrypted, df_train_low_resource])
    df_combined_train = df_combined_train.sample(frac=1).reset_index(drop=True)
    df_combined_train.to_csv('data/unused_data/combined_train_data.csv', index=False)

    # file = 'data/emoji/gold/plaintext.txt'
    #
    # with open(file, 'r', encoding='utf-8') as f:
    #     input = f.read()
    #
    # lines = input.split('\n')
    #
    # new_lines = []
    # for l in lines:
    #     nl = emoji.demojize(l)
    #     new_lines.append(nl)
    #
    # new_file = 'data/emoji/gold/plaintext.txt'
    # with open(new_file, 'w', encoding='utf-8') as f:
    #     for l in new_lines:
    #         f.write(l + '\n')
