import csv
import os
import random
import re

import numpy as np
import pandas as pd
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64


def file_parser(in_file):
    with open(in_file, 'r') as file:
        text = file.read()

    return text

def csv_file_parser(CSVfile):
    df = pd.read_csv(CSVfile, header=None)
    return df

def encrypt_word(word, key):
    """Encrypts a word using AES-256 in CBC mode."""

    # Convert the key to bytes if it's not already
    if isinstance(key, str):
        key = key.encode()

    # Generate a random initialization vector (IV)
    iv = os.urandom(16)

    # Create an AES cipher object
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Pad the word to a multiple of 16 bytes
    padded_word = pad(word.encode(), 16)

    # Encrypt the padded word
    encrypted_word = cipher.encrypt(padded_word)

    # Encode the encrypted word and IV using base64 for storage or transmission
    encoded_word = base64.b64encode(iv + encrypted_word).decode()

    return encoded_word

def decrypt_word(encoded_word, key):
    """Decrypts a word that was encrypted using the encrypt_word function."""

    # Convert the key to bytes if it's not already
    if isinstance(key, str):
        key = key.encode()

    # Decode the base64 encoded string
    decoded_word = base64.b64decode(encoded_word)

    # Extract the IV from the beginning of the decoded string
    iv = decoded_word[:16]

    # Extract the encrypted word from the remaining part of the decoded string
    encrypted_word = decoded_word[16:]

    # Create an AES cipher object
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Decrypt the encrypted word
    decrypted_word = cipher.decrypt(encrypted_word)

    # Unpad the decrypted word
    unpadded_word = unpad(decrypted_word, 16)

    return unpadded_word.decode()

def get_random_word(sentence):
    words = sentence.split(' ')
    return random.choice(words)

def write_file(filename, data):
    with open(filename, 'w') as file:
        for x in data:
            file.write(x + "\n")
    # with open(filename, 'w', ) as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['ID', 'Prediction'])
    #     for x in data:
    #         writer.writerow(x)

def replace_word(text, old_word, new_word):
    return re.sub(r'\b' + re.escape(old_word) + r'\b', new_word, text)

def write_enc_file(arr_one, file_name):
    out_dir = "data/encoded/"
    orig_filename = os.path.basename(file_name)
    with open(out_dir + orig_filename, 'w') as enc_file:
        for x in arr_one:
            enc_file.write(x + "\n")

if __name__ == '__main__':

    files = ['data/storytale/D1.txt','data/storytale/D2.txt','data/storytale/D3.txt','data/storytale/D4.txt']

    for file in files:
        data = file_parser(file)
        words = data.split(' ')
        np_words = np.array(words)

        n_s = [3, 5, 10]
        for n_i in n_s:
            n_word_groups = [' '.join(np_words[i:i + n_i]) for i in range(0, len(np_words) - (n_i -1), 1)]
            orig_filename = os.path.basename(file)
            orig_filename = os.path.splitext(orig_filename)[0]
            filename = "data/parsed/" + orig_filename + "_" + str(n_i) + "_word_groups.txt"
            np.savetxt(filename, n_word_groups, delimiter=",", fmt="%s")

        filename = "data/parsed/" + orig_filename + "_1_word_groups.txt"
        np.savetxt(filename, words, delimiter=",", fmt="%s")


    directory = "data/parsed"

    key = os.urandom(32)
    print("key:", key)

    for file in os.listdir(directory):
        file_name = os.path.join(directory, file)
        data = csv_file_parser(file_name)
        data = np.array(data)
        if any(file for n in ['_3_', '_5_', '_10_'] if (n in file)):
            n = 1
            if ('_3_' in file):
                n = 3
            if ('_5_' in file):
                n = 5
            if ('_10_' in file):
                n = 10
            # encrypt random word
            for i in range(n):
                encrypted_data = []
                for x in data:
                    if (x[0].count(" ") > 0):
                        random_word = x[0].split(' ')[i]
                    else:
                        random_word = x[0]
                    random_word_encrypted = encrypt_word(random_word, key)
                    encrypted = replace_word(x[0], random_word, random_word_encrypted)
                    encrypted_data.append(encrypted)

                if (n > 1):
                    file_new = file.split('.')[0]+ '_' + str(i)
                    # file_name_new = file_new + '.csv'
                    file_name_new = file_new + '.txt'
                    file_name = os.path.join(directory, file_name_new)
                write_enc_file(encrypted_data, file_name)
        else:
            encrypted_data = []
            for x in data:
                x_encrypted = encrypt_word(x[0], key)
                encrypted_data.append(x_encrypted)

            write_enc_file(encrypted_data, file_name)





