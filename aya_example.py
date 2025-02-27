# pip install -q transformers
import os
import time
import torch

# from safetensors import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer







if __name__ == "__main__":

    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    checkpoint = "CohereForAI/aya-101"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    in_file = "data/encoded/D1_5_word_groups_2.txt"
    with open(in_file, 'r') as file:
        text = file.read()
    n_grams = text.split('\n')

    # inputs = []
    # for x in n_grams:
    #     inputs.append(tokenizer.encode(x, return_tensors="pt"))
    #
    # outputs = []
    #
    # for i in inputs:
    #     out = aya_model.generate(i, max_new_tokens=128)
    #     outputs.append(tokenizer.decode(out[0]))

    # device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.backends.mps.is_available:
    #     device = torch.device("mps")
    # inputs = tokenizer(n_grams, padding=True, return_tensors="pt", truncation=True)
    inputs = tokenizer(n_grams, padding=True, return_tensors="pt", truncation=True).input_ids.to(device)

    with torch.no_grad():
        outputs = aya_model.generate(inputs, max_new_tokens=128)

    decoded_out = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    end = time.time()

    print("Time taken: ", end-start)
    for o in outputs:
        print(o)

    orig_filename = os.path.basename(in_file)
    orig_filename = os.path.splitext(orig_filename)[0]
    filename = "data/parsed/" + orig_filename + "_predictions.txt"

    with open(filename, "w") as txt_file:
        for line in outputs:
            txt_file.write(" ".join(line) + "\n")