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

    in_file = "data/encoded/D1_5_word_groups_2_test.txt"
    with open(in_file, 'r') as file:
        text = file.read()
    n_grams = text.split('\n')

    # append prompt
    # prompt_prefix = "Can you tell what the encoded word is? Respond with the original passage replacing your translation of the encoded word: "
    prompt_prefix = "Can you tell what the encoded word is? Respond with the original passage replacing your translation of the encoded word, for example if I promt 'my yodz2rMf+AuOmKOjdaAplIiyHivku2xJPOIX7y3uJQ8= is John Smith' and your guess is 'name', then return 'my name is John Smith': "

    n_grams = [prompt_prefix + x for x in n_grams]

    inputs = []
    outputs = []


    for x in n_grams:
        inputs.append(tokenizer.encode(x, return_tensors="pt"))

    # for i in inputs:
    #     out = aya_model.generate(i, max_new_tokens=128)
    #     outputs.append(tokenizer.decode(out[0]))

    # device = torch.device("mps" if torch.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.backends.mps.is_available:
    #     device = torch.device("mps")
    # inputs = tokenizer(n_grams, padding=True, return_tensors="pt").input_ids.to(device)

    inputs2 = tokenizer(n_grams, padding=True, return_tensors="pt")

    for i in inputs:
        with torch.no_grad():
            output = aya_model.generate(i, max_new_tokens=128)
            outputs.append(output)
            # output = aya_model.generate(inputs[0], max_new_tokens=128)
            # outputs = aya_model.generate(inputs, max_new_tokens=128)

    # decoded_out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # decoded_out = tokenizer.decode(output[0], skip_special_tokens=True)

    decoded_outs = []

    for o in outputs:
        dec_out = tokenizer.decode(o, skip_special_tokens=True)
        decoded_outs.append(dec_out)
        print(dec_out + "\n")
        decoded_outs.append(dec_out)


    end = time.time()

    print("Time taken: ", end-start)
    # for o in outputs:
    #     print(o)

    # print(decoded_out)

    orig_filename = os.path.basename(in_file)
    orig_filename = os.path.splitext(orig_filename)[0]
    filename = "data/parsed/" + orig_filename + "_predictions.txt"

    with open(filename, "w") as txt_file:
        for line in decoded_outs:
            txt_file.write(line + "\n")
        # txt_file.write(decoded_out)