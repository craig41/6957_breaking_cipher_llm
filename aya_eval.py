# pip install -q transformers
import os
import time
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


if __name__ == "__main__":

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    checkpoint = "CohereForAI/aya-101"
    # checkpoint = "CohereForAI/aya-expanse-8b"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    directory = "data/encoded_limited_lines/"

    for in_file in os.scandir(directory):
        with open(in_file, 'r') as file:
            text = file.read()
        n_grams = text.split('\n')

        prompt_prefix = "This message has an encoded word in it, can you tell me what the encoded word is by responding with the entire message replacing the encoded word with the actual word, for example if I give you the message 'my yodz2rMf+AuOmKOjdaAplIiyHivku2xJPOIX7y3uJQ8= is John Smith' and the encoded word is 'name', then return 'my name is John Smith': "

        n_grams = [prompt_prefix + x for x in n_grams]

        inputs = []
        decoded_outs = []

        for x in n_grams:
            inputs.append(tokenizer.encode(x, return_tensors="pt"))


        for i in inputs:
            with torch.no_grad():
                output = aya_model.generate(i, max_new_tokens=128)
                decoded_out = tokenizer.decode(output[0])
            decoded_outs.append(decoded_out)


        end = time.time()

        print("Time taken: ", end-start)

        orig_filename = os.path.basename(in_file)
        orig_filename = os.path.splitext(orig_filename)[0]
        filename = "data/aya_pred_partial/" + orig_filename + ".txt"

        with open(filename, "w") as txt_file:
            for line in decoded_outs:
                txt_file.write(line + "\n")