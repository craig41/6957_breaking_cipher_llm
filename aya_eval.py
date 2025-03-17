
import os
import time
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main(args):
    print("getting started")
    start = time.time()
    print("step 1")

    print("step 2")

    checkpoint = "CohereForAI/aya-101"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    print("step 3")
    print("step 4")

    # in_file = "data/encoded/D1_5_word_groups_2_test.txt"
    directory = "data/encoded/"

    for in_file in os.scandir(directory):
        print("step 5")
        with open(in_file, 'r') as file:
            text = file.read()
        n_grams = text.split('\n')

        prompt_prefix = "Can you tell what the encoded word is? Respond with the original passage replacing your translation of the encoded word, for example if I promt 'my yodz2rMf+AuOmKOjdaAplIiyHivku2xJPOIX7y3uJQ8= is John Smith' and your guess is 'name', then return 'my name is John Smith': "

        n_grams = [prompt_prefix + x for x in n_grams]

        inputs = []
        outputs = []

        print("step 5")
        for x in n_grams:
            inputs.append(tokenizer.encode(x, return_tensors="pt"))
            break

        inputs2 = tokenizer(n_grams, padding=True, return_tensors="pt")

        print("step 6")
        for i in inputs:
            with torch.no_grad():
                output = aya_model.generate(i, max_new_tokens=128)
                outputs.append(output)


        decoded_outs = []

        print("step 7")
        for o in outputs:
            dec_out = tokenizer.decode(o, skip_special_tokens=True)
            decoded_outs.append(dec_out)
            print(dec_out + "\n")
            decoded_outs.append(dec_out)

        end = time.time()

        print("Time taken: ", end-start)

        print("step 8")
        orig_filename = os.path.basename(in_file)
        orig_filename = os.path.splitext(orig_filename)[0]
        filename = "data/parsed/" + orig_filename + "_predictions.txt"

        with open(filename, "w") as txt_file:
            for line in decoded_outs:
                txt_file.write(line + "\n")
        break


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./output/")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    main(args)
