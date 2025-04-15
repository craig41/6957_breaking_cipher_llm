from datasets import load_dataset
import argparse
import os
from huggingface_hub import login

from constants import Constants

HUGGINGFACE_TOKEN =  Constants.HUGGINGFACE_TOKEN_2


def read_data(source_glotto, target_glotto, split='dev', num_lines=5):
    """
    Reads and prints examples from the official Flores dataset.

    Args:
        source_lang (str): Source language (e.g., 'ind_Latn').
        target_lang (str): Target language (e.g., 'eng_Latn').
        split (str): Dataset split ('dev', 'test', 'devtest').
        num_lines (int): Number of lines to preview.
    """
    dataset = load_dataset("openlanguagedata/flores_plus", split=split, cache_dir=os.getcwd() + "/data",
                           trust_remote_code=True)

    japanese = dataset.filter(lambda x: x['glottocode'] == 'nucl1643')
    jingpho = dataset.filter(lambda x: x['glottocode'] == 'kach1280')
    maithili = dataset.filter(lambda x: x['glottocode'] == 'mait1250')
    marathi = dataset.filter(lambda x: x['glottocode'] == 'mara1378')
    rundi = dataset.filter(lambda x: x['glottocode'] == 'rund1242')
    english = dataset.filter(lambda x: x['glottocode'] == 'stan1293')

    english_small = english.select(range(200))

    if not os.path.exists(os.getcwd() + "/data/low_resource_language/"):
        os.mkdir(os.getcwd() + "/data/low_resource_language/")

    count = 0
    with open(os.getcwd() + "/data/low_resource_language/" + "golds_small.txt", "w") as file:
        for item in english:
            count += 1
            if count > 250:
                file.write(item["text"] + "\n")

    count = 0
    with open(os.getcwd() + "/data/low_resource_language/" + "japanese.txt", "w") as file:
        for item in japanese:
            count += 1
            if count > 250:
                file.write(item["text"] + "\n")

    count = 0
    with open(os.getcwd() + "/data/low_resource_language/" + "jingpho.txt", "w") as file:
        for item in jingpho:
            count += 1
            if count > 250:
                file.write(item["text"] + "\n")

    count = 0
    with open(os.getcwd() + "/data/low_resource_language/" + "maithili.txt", "w") as file:
        for item in maithili:
            count += 1
            if count > 250:
                file.write(item["text"] + "\n")

    count = 0
    with open(os.getcwd() + "/data/low_resource_language/" + "marathi.txt", "w") as file:
        for item in marathi:
            count += 1
            if count > 250:
                file.write(item["text"] + "\n")

    count = 0
    with open(os.getcwd() + "/data/low_resource_language/" + "rundi.txt", "w") as file:
        for item in rundi:
            count += 1
            if count > 250:
                file.write(item["text"] + "\n")


    # japanese = japanese.select(range(200))
    #
    # for i, item in enumerate(zip(japanese, english)):
    #     if i >= num_lines:
    #         break
    #     ja_item, en_item = item
    #     print(f"Japanese: {ja_item['text']}\nEnglish: {en_item['text']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-src", "--source",
        default="ind_Latn",
        type=str,
        help="Source language (e.g., ind_Latn for Indonesian)"
    )
    parser.add_argument(
        "-tgt", "--target",
        default="eng_Latn",
        type=str,
        help="Target language (e.g., eng_Latn for English)"
    )
    parser.add_argument(
        "-sp", "--split",
        default="dev",
        type=str,
        help="Dataset split (dev, devtest, test)"
    )
    parser.add_argument(
        "-n", "--num",
        default=5,
        type=int,
        help="Number of lines to display"
    )
    args = parser.parse_args()
    login(token=HUGGINGFACE_TOKEN)
    read_data(args.source, args.target, args.split, args.num)