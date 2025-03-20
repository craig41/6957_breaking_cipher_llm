from datasets import load_dataset
import argparse
import os
from huggingface_hub import login

HUGGINGFACE_TOKEN = "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"

def read_data(source_glotto, target_glotto, split='dev', num_lines=5):
    """
    Reads and prints examples from the official Flores dataset.

    Args:
        source_lang (str): Source language (e.g., 'ind_Latn').
        target_lang (str): Target language (e.g., 'eng_Latn').
        split (str): Dataset split ('dev', 'test', 'devtest').
        num_lines (int): Number of lines to preview.
    """
    dataset = load_dataset("openlanguagedata/flores_plus", split=split, trust_remote_code=True)
    
    japanese = dataset.filter(lambda x: x['glottocode'] == 'nucl1643')
    english = dataset.filter(lambda x: x['glottocode'] == 'stan1293')

    for i, item in enumerate(zip(japanese, english)):
        if i >= num_lines:
            break
        ja_item, en_item = item
        print(f"Japanese: {ja_item['text']}\nEnglish: {en_item['text']}\n")

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
    login(token = HUGGINGFACE_TOKEN, add_to_git_credential=True)
    read_data(args.source, args.target, args.split, args.num)
