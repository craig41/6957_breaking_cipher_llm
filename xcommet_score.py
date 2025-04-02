import os
import re

from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)


def calc_xcommet(inputs, outputs, golds):
    # source - input, mt - output (ai guess), ref - gold

    # data = [
    #     {
    #         "src": "Boris Johnson teeters on edge of favour with Tory MPs",
    #         "mt": "Boris Johnson ist bei Tory-Abgeordneten völlig in der Gunst",
    #         "ref": "Boris Johnsons Beliebtheit bei Tory-MPs steht auf der Kippe"
    #     }
    # ]

    data = []

    j = 0
    for i, o, g in zip(inputs, outputs, golds):
        j = j + 1
        data_obj = {"src": i, "mt": o, "ref": g}
        data.append(data_obj)
        if j == 10: break

    model_output = model.predict(data, batch_size=8, gpus=1)
    # Segment-level scores
    print (model_output.scores)

    # System-level score
    print (model_output.system_score)

    # Score explanation (error spans)
    print (model_output.metadata.error_spans)



if __name__ == '__main__':
    # need input file
    input_dir = sorted(os.listdir("data/encoded_limited_lines"))
    for in_file in input_dir:
        with open(input_dir + in_file, 'r') as file:
            gold_lines = file.readlines()
            # Remove trailing newline characters from each line
            gold_lines = [line.rstrip('\n') for line in gold_lines]
            gold_lines = [re.sub(r"<pad>", "", line) for line in gold_lines]
            gold_lines = [re.sub(r"</s>", "", line) for line in gold_lines]
            gold_lines = [line.strip() for line in gold_lines]
            gold_lines = [string for string in gold_lines if string]

    # need prediction file
    # get dir for llama
    pred_dir = sorted(os.listdir("data/llama_pred_partial"))
    for pred_file in pred_dir:
        with open(pred_dir + pred_file, 'r') as file:
            pred_lines = file.readlines()
            # Remove trailing newline characters from each line
            pred_lines = [line.rstrip('\n') for line in pred_lines]
            pred_lines = [string for string in pred_lines if string]

    # need parsed file
    gold_dir = sorted(os.listdir("data/parsed"))
    for gold_file in gold_dir:
        with open(gold_dir + gold_file, 'r') as file:
            gold_lines = file.readlines()

    calc_xcommet(pred_lines, gold_lines, gold_lines)
