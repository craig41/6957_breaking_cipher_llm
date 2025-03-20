import os, sys ; sys.path.append(os.getcwd()) #allow local imports
os.environ["TRANSFORMERS_CACHE"] = "/scratch/general/vast/u1380656/huggingface_cache"

import argparse
from utils import *
from models import *
from workflows import *
from huggingface_hub import login

def generate_report(instances, outputs, conversations):
    pass

translated = []
original = []
def callback(instance, gold, output, conversation):
    translated.append(output)
    original.append(instance['text'])
    print(f'{str(conversation)}\n\nINPUT:\n{str(instance["text"])}\nOUTPUT:\n{str(output)}\n')

# # # # # TEMPORARY
# # # # # 

def main(args, key):
    print(args)
    if key is not None:
        args.model_args = key
    if args.n is None:
            data = read_data(args.data, split=args.split)
            data = data.filter(lambda x: x['glottocode'] == args.source)
            data = data[args.s:]
    else:
        data = read_data(args.data, split=args.split)
        data = data.filter(lambda x: x['glottocode'] == args.source)
        data = data[args.s:args.n+args.s]

    run (
            data,
            pipeline = Pipeline (
                                    load_model(args.model),
                                    load_workflow(args.workflow)
                                ), 
            per_instance_callback=callback,
            report_generator = generate_report
        )
    with open(args.output + '/translated.txt', 'w') as f:
        for line in translated:
            f.write(line + '\n')

    with open(args.output + '/original.txt', 'w') as f:
        for line in original:
            f.write(line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="select which model to load"
    )
    parser.add_argument(
        "-a",
        "--model_args",
        type=str,
        default=None,
        help="args to pass to the model"
    )
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        help="select which workflow to load"
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="hf dataset to load"
    )
    parser.add_argument(
        "-s",
        "--s",
        type=int,
        default=0,
        help="number of instances to skip over"
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=None,
        help="number of instances to test over"
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        default=None,
        help="API key to be passed to main model"
    )
    parser.add_argument(
        "-src",
         "--source",
         type=str,
         default="stan1293"
    )
    parser.add_argument(
        "--split",
         type=str,
         default="dev"
    )

    parser.add_argument(
        "--output",
         type=str,
         default=None
    )

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    key = args.key
    if key is not None:
        del args.key
    login(token = key)
    main(args, key)
