#!/scratch/general/vast/u1380656/CondaQA_Private/condaqa/bin/python
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --mem=80GB
#SBATCH -o outputs-%j
import os, sys ; sys.path.append(os.getcwd()) #allow local imports
os.environ["TRANSFORMERS_CACHE"] = "/scratch/general/vast/u1380656/huggingface_cache"

import argparse
from utils import *
from models import *
from workflows import *

def generate_report(instances, golds, outputs, conversations):
    #def mean(A):
    #    return sum(A)/len(A) if len(A) > 0 else 0
    #scores = {1:[], 2:[], 3:[]}
    #for instance, gold, output in zip(instances, golds, outputs):
    #    scores[instance['PassageEditID']].append(gold == output)
    #for key, value in scores.items():
    #    print(key, mean(value))
    #for conversation in conversations:
    #    print(conversation)
    # # # # # for instance, output, conversation in zip(instances, outputs, conversations):
    #    print("original:", instance['original sentence'])
    #    print("edit:", output)
    #    print(conversation)
    #    print()
    pass

# # # # # TEMPORARY
# # # # # 

def main(args, key):
    print(args)
    if key is not None:
        args.model_args = key
    if args.n is None:
            data = read_data(args.data)
    else:
        data = read_data(args.data)[slice(args.s, args.s+args.n)]
    run (
            data,
            pipeline = Pipeline (
                                    load_model(args.model, *([args.model_args] if args.model_args else [])),
                                    load_workflow(args.workflow)
                                ), 
            per_instance_callback=(lambda instance, gold, output, conversation : print(f'{str(conversation)}\n\nINPUT:\n{str(instance["original passage"])}\nOUTPUT:\n{str(output)}\n')),
            # # # # # gold_key = (lambda instance : instance['PassageEditID'] == 1),
            report_generator = generate_report
        )


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
        choices=["flores"],
        help="file path to the data to run the workflow on"
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

    args = parser.parse_args()
    key = args.key
    if key is not None:
        del args.key
    main(args, key)
