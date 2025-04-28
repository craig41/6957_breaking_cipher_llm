import os, sys ; sys.path.append(os.getcwd()) #allow local imports
os.environ["TRANSFORMERS_CACHE"] = "/scratch/general/vast/u0013114/huggingface_cache"

import argparse
import torch
import csv
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import *
from models import *
from workflows import *
from huggingface_hub import login

def generate_report(instances, golds, outputs, conversations):
    pass

translated = []
original = []
def callback(instance, gold, output, conversation):
    translated.append(output)
    original.append(instance['text'])
    print(f"Original: {instance['text']}")
    print(f"Translated: {output}")
    print(f"Gold: {gold}")
    print("===")

def load_csv_data(csv_path, n=None, start=0):
    """
    Load data from a CSV file into a Dataset format compatible with the existing pipeline.
    
    Args:
        csv_path: Path to the CSV file
        n: Number of examples to load (None for all)
        start: Starting index
        
    Returns:
        A HuggingFace Dataset object
    """
    print(f"Loading data from {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Handle selection range
    if n is not None:
        df = df.iloc[start:start+n]
    else:
        df = df.iloc[start:]
    
    # Create a dataset-compatible format
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append({
            "text": row['input'],
            "glottocode": "custom", # We don't need real glottocodes for test data
            "gold": row['gold'],
            "category": row.get('category', 'unknown')
        })
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    print(f"Loaded {len(dataset)} examples from {csv_path}")
    
    return dataset

def _llama3_1_8b_instruct_lora(lora_adapter_path=None):
    """Load Llama 3.1 8B model with optional LoRA adapter"""
    # Base model ID
    base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the base model
    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Apply LoRA adapter if provided
    if lora_adapter_path is not None and os.path.exists(lora_adapter_path):
        print(f"Loading LoRA adapter from: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        # Option to merge weights for faster inference
        # model = model.merge_and_unload()
        print("LoRA adapter loaded successfully")
    else:
        print("No LoRA adapter loaded or path doesn't exist")
    
    # Set model configuration
    config = model.config
    if hasattr(config, 'rope_scaling'):
        config.rope_scaling = {
            "type": "llama3",
            "factor": 8.0
        }
    
    def query(messages, n=None):
        import re
        # Process messages through the chat template
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
        
        # Generate response
        attention_mask = model_inputs.ne(tokenizer.pad_token_id)
        outputs = model.generate(
            model_inputs,
            attention_mask=attention_mask,
            max_new_tokens=512, 
            do_sample=False, 
            temperature=None, 
            top_p=None,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Decode and clean up response
        raw_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Clean the response to remove unwanted tokens
        cleaned_response = re.sub(r'<\|eot_id\|>', '', raw_response)
        
        # Pattern to detect each occurrence of the assistant's response
        pattern = "assistant\n\n"
        pattern2 = "assistant: "
        pattern3= "assistant\n"
        pattern4 = "assistant "
        pattern5 = "assistant"
        
        # Split the response based on the pattern and grab the last split
        if pattern in cleaned_response:

            response_splits = re.split(pattern, cleaned_response)
        elif pattern2 in cleaned_response:
            response_splits = re.split(pattern2, cleaned_response)
        elif pattern3 in cleaned_response:
            response_splits = re.split(pattern3, cleaned_response)
        elif pattern4 in cleaned_response:
            response_splits = re.split(pattern4, cleaned_response)
        elif pattern5 in cleaned_response:
            response_splits = re.split(pattern5, cleaned_response)
        else:
            print("No pattern found in response")
            response_splits = [cleaned_response]
        
        # The last item should contain the most recent assistant response
        if len(response_splits) > 1:
            last_response = response_splits[-1].split('</s>')[0].strip()
        else:
            last_response = cleaned_response.strip()
        
        return last_response
    
    return query

def main(args, key):
    print(args)
    if key is not None:
        args.model_args = key
    
    # Load data based on source type
    if args.csv_data:
        # Load from CSV file
        data = load_csv_data(args.csv_data, args.n, args.s)
    else:
        # Load from HuggingFace dataset
        if args.n is None:
            data = read_data(args.data, split=args.split)
            data = data.filter(lambda x: x['glottocode'] == args.source)
            data = data.select(range(args.s, len(data)))
        else:
            data = read_data(args.data, split=args.split)
            data = data.filter(lambda x: x['glottocode'] == args.source)
            data = data.select(range(args.s, args.n + args.s))
    
    # Choose the model based on arguments
    if args.model == "llama3_1_8b_instruct_lora":
        model_fn = _llama3_1_8b_instruct_lora(args.lora_path)
    else:
        # Use the standard model loading for other models
        model_fn = load_model(args.model)
    
    # Run the pipeline
    run(
        data,
        pipeline = Pipeline(
            model_fn,
            load_workflow(args.workflow)
        ), 
        per_instance_callback=callback,
        report_generator=generate_report
    )
    
    # Save output files
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
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter to load onto the base model"
    )
    parser.add_argument(
        "--csv-data",
        type=str,
        default=None,
        help="Path to CSV file containing test data (overrides -d argument)"
    )

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    key = args.key
    if key is not None:
        del args.key
    login(token = key)
    main(args, key)