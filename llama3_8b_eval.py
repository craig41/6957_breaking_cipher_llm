import argparse

import transformers
import torch
from transformers import pipeline

from constants import Constants

access_token = Constants.HUGGINGFACE_TOKEN

pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", token=access_token)
# def text


if __name__ == "__main__":
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # messages = [
    #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    #     {"role": "user", "content": "Who are you?"},
    # ]

    messages_2 = [
        {"role": "user", "content": "can you translate the encoded word in this text: upon a time mvrfrEwmesd3X+EnXnrQ+ttn915LJCqR8N7/w5fVbc8= a quiet village nestled among the"},
        {"role": "user", "content": "can you translate the encoded word in this text: was also wise she /etQ+2yiHLq7S2vhewCnkEOadXSFOW0qWtLqhilq1mg= the power of wishes could"},
    ]

    outputs = pipe(
        messages_2,
        max_new_tokens=256,
    )

    for o in outputs:
        print(o["generated_text"][-1])