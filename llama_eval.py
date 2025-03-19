import transformers
import torch


from constants import Constants

model_id = "meta-llama/Meta-Llama-3-8B"

access_token = Constants.HUGGINGFACE_TOKEN

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)
pipeline("Hey how are you doing today?")