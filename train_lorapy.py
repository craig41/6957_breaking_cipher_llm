import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DataCollatorForCompletionOnlyLM

import csv
os.environ['HF_HOME'] = "/scratch/general/vast/u0013114/huggingface_cache"

from huggingface_hub import login

HUGGINGFACE_TOKEN = "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"
# Login to Hugging Face Hub
login(token=HUGGINGFACE_TOKEN)

# --- Configuration ---

# Model
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
new_adapter_name = "llama3-8b-instruct-translator" # Choose a name for your adapter

# LoRA Config
lora_r = 16             # LoRA rank (dimension)
lora_alpha = 32         # LoRA alpha (scaling factor)
lora_dropout = 0.05     # LoRA dropout
# Check model architecture print(model) to find target modules
# Common Llama modules: "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
lora_target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training Args
output_dir = f"./results_{new_adapter_name}" # Directory to save adapter & logs
num_train_epochs = 1        # Number of training epochs (adjust as needed)
# Adjust batch size and gradient accumulation based on your GPU memory
per_device_train_batch_size = 2  # Reduce if OOM
gradient_accumulation_steps = 4  # Increase effective batch size
gradient_checkpointing = True   # Saves memory
optim = "paged_adamw_8bit"  # Optimizer for 4-bit models
learning_rate = 2e-5        # Learning rate (adjust as needed)
lr_scheduler_type = "cosine" # Learning rate scheduler
max_grad_norm = 0.3         # Gradient clipping
warmup_ratio = 0.03         # Warmup ratio
logging_steps = 25          # Log training progress every N steps
save_strategy = "epoch"     # Save checkpoints at the end of each epoch
# max_steps = -1              # Set to > 0 to limit training steps instead of epochs
bf16 = True                 # Use bfloat16 precision (requires Ampere GPU or newer)
# fp16 = False              # Use fp16 if bf16 is not available (set bf16=False)

# Quantization Config (for 4-bit loading)
use_4bit = True             # Enable 4-bit quantization
bnb_4bit_quant_type = "nf4" # Quantization type
bnb_4bit_compute_dtype = torch.bfloat16 # Compute dtype for 4-bit base models
bnb_4bit_use_double_quant = False # Use double quantization

# --- Model & Tokenizer Loading ---

print(f"Loading base model: {base_model_id}")
use_4bit = True
bf16 = True  # Or False if using fp16
compute_dtype = torch.bfloat16

bnb_config = None
if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
        # Consider setting bf16=True and using torch.bfloat16 if available

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto", # Automatically distribute model across available devices
    trust_remote_code=True, # Necessary for some models
    # token = "hf_YOUR_TOKEN_HERE" # Add if you face auth issues
)
model.config.use_cache = False # Important for training
model.config.pretraining_tp = 1 # Set to 1 for fine-tuning

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
# Llama 3 specific settings. Might need adjustment based on tokenizer version.
tokenizer.pad_token = tokenizer.eos_token # Llama does not have a pad token by default
tokenizer.padding_side = "right" # Standard for fine-tuning

# --- Load Dataset ---
with open(os.getcwd() + "/data/finetune-data/combined_train_data.csv", "r") as f:
    reader = csv.reader(f)
    data = list(reader)

print(data[1])

# Convert to dataset
train_data = []
for row in data[1:]:
    # Skip empty rows
    if len(row) < 3:
        continue
    train_data.append({"input": row[0], "gold": row[1], "type": row[2]})

train_dataset = load_dataset("csv", data_files={"train": os.getcwd() + "/../data/finetune-data/train_data.csv"})

train_dataset = train_dataset["train"]
# Drop None values
train_dataset = train_dataset.filter(lambda x: x["input"] is not None and x["gold"] is not None)
print(train_dataset)

# --- Preprocessing Function ---
# We need to format the data into a prompt that the instruct model understands.
# Llama 3 Instruct uses a specific chat format. We'll adapt it for translation.

def create_translation_prompt(example):
    """Creates a formatted prompt for translation."""
    source_text = example["input"]
    target_text = example["gold"]

    if source_text is None or target_text is None:
        raise ValueError("Source or target text is None.")

    # Using a simplified instruction format. You might experiment with Llama 3's chat template.
    # Ref: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    # Note: SFTTrainer handles chat templating automatically if the dataset has 'messages'
    # For simpler datasets, manual formatting like below works.
    prompt_string = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Translate the following text into plain English:
    
    English: {source_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    full_string = prompt_string + target_text + tokenizer.eos_token
    # Tokenize the full string
    tokenized_full = tokenizer(full_string, truncation=True, max_length=512, padding=False) # Don't pad here, collator will handle it

    # Tokenize the prompt part ONLY to find its length
    tokenized_prompt = tokenizer(prompt_string, truncation=True, max_length=512, padding=False)
    prompt_len = len(tokenized_prompt['input_ids'])

    # Create labels: -100 for prompt, actual tokens for target
    labels = list(tokenized_full['input_ids'])
    labels[:prompt_len] = [-100] * prompt_len
    return {
        "input_ids": tokenized_full['input_ids'],
        "attention_mask": tokenized_full['attention_mask'],
        "labels": labels,
    } # SFTTrainer expects a 'text' column

# Apply the formatting
print("Formatting dataset...")
tokenized_dataset = train_dataset.map(create_translation_prompt, remove_columns=train_dataset.column_names)

# Check the first example
print(tokenized_dataset[0])

# --- PEFT Configuration ---

print("Setting up PEFT/LoRA...")
# Prepare model for k-bit training if using quantization
if use_4bit:
    print("Preparing model for K-bit training...")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing # Pass the GC flag
    )
    print("Model prepared for K-bit training.")

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=lora_target_modules,
)

model = get_peft_model(model, peft_config)
print("PEFT model created.")
model.print_trainable_parameters()

# --- Training Setup ---

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Crucial for Causal LM, ensures labels are not shifted
)
print("Setting up Training Arguments...")
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    optim=optim,
    logging_steps=logging_steps,
    save_strategy=save_strategy,
    learning_rate=learning_rate,
    # bf16=bf16, # SFTTrainer uses Trainer's args
    # fp16=fp16, # SFTTrainer uses Trainer's args
    max_grad_norm=max_grad_norm,
    # max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True, # Speeds up training by batching similar lengths
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard", # or "wandb" if installed and configured
    push_to_hub=False, # Set to True to push adapter to Hub
    hub_model_id=new_adapter_name if False else None, # Only if push_to_hub=True
)

print("Initializing Trainer...")
trainer = Trainer(
    model=model, # Pass the PEFT-enhanced model
    args=training_arguments,
    train_dataset=tokenized_dataset,
    # eval_dataset=None, # Add eval dataset here if you have one
    tokenizer=tokenizer, # Pass tokenizer for padding/saving
    data_collator=data_collator, # Use the completion-only collator
)

# --- Start Training ---

print("Starting training...")
trainer.train()

# --- Save Adapter ---

print(f"Saving LoRA adapter to {output_dir}...")
# SFTTrainer automatically saves the adapter during training based on save_strategy
# You can also explicitly save the final adapter
final_adapter_path = os.path.join(output_dir, "final_adapter")
trainer.save_model(final_adapter_path) # Saves adapter config & weights
print(f"Adapter saved to {final_adapter_path}")

# --- (Optional) Inference Example ---
print("\n--- Example Inference ---")
print("Loading base model and merging adapter for inference...")

# Load the base model again (can be the same 4-bit config or full precision)
base_model_for_inference = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config, # Use same quantization if needed
    device_map="auto",
    trust_remote_code=True,
    # token = "hf_YOUR_TOKEN_HERE"
)

# Load the PEFT model (adapter) on top of the base model
# Use the path where the adapter was saved by the trainer (usually output_dir or final_adapter_path)
# SFTTrainer saves adapters inside the checkpoint folders (e.g., output_dir/checkpoint-XXX)
# and potentially a final adapter if you call save_model explicitly.
# Check your output_dir for the exact path. Let's assume the last checkpoint or the explicit save.
adapter_path = final_adapter_path # Or check trainer.state.best_model_checkpoint
if not os.path.exists(adapter_path):
    # Fallback to the main output dir if final_adapter wasn't created explicitly
     adapter_path = output_dir
     print(f"Final adapter path not found, trying main output directory: {adapter_path}")

# Make sure the adapter path exists before loading
if os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
    model_inf = PeftModel.from_pretrained(base_model_for_inference, adapter_path)
    model_inf = model_inf.merge_and_unload() # Merge adapter into the base model for faster inference

    print("Model merged. Ready for inference.")

    # Use the same tokenizer
    tokenizer_inf = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer_inf.pad_token = tokenizer_inf.eos_token
    tokenizer_inf.padding_side = "right"

    # Create the inference prompt (ONLY the user part)
    source_text_example = "Hello, how are you today?"
    inference_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Translate the following English text to French:
English: {source_text_example}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

French: """ # Note the prompt stops here, expecting the model to generate the rest

    # Setup pipeline
    translator = pipeline(
        "text-generation",
        model=model_inf,
        tokenizer=tokenizer_inf,
        torch_dtype=compute_dtype, # Use the compute dtype
        device_map="auto"
    )

    # Generate translation
    print(f"\nTranslating: '{source_text_example}'")
    result = translator(
        inference_prompt,
        max_new_tokens=50,          # Max length of the generated translation
        do_sample=False,            # Use greedy decoding for more deterministic output
        # temperature=0.7,          # Adjust temperature/top_k/top_p for sampling
        # top_k=50,
        # top_p=0.9,
        eos_token_id=tokenizer_inf.eos_token_id,
        pad_token_id=tokenizer_inf.pad_token_id # Important for generation
    )

    # Extract generated text (might need adjustment based on pipeline output format)
    generated_text = result[0]['generated_text']
    # Extract only the generated French part after the prompt
    french_translation = generated_text.split("French: ")[-1].replace("<|eot_id|>", "").strip()

    print(f"Generated Translation: {french_translation}")

else:
    print(f"Could not find adapter config at {adapter_path}/adapter_config.json. Skipping inference example.")


print("\nFine-tuning complete.")