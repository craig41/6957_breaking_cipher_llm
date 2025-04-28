import os
import argparse
os.environ['HF_HOME'] = "/scratch/general/vast/u1380656/huggingface_cache"

from huggingface_hub import login

HUGGINGFACE_TOKEN = "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"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, default_data_collator, get_linear_schedule_with_warmup
import torch
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
def main():
    parser = argparse.ArgumentParser(description="Train a LoRA model for translation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    args = parser.parse_args()

    login(token=HUGGINGFACE_TOKEN)

    base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    use_4bit = True
    compute_dtype = torch.bfloat16
    use_4bit = True             # Enable 4-bit quantization
    bnb_4bit_quant_type = "nf4" # Quantization type
    bnb_4bit_use_double_quant = False
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto", # Automatically distribute model across available devices
        trust_remote_code=True, # Necessary for some models
        # token = "hf_YOUR_TOKEN_HERE" # Add if you face auth issues
    )
    model.config.use_cache = False # Important for training
    model.config.pretraining_tp = 1
    
    train_dataset = load_dataset("csv", data_files={"train": os.getcwd() + "/../data/finetune-data-full/combined_train_data.csv"})

    train_dataset = train_dataset["train"]
    # Drop None values
    train_dataset = train_dataset.filter(lambda x: x["input"] is not None and x["gold"] is not None)
    print(train_dataset)
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True, # Necessary for some models
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in train_dataset["gold"]])
    print(f"Target max length: {target_max_length}")

    input_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in train_dataset["input"]])
    print(f"Input max length: {input_max_length}")
    train_dataset = train_dataset.filter(lambda x: len(tokenizer(x["input"])["input_ids"]) < 100 and len(tokenizer(x["gold"])["input_ids"]) < 100)

    print(f"Filtered dataset size: {len(train_dataset)}")
    # Switch from prompt tuning to LoRA for better results
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,  # Rank of the update matrices
        lora_alpha=32,  # Parameter for scaling
        lora_dropout=0.1,  # Dropout probability for LoRA layers
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Store the system prompt for later use in inference
    prompt_tuning_init_text = "You are a translator that converts encoded or foreign text into plain English. When given input text, translate it accurately to English."
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def preprocess_function(examples, text_column="input", label_column="gold"):
        batch_size = len(examples[text_column])
        
        # Format conversations properly for instruction tuning
        conversations = []
        for i in range(batch_size):
            source_text = examples[text_column][i]
            target_text = examples[label_column][i]
            
            # Create proper conversation format
            conversation = [
                {"role": "system", "content": prompt_tuning_init_text},
                {"role": "user", "content": f"Translate this text to English: {source_text}"},
                {"role": "assistant", "content": target_text}
            ]
            conversations.append(conversation)
        
        # Print an example to debug
        if batch_size > 0:
            print("\nExample conversation:")
            for message in conversations[0]:
                print(f"{message['role']}: {message['content']}")
                
        # Process using the model's chat template
        tokenized_inputs = []
        for conversation in conversations:
            tokenized_input = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False
            )
            tokenized_inputs.append(tokenized_input)
        
        # Prepare the inputs and labels for training
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        max_length = 512  # Set a maximum length for the input sequences
        for tokenized_input in tokenized_inputs:
            # Extract input_ids and create attention mask
            input_ids = tokenized_input[:max_length]
            attention_mask = torch.ones_like(input_ids)
            # Find where the assistant's response starts
            # This is model-specific and might need adjustment
            text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
            assistant_tokens = tokenizer("assistant\n\n", add_special_tokens=False)["input_ids"]
            # Find the position of the assistant token sequence
            labels = input_ids.clone()[0]
            
            # Set labels for non-assistant tokens to -100 (ignored in loss)
            # This makes the model only learn to predict the assistant's response
            assistant_pattern = torch.tensor([78191, 128007,271], dtype=torch.long)
            # Find all occurrences of the pattern
            occurrences = []
            for i in range(len(labels) - len(assistant_pattern) + 1):
                if torch.all(labels[i:i+len(assistant_pattern)] == assistant_pattern):
                    occurrences.append(i + len(assistant_pattern))  # Start after "assistant\n\n"
            
            # Use the last occurrence if any are found
            if occurrences:
                # Set all tokens before the assistant's response to -100
                labels[:occurrences[-1]] = -100
            else:
                # If pattern not found, don't train on this example
                print("No assistant token found, ignoring this example.")
                labels[:] = -100
            
            # Add to model inputs
            model_inputs["input_ids"].append(input_ids[0])
            model_inputs["attention_mask"].append(attention_mask[0])
            model_inputs["labels"].append(labels)
        
        # Convert lists to tensors
        for key in model_inputs:
            if model_inputs[key]:
                # Pad sequences to the same length
                max_len = max(len(x) for x in model_inputs[key])
                padded_inputs = []
                
                for x in model_inputs[key]:
                    if len(x) < max_len:
                        if key == "labels":
                            # Pad with -100 for labels
                            padding = torch.full((max_len - len(x),), -100, dtype=x.dtype)
                        else:
                            # Pad with 0 for input_ids and attention_mask
                            padding = torch.zeros(max_len - len(x), dtype=x.dtype)
                        x = torch.cat([x, padding])
                    if len(x.shape) == 1:
                        padded_inputs.append(x)
                
                model_inputs[key] = torch.stack(padded_inputs)
        
        return model_inputs
    
    processed_ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
        batch_size=7988
    )
    print(processed_ds[0])

    batch_size = 16
    
    train_dataloader = DataLoader(processed_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    
    lr = 2e-4  # Lower learning rate for LoRA
    num_epochs = args.epochs  # Use epochs from command line argument
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * (len(train_dataloader) * num_epochs)),  # 10% warmup
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    device = "cuda"
    model = model.to(device)
    
    def evaluate_model(epoch=None):
        """Evaluate the model on test examples and print results"""
        model.eval()
        import re
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Try to import NLTK, install if not available
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
        except (ImportError, LookupError):
            import subprocess
            subprocess.run(["pip", "install", "nltk"])
            import nltk
            nltk.download('punkt')
        
        print("\n===== Testing the model" + (f" after epoch {epoch}" if epoch is not None else "") + " =====")
        
        # Test on several examples
        num_samples = min(5, len(test_dataset))  # Test on first 5 examples or fewer if dataset is smaller
        
        # Track metrics
        bleu_scores = []
        exact_matches = 0
        
        for i in range(num_samples):
            # Format input properly as a conversation - EXACTLY as we did in training
            messages = [
                {"role": "system", "content": prompt_tuning_init_text},
                {"role": "user", "content": f"Translate this text to English: {test_dataset[i]['input']}"}
            ]
            
            print(f"\nExample {i+1}:")
            print(f"Input: {test_dataset[i]['input']}")
            print(f"Expected: {test_dataset[i]['gold']}")
            
            # Process using chat template
            model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
            
            # Generate with appropriate parameters
            with torch.no_grad():
                outputs = model.generate(
                    model_inputs,
                    max_new_tokens=target_max_length,
                    do_sample=False,
                    temperature=0.1,  # Low temperature for more deterministic outputs
                    top_p=0.95
                )
                response = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
                
                # Extract just the model's response part
                final_response = ""
                if "assistant" in response:
                    # Find the last assistant response
                    pattern = "assistant\n\n"
                    response_splits = re.split(pattern, response)
                    if len(response_splits) > 1:
                        final_response = response_splits[-1].strip()
                        print(f"Model output: {final_response}")
                    else:
                        final_response = response.strip()
                        print(f"Raw response: {response}")
                else:
                    final_response = response.strip()
                    print(f"Raw response: {response}")
                
                # Calculate BLEU score
                reference = test_dataset[i]['gold'].lower().split()
                hypothesis = final_response.lower().split()
                
                try:
                    # Use smoothing for short sentences
                    smoothie = SmoothingFunction().method1
                    bleu = sentence_bleu([reference], hypothesis, smoothing_function=smoothie)
                    bleu_scores.append(bleu)
                    print(f"BLEU score: {bleu:.4f}")
                except Exception as e:
                    print(f"Error calculating BLEU: {e}")
                
                # Check for exact match
                if final_response.lower() == test_dataset[i]['gold'].lower():
                    exact_matches += 1
                    print("✓ Exact match!")
        
        # Print summary metrics
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        exact_match_pct = exact_matches / num_samples * 100 if num_samples > 0 else 0
        
        print("\n----- Evaluation Summary -----")
        print(f"Average BLEU score: {avg_bleu:.4f}")
        print(f"Exact match rate: {exact_match_pct:.2f}% ({exact_matches}/{num_samples})")
        print("===== End of test =====")
        
        # Return to training mode
        model.train()
        
        # Return metrics for tracking
        return {
            "epoch": epoch,
            "avg_bleu": avg_bleu,
            "exact_match_pct": exact_match_pct
        }
    
    # Load the test dataset once before training
    print("Loading test dataset...")
    test_dataset = load_dataset("csv", data_files={"test": os.getcwd() + "/../data/finetune-data-full/combined_test_data.csv"})
    test_dataset = test_dataset["test"]
    # Drop None values
    test_dataset = test_dataset.filter(lambda x: x["input"] is not None and x["gold"] is not None)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create a list to store all metrics during training
    metrics_history = []
    
    try:
        # Try to import tensorboard for metric visualization
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_available = True
        # Create tensorboard writer
        log_dir = os.getcwd() + "/results_llama3-8b-instruct-translator/runs/"
        writer = SummaryWriter(log_dir=log_dir)
        print(f"Tensorboard logs will be saved to {log_dir}")
    except ImportError:
        tensorboard_available = False
        print("Tensorboard not available. Install it with 'pip install tensorboard' for metric visualization.")
    
    # Evaluate once before training starts
    print("\nEvaluating model before training (epoch 0):")
    initial_metrics = evaluate_model(epoch=0)
    metrics_history.append(initial_metrics)
    
    # Log initial metrics to tensorboard
    if tensorboard_available:
        writer.add_scalar('Loss/train', 0, 0)  # No loss for initial model
        writer.add_scalar('BLEU/test', initial_metrics['avg_bleu'], 0)
        writer.add_scalar('ExactMatch/test', initial_metrics['exact_match_pct'], 0)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
        
        # Evaluate on test data after each epoch
        epoch_metrics = evaluate_model(epoch=epoch+1)  # +1 because we already evaluated at epoch 0
        metrics_history.append(epoch_metrics)
        
        # Log metrics to tensorboard
        if tensorboard_available:
            writer.add_scalar('Loss/train', train_epoch_loss.item(), epoch+1)
            writer.add_scalar('Perplexity/train', train_ppl.item(), epoch+1)
            writer.add_scalar('BLEU/test', epoch_metrics['avg_bleu'], epoch+1)
            writer.add_scalar('ExactMatch/test', epoch_metrics['exact_match_pct'], epoch+1)
        
        # Save checkpoint after each epoch if desired
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:  # Save every 5 epochs or at the final epoch
            checkpoint_dir = os.getcwd() + f"/results_llama3-8b-instruct-translator/checkpoint_epoch_{epoch+1}/"
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_dir}")
    
    # --- Save Adapter ---
    output_dir = os.getcwd() + "/results_llama3-8b-instruct-translator/"
    print(f"Saving final LoRA adapter to {output_dir}...")
    final_adapter_path = output_dir + f"/cipher_adapter_epochs_{num_epochs}/"
    os.makedirs(final_adapter_path, exist_ok=True)
    model.save_pretrained(final_adapter_path)
    
    # Save metrics history to CSV
    try:
        import pandas as pd
        metrics_df = pd.DataFrame(metrics_history)
        metrics_file = output_dir + "/training_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Training metrics saved to {metrics_file}")
    except ImportError:
        print("pandas not available, metrics history not saved to CSV")
    
    # Close tensorboard writer if it was created
    if tensorboard_available:
        writer.close()
    
    # Print a summary of improvement
    if len(metrics_history) > 1:
        initial_bleu = metrics_history[0]['avg_bleu']
        final_bleu = metrics_history[-1]['avg_bleu']
        bleu_improvement = final_bleu - initial_bleu
        
        initial_exact = metrics_history[0]['exact_match_pct']
        final_exact = metrics_history[-1]['exact_match_pct']
        exact_improvement = final_exact - initial_exact
        
        print("\n===== Training Summary =====")
        print(f"BLEU score: {initial_bleu:.4f} → {final_bleu:.4f} ({bleu_improvement:+.4f})")
        print(f"Exact match: {initial_exact:.2f}% → {final_exact:.2f}% ({exact_improvement:+.2f}%)")
        print("=============================")

if __name__ == "__main__":
    main()