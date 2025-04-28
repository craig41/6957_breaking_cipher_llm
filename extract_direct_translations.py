#!/usr/bin/env python3

import re
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

# Import for batching
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class TranslationBatch:
    """Class for keeping track of a batch of translations to process"""
    indices: List[int]
    originals: List[str]
    translated_blocks: List[str]
    clean_translations: List[Optional[str]] = None
    
    def __post_init__(self):
        self.clean_translations = [None] * len(self.indices)
        
    def __len__(self):
        return len(self.indices)

def extract_translation_blocks(input_file):
    """Extract original text and translation blocks from the model output file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by separator and extract the translation blocks
    blocks = content.split("===")
    translation_pairs = []
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        original_match = re.search(r'Original: (.*?)(?:\n|$)', block)
        translated_match = re.search(r'Translated:(.*?)(?=Gold:|$)', block, re.DOTALL)
        
        if original_match and translated_match:
            original_text = original_match.group(1).strip()
            translated_block = translated_match.group(1).strip()
            translation_pairs.append((original_text, translated_block))
    
    return translation_pairs

class TranslationCleaner:
    """Class to handle loading the model once and processing many translations"""
    
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.prompt_template = """
You are a helpful translation assistant. Below is a block of text that contains a translation of some content, but it might include explanations, analysis, or extra text that is not part of the actual translation.

Your task is to identify and extract ONLY the translation itself - the actual translated content, without any explanations, notes, or commentary.

Here is the translation block:
```
{translated_text}
```

Respond with ONLY the clean translation text itself, with no other explanation, prefix, or suffix. Your response should be a single line containing only the translation. If there are multiple possible translations, choose the most complete one.
"""
        self._init_model()
        
    def _init_model(self):
        """Initialize the model and tokenizer"""
        print(f"Loading model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        print("Model loaded successfully")
    
    def clean_translation_block(self, translated_block):
        """Clean a single translation block by running it through the model"""
        # Remove any very long lines that might confuse the model
        lines = translated_block.split('\n')
        filtered_lines = []
        for line in lines:
            if len(line) < 1000:  # Skip extremely long lines
                filtered_lines.append(line)
        
        # Limit total size
        filtered_text = '\n'.join(filtered_lines)
        if len(filtered_text) > 3500:
            filtered_text = filtered_text[:3500] + "..."
        
        # Format prompt for the model
        prompt = self.prompt_template.format(translated_text=filtered_text)
        
        # Create chat messages
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Process through the chat template
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=256,
                temperature=None,  # Use greedy decoding for most reliable extraction
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode response, removing the input prompt
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Look for the actual translation (first significant line)
        for line in response.split('\n'):
            line = line.strip()
            if line and len(line) > 4:
                # Skip typical headers and explanations
                if any(x in line.lower() for x in ["translation:", "here is", "translated:"]):
                    continue
                return line
        
        # Fallback to returning whole response
        return response
    
    def process_batch(self, batch: TranslationBatch) -> TranslationBatch:
        """Process a batch of translations sequentially (limited by GPU memory)"""
        for i, translated_block in enumerate(batch.translated_blocks):
            batch.clean_translations[i] = self.clean_translation_block(translated_block)
        return batch

def create_batches(translation_pairs, batch_size=4):
    """Convert translation pairs into batches"""
    batches = []
    
    for i in range(0, len(translation_pairs), batch_size):
        chunk = translation_pairs[i:i+batch_size]
        indices = list(range(i, i+len(chunk)))
        originals = [pair[0] for pair in chunk]
        translated_blocks = [pair[1] for pair in chunk]
        
        batch = TranslationBatch(
            indices=indices,
            originals=originals,
            translated_blocks=translated_blocks
        )
        batches.append(batch)
    
    return batches

def main(input_file, output_file, gold_file=None, batch_size=4, num_workers=1):
    # Extract translation blocks from the input file
    translation_pairs = extract_translation_blocks(input_file)
    print(f"Extracted {len(translation_pairs)} translation pairs")
    
    # Create a list to store the final translations
    # If we have a gold file, load it to determine the expected line count
    expected_count = 2100  # Default
    if gold_file:
        try:
            with open(gold_file, 'r', encoding='utf-8') as f:
                gold_lines = f.readlines()
                expected_count = len(gold_lines)
                print(f"Will generate {expected_count} translations based on gold file")
        except Exception as e:
            print(f"Error reading gold file: {e}")
    
    # Create the translation cleaner
    cleaner = TranslationCleaner()
    
    # Create batches of translations
    batches = create_batches(translation_pairs, batch_size)
    print(f"Created {len(batches)} batches with size {batch_size}")
    
    # Process all batches
    clean_translations = [None] * len(translation_pairs)
    
    # Process batches with tqdm for progress tracking
    for batch in tqdm(batches, desc="Processing batches"):
        processed_batch = cleaner.process_batch(batch)
        
        # Store the results
        for idx, trans in zip(processed_batch.indices, processed_batch.clean_translations):
            if idx < len(clean_translations):
                clean_translations[idx] = trans
    
    # Make sure we have exactly the expected number of translations
    translations = clean_translations
    
    if len(translations) < expected_count:
        # Fill in missing translations
        translations.extend([f"[Missing translation {i}]" for i in range(len(translations), expected_count)])
    elif len(translations) > expected_count:
        # Trim extra translations
        translations = translations[:expected_count]
    
    # Write the clean translations to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for translation in translations:
            if translation is None:
                translation = "[Processing error]"
            f.write(f"{translation}\n")
    
    print(f"Saved {len(translations)} clean translations to {output_file}")
    
    # Return statistics
    success_count = sum(1 for t in translations if t is not None and not t.startswith("["))
    print(f"Successfully processed {success_count}/{len(translations)} translations")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_direct_translations.py input_file output_file [gold_file] [batch_size]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    gold_file = None
    if len(sys.argv) > 3:
        gold_file = sys.argv[3]
    
    batch_size = 4  # Default batch size
    if len(sys.argv) > 4:
        try:
            batch_size = int(sys.argv[4])
        except ValueError:
            print(f"Invalid batch size: {sys.argv[4]}. Using default: 4")
    
    main(input_file, output_file, gold_file, batch_size)