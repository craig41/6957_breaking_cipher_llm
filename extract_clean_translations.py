#!/usr/bin/env python3

import re
import sys
import subprocess
import json
import os
from concurrent.futures import ThreadPoolExecutor

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

def ask_llm_for_best_translation(translated_block, prompt_template):
    """Use llama or another local model to extract the best translation from the block."""
    # Remove any very long lines that might confuse the model
    lines = translated_block.split('\n')
    filtered_lines = []
    for line in lines:
        if len(line) < 1000:  # Skip extremely long lines
            filtered_lines.append(line)
    
    # Limit total size
    filtered_text = '\n'.join(filtered_lines)
    if len(filtered_text) > 4000:
        filtered_text = filtered_text[:4000] + "..."
    
    # Format prompt for the model
    prompt = prompt_template.format(translated_text=filtered_text)
    
    # Create a temporary file for this translation
    import tempfile
    tmp_input = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    tmp_output_dir = tempfile.mkdtemp()
    
    try:
        # Format as CSV
        tmp_input.write("input,gold\n")
        tmp_input.write(f"\"{prompt.replace('\"', '\"\"')}\",\"\"\n")
        tmp_input.close()
        
        # Command to run the local model
        run_local_path = "/scratch/general/vast/u1380656/6957_breaking_cipher_llm/src/run_local.py"
        base_cmd = [
            "python3", run_local_path,
            "--model", "llama3_1_8b_instruct",  # Use the Llama 3.1 model 
            "--workflow", "translate",
            "--csv-data", tmp_input.name,
            "--output", tmp_output_dir,
            "--n", "1"  # Process just one example
        ]
        
        # Run the command
        try:
            result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=120)
            
            # Check for successful execution
            if result.returncode != 0:
                print(f"Model execution failed with code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr[:500]}...")
                return "[Model error: execution failed]"
            
            # Read the output file
            translation_file = os.path.join(tmp_output_dir, "translated.txt")
            if not os.path.exists(translation_file):
                print(f"Output file {translation_file} not found")
                return "[Model error: output file not found]"
                
            with open(translation_file, 'r') as f:
                raw_output = f.read().strip()
            
            # Clean up the translation
            # First, split into lines and pick the one that's most likely to be a translation
            lines = raw_output.split('\n')
            
            # Find lines that contain translation content
            translation_lines = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 5:
                    # Skip lines that look like explanations
                    if not any(x in line.lower() for x in ['translation:', 'translated:', 'here is', 'the text']):
                        translation_lines.append(line)
            
            if translation_lines:
                return translation_lines[0]  # Return the first good line
            else:
                return raw_output if raw_output else "[No translation output]"
                
        except subprocess.TimeoutExpired:
            print("Model execution timed out")
            return "[Model timeout]"
            
    except Exception as e:
        print(f"Error running model: {e}")
        return f"[Error: {str(e)[:100]}]"
        
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_input.name):
            os.unlink(tmp_input.name)
        
        # Remove the temporary output directory
        import shutil
        shutil.rmtree(tmp_output_dir, ignore_errors=True)

def process_translation(idx, pair, prompt_template):
    """Process a single translation pair"""
    original_text, translated_block = pair
    print(f"Processing translation {idx+1}...")
    clean_translation = ask_llm_for_best_translation(translated_block, prompt_template)
    return (idx, original_text, clean_translation)

def main(input_file, output_file, gold_file=None, num_workers=1):
    # Define the prompt template
    prompt_template = """
You are a helpful translation assistant. Below is a block of text that contains a translation of some content, but it might include explanations, analysis, or extra text that is not part of the actual translation.

Your task is to identify and extract ONLY the translation itself - the actual translated content, without any explanations, notes, or commentary.

Here is the translation block:
```
{translated_text}
```

Respond with ONLY the clean translation text itself, with no other explanation, prefix, or suffix. Your response should be a single line containing only the translation. If there are multiple possible translations, choose the most complete one.
"""

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
    
    # Process translations in parallel if workers > 1
    clean_translations = []
    
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all translation jobs
            futures = [executor.submit(process_translation, i, pair, prompt_template) 
                      for i, pair in enumerate(translation_pairs)]
            
            # Collect results as they complete
            for future in futures:
                idx, original, translation = future.result()
                clean_translations.append((idx, original, translation))
    else:
        # Process sequentially
        for i, pair in enumerate(translation_pairs):
            result = process_translation(i, pair, prompt_template)
            clean_translations.append(result)
    
    # Sort by the original index to maintain order
    clean_translations.sort(key=lambda x: x[0])
    
    # Extract just the translation strings
    translations = [t[2] for t in clean_translations]
    
    # Make sure we have exactly the expected number of translations
    if len(translations) < expected_count:
        # Fill in missing translations
        translations.extend([f"[Missing translation {i}]" for i in range(len(translations), expected_count)])
    elif len(translations) > expected_count:
        # Trim extra translations
        translations = translations[:expected_count]
    
    # Write the clean translations to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(f"{translation}\n")
    
    print(f"Saved {len(translations)} clean translations to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_clean_translations.py input_file output_file [gold_file] [num_workers]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    gold_file = None
    if len(sys.argv) > 3:
        gold_file = sys.argv[3]
    
    num_workers = 1  # Default to sequential processing
    if len(sys.argv) > 4:
        try:
            num_workers = int(sys.argv[4])
        except ValueError:
            print(f"Invalid worker count: {sys.argv[4]}. Using default: 1")
    
    main(input_file, output_file, gold_file, num_workers)