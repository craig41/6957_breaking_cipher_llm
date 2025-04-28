#!/usr/bin/env python3
import csv
import os

def main():
    # Define paths
    input_file = "/scratch/general/vast/u1380656/6957_breaking_cipher_llm/data/finetune-data-full/combined_test_data.csv"
    output_dir = "/scratch/general/vast/u1380656/6957_breaking_cipher_llm/data/gold"
    original_file = os.path.join(output_dir, "original.txt")
    translated_file = os.path.join(output_dir, "translated.txt")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file and extract data
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Write original and gold (translated) texts to separate files
    with open(original_file, 'w', encoding='utf-8') as f_orig, \
         open(translated_file, 'w', encoding='utf-8') as f_trans:
        for row in data:
            # Extract input and gold translation
            input_text = row['input']
            gold_text = row['gold']
            
            # Write to respective files
            f_orig.write(f"{input_text}\n")
            f_trans.write(f"{gold_text}\n")
    
    print(f"Extracted {len(data)} test examples to {original_file} and {translated_file}")

if __name__ == "__main__":
    main()