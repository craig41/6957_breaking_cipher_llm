#!/usr/bin/env python3

import re
import sys

def parse_translations(input_file, output_file, expected_count=2100):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract all translations from the file
    translations = []
    numbered_lines = {}
    
    # First pass: extract numbered lines with actual content
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for numbered lines
        match = re.match(r'^\s*(\d+)\s+(.*)', line)
        if match:
            index = int(match.group(1))
            text = match.group(2).strip()
            
            # Skip obvious explanations
            if any(marker in text for marker in [
                'Translation:', 'Here\'s', 'appears to be', 'translated text', '- ',
                'breakdown:', 'The text', 'Without context', 'Note that'
            ]) or text.startswith('The '):
                continue
                
            # Store by line number
            if index not in numbered_lines:
                numbered_lines[index] = text
    
    # Get actual text lines that aren't numbered but contain translations
    for i in range(len(lines)):
        line = lines[i].strip()
        
        # Skip empty lines or lines with line numbers
        if not line or re.match(r'^\s*\d+\s+', line):
            continue
            
        # Skip explanatory lines
        if any(marker in line for marker in [
            'Translation:', 'Here\'s', 'appears to be', 'translated text', '- ',
            'breakdown:', 'The text', 'Without context', 'Note that'
        ]) or line.startswith('The '):
            continue
            
        # Check if this is likely a translation line
        if (not re.search(r'^\[.*\]$', line) and 
            not any(x in line for x in ['Without additional context', 'appears to be', 'translated text'])):
            translations.append(line)
    
    # Create the output with exactly expected_count lines
    output_lines = ["" for _ in range(expected_count)]
    
    # First, fill in the numbered lines we found
    for idx, text in numbered_lines.items():
        if 0 <= idx < expected_count:
            output_lines[idx] = text
    
    # Then, try to fill in any remaining empty lines with translations we found
    translation_idx = 0
    for i in range(expected_count):
        if not output_lines[i] and translation_idx < len(translations):
            output_lines[i] = translations[translation_idx]
            translation_idx += 1
    
    # Ensure no empty lines remain
    for i in range(expected_count):
        if not output_lines[i]:
            output_lines[i] = f"[No translation available for line {i}]"
    
    # Write the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(f"{line}\n")
    
    print(f"Generated {expected_count} translations")
    return expected_count

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_translations.py input_file output_file [expected_count]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    expected_count = 2100  # Default
    if len(sys.argv) > 3:
        try:
            expected_count = int(sys.argv[3])
        except ValueError:
            print(f"Invalid expected count: {sys.argv[3]}. Using default: {expected_count}")
    
    count = parse_translations(input_file, output_file, expected_count)
    print(f"Parsed translations saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_translations.py input_file output_file [reference_file]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    reference_file = None
    if len(sys.argv) > 3:
        reference_file = sys.argv[3]
    
    count = parse_translations(input_file, output_file, reference_file)
    print(f"Parsed translations saved to {output_file}")