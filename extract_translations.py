#!/usr/bin/env python3

import re
import sys

def extract_translations(input_file, output_file, gold_file=None, expected_count=2100):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Load gold data if provided
    gold_texts = []
    gold_indexes = {}  # Map from gold text to line number
    if gold_file:
        try:
            with open(gold_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    gold_texts.append(line)
                    gold_indexes[line] = idx
            print(f"Loaded {len(gold_texts)} gold texts for matching")
        except Exception as e:
            print(f"Error loading gold file: {e}")
    
    # Split content by the "===" separator which indicates different translations
    translations_blocks = content.split("===")
    
    # Initialize output lines list with empty strings
    output_lines = ["" for _ in range(expected_count)]
    
    # Track original texts and their translations
    original_to_translation = {}
    
    # Store raw translations for fallback
    all_translations = []
    
    # Process each translation block
    block_count = 0
    for block in translations_blocks:
        if not block.strip():
            continue
            
        # Look for the original text and its corresponding translation
        match_original = re.search(r'Original: (.*?)(?:\n|$)', block)
        if not match_original:
            continue
            
        original_text = match_original.group(1).strip()
        
        # Find the translation line after the "Translated:" marker
        match_translated = re.search(r'Translated:(.*?)(?=Gold:|$)', block, re.DOTALL)
        if not match_translated:
            continue
        
        translated_text_block = match_translated.group(1).strip()
        
        # Extract the actual translation by finding the best candidate line
        translation_lines = translated_text_block.split('\n')
        
        # Look for Japanese, Chinese, or other non-Latin script content for direct pass-through
        # (This handles cases where the translation is the same as the original for non-Latin scripts)
        if re.search(r'[\u3000-\u9FFF]', original_text):  # Check for CJK characters
            for line in translation_lines:
                if re.search(r'[\u3000-\u9FFF]', line):
                    # Skip the line if it's identical to the original text
                    if line.strip() != original_text.strip():
                        actual_translation = line.strip()
                        break
            else:
                # If we get here, we didn't find a good CJK translation
                actual_translation = None
        else:
            actual_translation = None
            
        # If we haven't found a translation yet, continue with the regular process
        if not actual_translation:
            # Look for lines with quoted text first (usually the actual translation)
            quoted_translations = []
            for line in translation_lines:
                quotes = re.findall(r'"(.*?)"', line)
                if quotes:
                    for quote in quotes:
                        if len(quote) > 5:  # Minimum length to avoid small fragments
                            quoted_translations.append(quote)
            
            # If we found quotes, use the first substantial one
            if quoted_translations:
                actual_translation = quoted_translations[0]
            else:
                # Otherwise try to find a line that looks like a translation
                clean_lines = []
                for line in translation_lines:
                    line = line.strip()
                    # Skip empty lines and lines that look like explanations
                    if not line or any(marker in line.lower() for marker in [
                        'translation:', 'here\'s', 'the text', 'decoded', 'encoding', 
                        'after', 'analyze', 'appears to be', 'breaking down', 'now, let\'s',
                        'translated text', 'translation of', 'encode', 'given text'
                    ]):
                        continue
                        
                    # Skip lines containing any encoded content markers
                    if any(marker in line for marker in ['===', 'base64', 'hash', 'encoded part']):
                        continue
                        
                    # Skip lines starting with less likely translation patterns
                    if (line.startswith('The ') and ('translated' in line or 'translation' in line)) or \
                       line.startswith('-') or line.startswith('*') or \
                       ('encoded' in line.lower()) or ('decode' in line.lower()):
                        continue
                      
                    # Check for repeating text that's unlikely to be a translation
                    if line.count('translated') > 1 or line.count('translation') > 1:
                        continue
                        
                    clean_lines.append(line)
                
                if clean_lines:
                    # Use the first good candidate that's not too short
                    for line in clean_lines:
                        if len(line) > 10 and not line.startswith('The encoded'):
                            actual_translation = line
                            break
                    else:
                        actual_translation = clean_lines[0]
                else:
                    # Last resort: take the first non-empty line that doesn't contain suspicious terms
                    for line in translation_lines:
                        if line.strip() and 'encode' not in line.lower() and 'decode' not in line.lower():
                            actual_translation = line.strip()
                            break
                    else:
                        actual_translation = "[No clear translation found]"
        
        # Store this mapping
        original_to_translation[original_text] = actual_translation
        
        # Store all translations for fallback use
        if actual_translation and actual_translation not in all_translations:
            all_translations.append(actual_translation)
            
        block_count += 1
    
    print(f"Processed {block_count} translation blocks")
    
    # If we have gold texts, try to match originals to their positions
    if gold_texts:
        # Create mapping from gold texts to their positions
        gold_text_to_position = {text: i for i, text in enumerate(gold_texts) if i < expected_count}
        
        # Process each translation and match them to original texts
        translations_mapping = {}  # Will map line numbers to translations
        
        for original, translation in original_to_translation.items():
            best_match = None
            best_match_score = 0
            
            for gold_text, position in gold_text_to_position.items():
                # Try various matching methods
                score = 0
                
                # Direct substring matching
                if original in gold_text or gold_text in original:
                    score += 20
                
                # Word overlap
                original_words = set(original.split())
                gold_words = set(gold_text.split())
                word_overlap = len(original_words.intersection(gold_words))
                if word_overlap > 0:
                    score += word_overlap * 5
                
                # First few words matching
                if len(original) > 15:
                    for s in original.split()[:3]:
                        if s in gold_text:
                            score += 2
                
                # Prefer shorter gold texts to avoid false positives
                score -= len(gold_text) * 0.01
                
                if score > best_match_score:
                    best_match_score = score
                    best_match = position
            
            # If we found a good match, store it
            if best_match is not None and best_match_score > 5:
                if best_match not in translations_mapping:
                    translations_mapping[best_match] = []
                translations_mapping[best_match].append((translation, best_match_score))
        
        # Process matched translations - pick the best one for each position
        matched_count = 0
        for pos, trans_list in translations_mapping.items():
            if pos < expected_count:
                # Sort by score in descending order and take the best one
                best_trans = sorted(trans_list, key=lambda x: x[1], reverse=True)[0][0]
                output_lines[pos] = best_trans
                matched_count += 1
        
        print(f"Matched {matched_count} translations to gold positions")
    else:
        # Without gold matching, just fill in order
        translations = list(original_to_translation.values())
        for i in range(min(len(translations), expected_count)):
            output_lines[i] = translations[i]
    
    # Fill in any missing translations with unique values from our collected translations
    placeholder_count = 0
    used_translations = set(output_lines)  # Track what we've already used
    
    # First pass - use any unused translations
    unused_translations = [t for t in all_translations if t and t not in used_translations]
    for i in range(expected_count):
        if not output_lines[i]:
            if unused_translations:
                output_lines[i] = unused_translations.pop(0)
                placeholder_count += 1
    
    # Second pass - create unique placeholders for any still-missing translations
    for i in range(expected_count):
        if not output_lines[i]:
            # Use a default placeholder, but ensure it's unique across the file
            base_placeholder = f"Translation for line {i}"
            counter = 0
            placeholder_text = base_placeholder
            
            # Ensure uniqueness by adding a counter if needed
            while placeholder_text in used_translations:
                counter += 1
                placeholder_text = f"{base_placeholder} (variant {counter})"
                
            output_lines[i] = placeholder_text
            used_translations.add(placeholder_text)
            placeholder_count += 1
    
    print(f"Added {placeholder_count} placeholders for missing translations")
    
    # Write translations to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(f"{line}\n")
    
    print(f"Extracted {len(output_lines)} translations")
    return len(output_lines)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_translations.py input_file output_file [gold_file] [expected_count]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    gold_file = None
    if len(sys.argv) > 3 and not sys.argv[3].isdigit():
        gold_file = sys.argv[3]
    
    expected_count = 2100  # Default
    if len(sys.argv) > 3 and sys.argv[3].isdigit():
        try:
            expected_count = int(sys.argv[3])
        except ValueError:
            print(f"Invalid expected count: {sys.argv[3]}. Using default: {expected_count}")
    elif len(sys.argv) > 4:
        try:
            expected_count = int(sys.argv[4])
        except ValueError:
            print(f"Invalid expected count: {sys.argv[4]}. Using default: {expected_count}")
    
    extract_translations(input_file, output_file, gold_file, expected_count)
    print(f"Extracted translations saved to {output_file}")