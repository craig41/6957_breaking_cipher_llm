#!/usr/bin/env python3
"""
Calculate average XCOMET score from an output file and write the result to a destination folder.
Usage: python calculate_average_xcomet.py <output_file>
Example: python calculate_average_xcomet.py outputs-4124354
"""

import sys
import os
import re
import statistics

def extract_folder_from_file(output_file):
    """Extract the destination folder from the output file content."""
    try:
        with open(output_file, 'r') as f:
            for line in f:
                if 'score:' in line:
                    # Look for paths that end with /translated.txt
                    match = re.search(r'(/scratch/general/vast/u\d+/[\w/\-_]+/)translated\.txt', line)
                    if match:
                        return match.group(1)
    except Exception as e:
        print(f"Error extracting folder from {output_file}: {e}")
    
    return None

def calculate_average_score(output_file):
    """Parse the XCOMET output file and calculate the average score."""
    scores = []
    try:
        with open(output_file, 'r') as f:
            for line in f:
                if 'score:' in line:
                    # Extract score value using regex
                    match = re.search(r'score:\s+(\d+\.\d+)', line)
                    if match:
                        score = float(match.group(1))
                        scores.append(score)
    except Exception as e:
        print(f"Error processing file {output_file}: {e}")
        return None

    if not scores:
        print(f"No scores found in {output_file}")
        return None

    # Calculate statistics
    avg_score = sum(scores) / len(scores)
    median_score = statistics.median(scores)
    min_score = min(scores)
    max_score = max(scores)
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0

    results = {
        'average': avg_score,
        'median': median_score,
        'min': min_score,
        'max': max_score,
        'std_dev': std_dev,
        'count': len(scores)
    }
    
    return results

def save_results(dest_folder, results):
    """Save the results to the destination folder."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    output_file = os.path.join(dest_folder, 'xcomet_result.txt')
    
    with open(output_file, 'w') as f:
        f.write(f"XCOMET Score Summary\n")
        f.write(f"====================\n")
        f.write(f"Average Score: {results['average']:.4f}\n")
        f.write(f"Median Score: {results['median']:.4f}\n")
        f.write(f"Min Score: {results['min']:.4f}\n")
        f.write(f"Max Score: {results['max']:.4f}\n")
        f.write(f"Standard Deviation: {results['std_dev']:.4f}\n")
        f.write(f"Number of Segments: {results['count']}\n")
    
    print(f"Results saved to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_average_xcomet.py <output_file>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    # Check if output_file is just a filename or a full path
    if not os.path.isfile(output_file):
        # Try prepending the eval directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(base_dir, output_file)
        
        if not os.path.isfile(output_file):
            print(f"Error: Could not find file {sys.argv[1]}")
            sys.exit(1)
    
    # Extract destination folder from the output file
    dest_folder = extract_folder_from_file(output_file)
    if not dest_folder:
        print("Error: Could not extract destination folder from the output file.")
        print("Please specify destination folder as a second argument.")
        sys.exit(1)
    
    results = calculate_average_score(output_file)
    
    if results:
        save_results(dest_folder, results)
        # Print the average score for easy reference
        print(f"Average XCOMET score: {results['average']:.4f}")
        print(f"Destination folder: {dest_folder}")
    else:
        print("Failed to calculate scores")
        sys.exit(1)

if __name__ == "__main__":
    main()