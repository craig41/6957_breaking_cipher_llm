#!/usr/bin/env python3

import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def extract_bert_score(file_path):
    """Extract BERT scores from result file."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist")
        return None
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Extract mean scores
        mean_f1_match = re.search(r'Mean F1 for .+: ([0-9.]+)', content)
        if mean_f1_match:
            mean_f1 = float(mean_f1_match.group(1))
            return {
                'bert_f1': mean_f1
            }
        else:
            print(f"Couldn't find Mean F1 in {file_path}")
            return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_xcomet_score(file_path):
    """Extract XCOMET scores from result file."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist")
        return None
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # First try to find Average Score
        avg_match = re.search(r'Average Score: ([0-9.]+)', content)
        if avg_match:
            xcomet_score = float(avg_match.group(1))
            return {
                'xcomet': xcomet_score
            }
            
        # Fall back to looking for XCOMET: pattern
        xcomet_match = re.search(r'XCOMET: ([0-9.]+)', content)
        if xcomet_match:
            xcomet_score = float(xcomet_match.group(1))
            return {
                'xcomet': xcomet_score
            }
        else:
            print(f"Couldn't find XCOMET score in {file_path}")
            return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def collect_results(base_dir):
    """Collect results from all model variants."""
    results = []
    
    # Define the models and their variants
    models = {
        'aya_expanse_8b': ['aya_expanse_base_small', 'aya_expanse_8b_3_epochs', 'aya_expanse_8b_10_epochs'],
        'llama': ['llama_base_small', 'llama_epochs_3', 'llama_epochs_10']
    }
    
    for model, variants in models.items():
        for i, variant in enumerate(variants):
            # Construct folder name based on model and variant
            folder = variant
            
            folder_path = os.path.join(base_dir, folder)
            
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} does not exist, skipping...")
                continue
            
            # Get BERT and XCOMET results
            bert_path = os.path.join(folder_path, "bert_results.txt")
            xcomet_path = os.path.join(folder_path, "xcomet_result.txt")
            
            bert_results = extract_bert_score(bert_path)
            xcomet_results = extract_xcomet_score(xcomet_path)
            
            # Create a result entry
            result = {
                'model': model,
                'variant': variant,
                'epochs': 0 if i == 0 else (3 if i == 1 else 10)
            }
            
            # Add scores if available
            if bert_results:
                result.update(bert_results)
            if xcomet_results:
                result.update(xcomet_results)
                
            # Only add if we have at least one score
            if bert_results or xcomet_results:
                results.append(result)
    
    return pd.DataFrame(results)

def plot_results(df, output_dir):
    """Create plots for BERT and XCOMET scores."""
    # Set up the plotting style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 10))
    
    # Plot BERT-F1 scores
    if 'bert_f1' in df.columns:
        plt.subplot(2, 1, 1)
        
        # Plot lines for each model
        for model in df['model'].unique():
            model_data = df[df['model'] == model].sort_values('epochs')
            plt.plot(model_data['epochs'], model_data['bert_f1'], marker='o', linewidth=2, label=model)
        
        plt.title('BERT F1 Scores by Model and Training Epochs', fontsize=16)
        plt.xlabel('Training Epochs', fontsize=14)
        plt.ylabel('BERT F1 Score', fontsize=14)
        plt.xticks([0, 3, 10])
        plt.grid(True)
        plt.legend(fontsize=12)
    
    # Plot XCOMET scores
    if 'xcomet' in df.columns:
        plt.subplot(2, 1, 2)
        
        # Plot lines for each model
        for model in df['model'].unique():
            model_data = df[df['model'] == model].sort_values('epochs')
            plt.plot(model_data['epochs'], model_data['xcomet'], marker='o', linewidth=2, label=model)
        
        plt.title('XCOMET Scores by Model and Training Epochs', fontsize=16)
        plt.xlabel('Training Epochs', fontsize=14)
        plt.ylabel('XCOMET Score', fontsize=14)
        plt.xticks([0, 3, 10])
        plt.grid(True)
        plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'model_comparison.pdf'))
    print(f"Plots saved to {output_dir}")

    # Also create individual model plots
    plot_individual_models(df, output_dir)

def plot_individual_models(df, output_dir):
    """Create separate plots for each metric and model."""
    metrics = [col for col in df.columns if col not in ['model', 'variant', 'epochs']]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model].sort_values('epochs')
            if metric in model_data.columns and not model_data[metric].isna().all():
                plt.plot(model_data['epochs'], model_data[metric], marker='o', linewidth=2, label=model)
        
        metric_name = metric.replace('_', ' ').upper()
        plt.title(f'{metric_name} by Model and Training Epochs', fontsize=16)
        plt.xlabel('Training Epochs', fontsize=14)
        plt.ylabel(metric_name, fontsize=14)
        plt.xticks([0, 3, 10])
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.pdf'))

def create_summary_table(df, output_dir):
    """Create a summary table of all results."""
    if len(df) == 0:
        print("No data available for summary table")
        return
    
    # Format the table
    table_df = df.pivot_table(
        index='model', 
        columns='variant',
        values=['bert_f1', 'xcomet'],
        aggfunc='first'
    )
    
    # Save as CSV
    table_path = os.path.join(output_dir, 'results_summary.csv')
    table_df.to_csv(table_path)
    print(f"Summary table saved to {table_path}")
    
    # Print to console
    print("\nResults Summary:")
    print(table_df)

def main():
    parser = argparse.ArgumentParser(description='Plot BERT and XCOMET scores for different models.')
    parser.add_argument('--data-dir', type=str, default='/scratch/general/vast/u1380656/6957_breaking_cipher_llm/data/lora_full',
                        help='Directory containing model results')
    parser.add_argument('--output-dir', type=str, default='/scratch/general/vast/u1380656/6957_breaking_cipher_llm/eval/plots',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Collect results
    results_df = collect_results(args.data_dir)
    
    if len(results_df) == 0:
        print("No results found. Check the paths and try again.")
        return
    
    # Plot results
    plot_results(results_df, args.output_dir)
    
    # Create summary table
    create_summary_table(results_df, args.output_dir)

if __name__ == "__main__":
    main()