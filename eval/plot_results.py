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

def collect_results(base_dir, challenge_scores=None):
    """Collect results from all model variants.
    
    Args:
        base_dir: Directory containing model results
        challenge_scores: Optional dict with challenge scores in format 
                          {'model_name': {'bert_challenge': value, 'xcomet_challenge': value}}
    """
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
            
            # Add challenge scores if provided
            if challenge_scores and model in challenge_scores:
                model_challenges = challenge_scores[model]
                
                if 'bert_challenge' in model_challenges:
                    result['bert_challenge'] = model_challenges['bert_challenge']
                
                if 'xcomet_challenge' in model_challenges:
                    result['xcomet_challenge'] = model_challenges['xcomet_challenge']
                
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
            
            # Plot challenge score if available (as star marker)
            challenge_df = model_data[model_data['bert_challenge'].notna()]
            if len(challenge_df) > 0:
                plt.scatter(
                    challenge_df['epochs'], 
                    challenge_df['bert_challenge'],
                    marker='*',  # Star marker
                    s=150,       # Size
                    color=plt.gca().lines[-1].get_color(),  # Match line color
                    edgecolor='black',
                    label=f"{model} (challenge)"
                )
        
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
            
            # Plot challenge score if available (as star marker)
            challenge_df = model_data[model_data['xcomet_challenge'].notna()]
            if len(challenge_df) > 0:
                plt.scatter(
                    challenge_df['epochs'], 
                    challenge_df['xcomet_challenge'],
                    marker='*',  # Star marker
                    s=150,       # Size
                    color=plt.gca().lines[-1].get_color(),  # Match line color
                    edgecolor='black',
                    label=f"{model} (challenge)"
                )
        
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
    # Define metrics and their challenge counterparts
    metric_pairs = {
        'bert_f1': 'bert_challenge',
        'xcomet': 'xcomet_challenge'
    }
    
    # Create individual plots for each metric
    for metric, challenge_metric in metric_pairs.items():
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model].sort_values('epochs')
            
            # Skip if no data for this metric
            if metric not in model_data.columns or model_data[metric].isna().all():
                continue
                
            # Plot regular metric line
            plt.plot(model_data['epochs'], model_data[metric], marker='o', linewidth=2, label=model)
            
            # Add challenge points if available
            if challenge_metric in model_data.columns:
                challenge_data = model_data[model_data[challenge_metric].notna()]
                if len(challenge_data) > 0:
                    plt.scatter(
                        challenge_data['epochs'],
                        challenge_data[challenge_metric],
                        marker='*',
                        s=150,
                        color=plt.gca().lines[-1].get_color(),
                        edgecolor='black',
                        label=f"{model} (challenge)"
                    )
        
        # Format the plot
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
    parser.add_argument('--aya-bert-challenge', type=float, help='Aya BERT challenge score')
    parser.add_argument('--aya-xcomet-challenge', type=float, help='Aya XCOMET challenge score')
    parser.add_argument('--llama-bert-challenge', type=float, help='Llama BERT challenge score', default=0.8759)
    parser.add_argument('--llama-xcomet-challenge', type=float, help='Llama XCOMET challenge score', default=0.6295)
    parser.add_argument('--challenge-epoch', type=int, default=3, 
                        help='Which epoch to place the challenge score at (default: 3)')
    
    args = parser.parse_args()
    
    # Prepare challenge scores if provided
    challenge_scores = {}
    
    # Add Aya challenge scores if provided
    if args.aya_bert_challenge is not None or args.aya_xcomet_challenge is not None:
        challenge_scores['aya_expanse_8b'] = {}
        
        if args.aya_bert_challenge is not None:
            challenge_scores['aya_expanse_8b']['bert_challenge'] = args.aya_bert_challenge
            
        if args.aya_xcomet_challenge is not None:
            challenge_scores['aya_expanse_8b']['xcomet_challenge'] = args.aya_xcomet_challenge
    
    # Add Llama challenge scores if provided
    if args.llama_bert_challenge is not None or args.llama_xcomet_challenge is not None:
        challenge_scores['llama'] = {}
        
        if args.llama_bert_challenge is not None:
            challenge_scores['llama']['bert_challenge'] = args.llama_bert_challenge
            
        if args.llama_xcomet_challenge is not None:
            challenge_scores['llama']['xcomet_challenge'] = args.llama_xcomet_challenge
    
    # Collect results
    results_df = collect_results(args.data_dir, challenge_scores)
    
    if len(results_df) == 0:
        print("No results found. Check the paths and try again.")
        return
    
    # Set the challenge epoch if challenge scores were provided
    if challenge_scores and args.challenge_epoch is not None:
        # Only add challenge scores to rows with the specified epoch (default: 3)
        for model in challenge_scores:
            mask = (results_df['model'] == model) & (results_df['epochs'] == args.challenge_epoch)
            if sum(mask) > 0:
                for challenge_key in ['bert_challenge', 'xcomet_challenge']:
                    if challenge_key in challenge_scores[model]:
                        # Add challenge score only to the specified epoch
                        results_df.loc[mask, challenge_key] = challenge_scores[model][challenge_key]
                        
                        # Clear any challenge scores at other epochs
                        other_epochs_mask = (results_df['model'] == model) & (results_df['epochs'] != args.challenge_epoch)
                        if sum(other_epochs_mask) > 0:
                            results_df.loc[other_epochs_mask, challenge_key] = np.nan
    
    # Plot results
    plot_results(results_df, args.output_dir)
    
    # Create summary table
    create_summary_table(results_df, args.output_dir)

if __name__ == "__main__":
    main()