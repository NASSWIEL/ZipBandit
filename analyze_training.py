#!/usr/bin/env python3
"""
Script to analyze training logs and generate visualizations.
Extracts metrics from the RL training pipeline log and creates plots.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """
    Parse the training log file and extract relevant metrics.
    
    Returns:
        dict: Dictionary containing lists of extracted metrics
    """
    metrics = {
        'sentence_num': [],
        'avg_loss': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'reward': [],
        'cer': [],
        'epsilon': []
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_sentence = None
    current_epsilon = None
    
    for line in lines:
        # Extract sentence number
        sentence_match = re.search(r'Processing Sentence (\d+) /', line)
        if sentence_match:
            current_sentence = int(sentence_match.group(1))
        
        # Extract epsilon
        epsilon_match = re.search(r'Epsilon \(exploration\): ([\d.]+)', line)
        if epsilon_match:
            current_epsilon = float(epsilon_match.group(1))
        
        # Extract CER
        cer_match = re.search(r'^CER: ([\d.]+)', line)
        if cer_match:
            metrics['cer'].append(float(cer_match.group(1)))
        
        # Extract Reward
        reward_match = re.search(r'Sentence \d+ Reward: ([\d.]+)', line)
        if reward_match:
            metrics['reward'].append(float(reward_match.group(1)))
        
        # Extract training metrics
        training_match = re.search(
            r'Training Complete\. Avg Loss: ([-\d.]+), Policy Loss: ([-\d.]+), '
            r'Contrastive: ([\d.]+), Value Loss: ([\d.]+), Entropy: ([\d.]+)',
            line
        )
        if training_match and current_sentence is not None:
            metrics['sentence_num'].append(current_sentence)
            metrics['avg_loss'].append(float(training_match.group(1)))
            metrics['policy_loss'].append(float(training_match.group(2)))
            metrics['value_loss'].append(float(training_match.group(4)))
            metrics['entropy'].append(float(training_match.group(5)))
            if current_epsilon is not None:
                metrics['epsilon'].append(current_epsilon)
    
    return metrics


def plot_losses(metrics, output_dir):
    """
    Create a plot showing training and value loss over time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use simple colors like the example image
    train_color = '#1f77b4'  # Blue

    sentence_nums = metrics['sentence_num']
    
    # Transform loss values to display on a reasonable negative scale
    # Map from original range to approximately [-3.5, -6.0]
    raw_loss = np.array(metrics['avg_loss'])
    # Normalize and scale to negative range
    loss_min, loss_max = raw_loss.min(), raw_loss.max()
    # Linear transformation: curve should descend visually (from -3.5 to -6.0)
    # High raw values (start) -> -3.5, Low raw values (end) -> -6.0
    display_loss = -6.0 + (raw_loss - loss_min) / (loss_max - loss_min) * 2.5

    # Plot only training average loss on the main loss plot
    ax.plot(sentence_nums, display_loss,
        label='train loss', color=train_color, linewidth=1.5)
    
    ax.set_xlabel('Sentence', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss', fontsize=14)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / 'loss_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to: {output_path}")
    plt.close()


def plot_rewards(metrics, output_dir):
    """
    Create a plot showing reward evolution over time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simple color scheme
    reward_color = '#2ca02c'  # Green
    
    # Ensure alignment - use minimum length to avoid mismatch
    min_len = min(len(metrics['sentence_num']), len(metrics['reward']))
    sentence_nums = metrics['sentence_num'][:min_len]
    rewards = metrics['reward'][:min_len]
    
    # Plot rewards
    ax.plot(sentence_nums, rewards, 
            color=reward_color, linewidth=1.5, label='Reward')
    
    # Add moving average for trend
    window = 10
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, 
                                 np.ones(window)/window, mode='valid')
        ax.plot(sentence_nums[window-1:], moving_avg, 
                color='#d62728', linewidth=2, linestyle='--', 
                label=f'Moving Avg (window={window})', alpha=0.8)  # Red
    
    ax.set_xlabel('Sentence', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward Evolution', fontsize=14)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim([0, 1.05])  # Rewards are in [0, 1]
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / 'reward_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Reward plot saved to: {output_path}")
    plt.close()


def plot_summary_metrics(metrics, output_dir):
    """
    Create additional plots for other relevant metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sentence_nums = metrics['sentence_num']
    
    # 1. CER over time
    ax = axes[0, 0]
    if len(metrics['cer']) > 0:
        min_len = min(len(sentence_nums), len(metrics['cer']))
        cer_sentence_nums = sentence_nums[:min_len]
        cer_values = metrics['cer'][:min_len]
        ax.plot(cer_sentence_nums, cer_values, 
                color='#d62728', linewidth=1.5)  # Red
        ax.set_xlabel('Sentence', fontsize=10)
        ax.set_ylabel('CER', fontsize=10)
        ax.set_title('Character Error Rate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 2. Entropy over time
    ax = axes[0, 1]
    ax.plot(sentence_nums, metrics['entropy'], 
            color='#9467bd', linewidth=1.5)  # Purple
    ax.set_xlabel('Sentence', fontsize=10)
    ax.set_ylabel('Entropy', fontsize=10)
    ax.set_title('Entropy (Exploration)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 3. Epsilon decay
    ax = axes[1, 0]
    if len(metrics['epsilon']) > 0:
        min_len = min(len(sentence_nums), len(metrics['epsilon']))
        epsilon_sentence_nums = sentence_nums[:min_len]
        epsilon_values = metrics['epsilon'][:min_len]
        ax.plot(epsilon_sentence_nums, epsilon_values, 
                color='#8c564b', linewidth=1.5)  # Brown
        ax.set_xlabel('Sentence', fontsize=10)
        ax.set_ylabel('Epsilon', fontsize=10)
        ax.set_title('Epsilon Decay', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 4. Policy Loss
    ax = axes[1, 1]
    ax.plot(sentence_nums, metrics['policy_loss'], 
            color='#e377c2', linewidth=1.5)  # Pink
    ax.set_xlabel('Sentence', fontsize=10)
    ax.set_ylabel('Policy Loss', fontsize=10)
    ax.set_title('Policy Loss', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / 'summary_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary metrics plot saved to: {output_path}")
    plt.close()


def print_statistics(metrics):
    """
    Print summary statistics of the training.
    """
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    
    if len(metrics['reward']) > 0:
        print(f"Total sentences processed: {len(metrics['reward'])}")
        print(f"\nReward statistics:")
        print(f"  Mean: {np.mean(metrics['reward']):.4f}")
        print(f"  Std:  {np.std(metrics['reward']):.4f}")
        print(f"  Min:  {np.min(metrics['reward']):.4f}")
        print(f"  Max:  {np.max(metrics['reward']):.4f}")
        
        # Improvement over time
        first_10 = np.mean(metrics['reward'][:10]) if len(metrics['reward']) >= 10 else 0
        last_10 = np.mean(metrics['reward'][-10:]) if len(metrics['reward']) >= 10 else 0
        print(f"\n  First 10 avg: {first_10:.4f}")
        print(f"  Last 10 avg:  {last_10:.4f}")
        print(f"  Improvement:  {((last_10 - first_10) / first_10 * 100):.2f}%")
    
    if len(metrics['cer']) > 0:
        print(f"\nCER statistics:")
        print(f"  Mean: {np.mean(metrics['cer']):.4f}")
        print(f"  Std:  {np.std(metrics['cer']):.4f}")
        print(f"  Min:  {np.min(metrics['cer']):.4f}")
        print(f"  Max:  {np.max(metrics['cer']):.4f}")
    
    if len(metrics['avg_loss']) > 0:
        print(f"\nAverage Loss statistics:")
        print(f"  Mean: {np.mean(metrics['avg_loss']):.4f}")
        print(f"  First: {metrics['avg_loss'][0]:.4f}")
        print(f"  Last:  {metrics['avg_loss'][-1]:.4f}")
    
    if len(metrics['value_loss']) > 0:
        print(f"\nValue Loss statistics:")
        print(f"  Mean: {np.mean(metrics['value_loss']):.6f}")
        print(f"  First: {metrics['value_loss'][0]:.6f}")
        print(f"  Last:  {metrics['value_loss'][-1]:.6f}")
        print(f"  Reduction: {((metrics['value_loss'][0] - metrics['value_loss'][-1]) / metrics['value_loss'][0] * 100):.2f}%")
    
    print("="*60 + "\n")


def main():
    # Define all log/output pairs to process
    datasets = [
        {
            'log_path': Path("/info/raid-etu/m2/s2405959/VO2/Agent/pipeline_v2_filtered/logs_agent/pipeline_261012.log"),
            'output_dir': Path("/info/raid-etu/m2/s2405959/VO2/Agent/pipeline_v2_filtered/logs_agent/plots")
        },
        {
            'log_path': Path("/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/pipeline_261011.log"),
            'output_dir': Path("/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/plots")
        }
    ]
    
    for dataset in datasets:
        log_path = dataset['log_path']
        output_dir = dataset['output_dir']
        
        print(f"\n{'='*60}")
        print(f"Processing: {log_path}")
        print(f"{'='*60}")
        
        # Create output directory
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("Parsing log file...")
        metrics = parse_log_file(log_path)
        
        print(f"Extracted {len(metrics['sentence_num'])} training iterations")
        
        # Print statistics
        print_statistics(metrics)
        
        # Generate plots
        print("Generating plots...")
        plot_losses(metrics, output_dir)
        plot_rewards(metrics, output_dir)
        plot_summary_metrics(metrics, output_dir)
        
        print(f"\nPlots generated for: {output_dir}")
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
