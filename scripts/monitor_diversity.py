"""
Enhanced monitoring dashboard for RL training with diversity metrics.

Analyzes training_progress.csv and generates comprehensive visualizations:
- Loss and reward curves
- Prompt diversity metrics
- Action usage histograms
- Reward distribution plots
- Exploration statistics

Usage:
    python monitor_diversity.py --csv logs_agent/training_progress.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import json


def load_training_data(csv_path):
    """Load training progress CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} training records from {csv_path}")
    return df


def plot_loss_and_reward(df, output_dir):
    """Plot loss and reward curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curve
    axes[0, 0].plot(df.index, df['loss'], color='steelblue', linewidth=1.5)
    axes[0, 0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(alpha=0.3)
    
    # Reward curve
    axes[0, 1].plot(df.index, df['reward'], color='forestgreen', linewidth=1.5, alpha=0.7)
    axes[0, 1].axhline(df['reward'].mean(), color='red', linestyle='--', label=f'Mean: {df["reward"].mean():.4f}')
    axes[0, 1].set_title('Reward Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Reward (1 - CER)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Reward distribution
    axes[1, 0].hist(df['reward'], bins=30, color='forestgreen', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(df['reward'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["reward"].mean():.4f}')
    axes[1, 0].axvline(df['reward'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["reward"].median():.4f}')
    axes[1, 0].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Loss vs Reward scatter
    axes[1, 1].scatter(df['reward'], df['loss'], alpha=0.5, s=20, color='purple')
    axes[1, 1].set_title('Loss vs Reward', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'loss_reward_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved loss/reward plot: {output_path}")
    plt.close()


def plot_diversity_metrics(df, output_dir):
    """Plot diversity and exploration metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prompt usage frequency
    if 'prompt_idx' in df.columns:
        prompt_counts = df['prompt_idx'].value_counts()
        top_prompts = prompt_counts.head(20)
        
        axes[0, 0].bar(range(len(top_prompts)), top_prompts.values, color='coral', alpha=0.7)
        axes[0, 0].set_title(f'Top 20 Prompt Usage (Total Unique: {len(prompt_counts)})', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Prompt Rank')
        axes[0, 0].set_ylabel('Usage Count')
        axes[0, 0].grid(alpha=0.3)
        
        # Highlight mode collapse
        max_usage = prompt_counts.iloc[0]
        max_prompt = prompt_counts.index[0]
        axes[0, 0].text(0.5, 0.95, 
                       f"Mode Collapse Alert: Prompt {max_prompt} used {max_usage}/{len(df)} times ({max_usage/len(df)*100:.1f}%)",
                       transform=axes[0, 0].transAxes,
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                       fontsize=10, fontweight='bold')
        
        # Prompt diversity over time (unique prompts in sliding window)
        window_size = min(50, len(df) // 4)
        unique_counts = [df['prompt_idx'].iloc[max(0, i-window_size):i].nunique() 
                        for i in range(window_size, len(df))]
        
        axes[0, 1].plot(range(window_size, len(df)), unique_counts, color='teal', linewidth=1.5)
        axes[0, 1].axhline(np.mean(unique_counts), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(unique_counts):.1f}')
        axes[0, 1].set_title(f'Prompt Diversity (Unique in Last {window_size} Iterations)', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Unique Prompts')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No prompt_idx column found', 
                       ha='center', va='center', fontsize=12)
        axes[0, 1].text(0.5, 0.5, 'No prompt_idx column found', 
                       ha='center', va='center', fontsize=12)
    
    # Entropy over time (if available)
    if 'entropy' in df.columns:
        axes[1, 0].plot(df.index, df['entropy'], color='darkviolet', linewidth=1.5, alpha=0.7)
        axes[1, 0].axhline(df['entropy'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["entropy"].mean():.4f}')
        axes[1, 0].set_title('Entropy Regularization Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No entropy column found', 
                       ha='center', va='center', fontsize=12)
    
    # CER distribution
    if 'cer' in df.columns:
        axes[1, 1].hist(df['cer'], bins=30, color='indianred', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(df['cer'].mean(), color='blue', linestyle='--', linewidth=2, 
                          label=f'Mean: {df["cer"].mean():.4f}')
        axes[1, 1].set_title('Character Error Rate (CER) Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('CER')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No CER column found', 
                       ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'diversity_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved diversity plot: {output_path}")
    plt.close()


def generate_statistics_report(df, output_dir):
    """Generate comprehensive statistics report."""
    report = []
    report.append("="*80)
    report.append("REINFORCEMENT LEARNING TRAINING STATISTICS")
    report.append("="*80)
    report.append("")
    
    # Basic stats
    report.append("TRAINING PROGRESS")
    report.append("-"*80)
    report.append(f"Total Iterations:        {len(df)}")
    report.append(f"Loss (final):            {df['loss'].iloc[-1]:.6f}")
    report.append(f"Loss (mean):             {df['loss'].mean():.6f} ± {df['loss'].std():.6f}")
    report.append(f"Loss (min):              {df['loss'].min():.6f}")
    report.append(f"Loss (max):              {df['loss'].max():.6f}")
    report.append("")
    
    # Reward stats
    report.append("REWARD METRICS")
    report.append("-"*80)
    report.append(f"Reward (final):          {df['reward'].iloc[-1]:.4f}")
    report.append(f"Reward (mean):           {df['reward'].mean():.4f} ± {df['reward'].std():.4f}")
    report.append(f"Reward (median):         {df['reward'].median():.4f}")
    report.append(f"Reward (min):            {df['reward'].min():.4f}")
    report.append(f"Reward (max):            {df['reward'].max():.4f}")
    report.append("")
    
    # CER stats
    if 'cer' in df.columns:
        report.append("CHARACTER ERROR RATE (CER)")
        report.append("-"*80)
        report.append(f"CER (final):             {df['cer'].iloc[-1]:.4f}")
        report.append(f"CER (mean):              {df['cer'].mean():.4f} ± {df['cer'].std():.4f}")
        report.append(f"CER (median):            {df['cer'].median():.4f}")
        report.append(f"CER (min):               {df['cer'].min():.4f}")
        report.append(f"CER (max):               {df['cer'].max():.4f}")
        report.append("")
    
    # Diversity stats
    if 'prompt_idx' in df.columns:
        prompt_counts = df['prompt_idx'].value_counts()
        report.append("DIVERSITY METRICS")
        report.append("-"*80)
        report.append(f"Unique Prompts Used:     {len(prompt_counts)}")
        report.append(f"Most Used Prompt:        Index {prompt_counts.index[0]} ({prompt_counts.iloc[0]} times, {prompt_counts.iloc[0]/len(df)*100:.1f}%)")
        
        if len(prompt_counts) >= 3:
            top3_usage = prompt_counts.iloc[:3].sum() / len(df) * 100
            report.append(f"Top 3 Prompts Coverage:  {top3_usage:.1f}%")
        
        # Mode collapse detection
        if prompt_counts.iloc[0] / len(df) > 0.3:
            report.append(f"[WARNING] MODE COLLAPSE DETECTED: Top prompt used >{prompt_counts.iloc[0]/len(df)*100:.1f}% of the time!")
        report.append("")
    
    # Entropy stats
    if 'entropy' in df.columns:
        report.append("ENTROPY REGULARIZATION")
        report.append("-"*80)
        report.append(f"Entropy (final):         {df['entropy'].iloc[-1]:.4f}")
        report.append(f"Entropy (mean):          {df['entropy'].mean():.4f} ± {df['entropy'].std():.4f}")
        report.append("")
    
    report.append("="*80)
    
    # Print to console
    report_text = '\n'.join(report)
    print(report_text)
    
    # Save to file
    output_path = Path(output_dir) / 'training_statistics.txt'
    with open(output_path, 'w') as f:
        f.write(report_text)
    print(f"\nSaved statistics report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive RL training monitoring with diversity metrics."
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='logs_agent/training_progress.csv',
        help='Path to training_progress.csv (default: logs_agent/training_progress.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='logs_agent',
        help='Output directory for plots and reports (default: logs_agent)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        return 1
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_training_data(args.csv)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_loss_and_reward(df, args.output_dir)
    plot_diversity_metrics(df, args.output_dir)
    
    # Generate statistics report
    print("\nGenerating statistics report...")
    generate_statistics_report(df, args.output_dir)
    
    print(f"\nMonitoring complete! Results saved to: {args.output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
