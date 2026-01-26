#!/usr/bin/env python3
"""
Split sentences dataset into training and validation sets.

This script provides reproducible train/val splits with configurable ratios
and fixed random seeds for experimental consistency.
"""

import argparse
import os
import random
from datetime import datetime


def split_dataset(input_path, train_path, val_path, train_ratio=0.8, seed=42):
    """
    Split sentences into training and validation sets.
    
    Args:
        input_path: Path to the input sentences file
        train_path: Path to save training sentences
        val_path: Path to save validation sentences
        train_ratio: Ratio of training data (default: 0.8 for 80/20 split)
        seed: Random seed for reproducibility (default: 42)
    """
    print("="*60)
    print(f"Dataset Split Started: {datetime.now()}")
    print("="*60)
    print(f"Input file: {input_path}")
    print(f"Train output: {train_path}")
    print(f"Val output: {val_path}")
    print(f"Train ratio: {train_ratio:.1%}")
    print(f"Val ratio: {1-train_ratio:.1%}")
    print(f"Random seed: {seed}")
    print("="*60)
    
    # Read all sentences
    with open(input_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    total = len(sentences)
    print(f"Total sentences loaded: {total}")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(sentences)
    print(f"Sentences shuffled with seed={seed}")
    
    # Calculate split point
    train_size = int(total * train_ratio)
    val_size = total - train_size
    
    # Split the data
    train_sentences = sentences[:train_size]
    val_sentences = sentences[train_size:]
    
    # Write training set
    with open(train_path, 'w', encoding='utf-8') as f:
        for sentence in train_sentences:
            f.write(sentence + '\n')
    
    # Write validation set
    with open(val_path, 'w', encoding='utf-8') as f:
        for sentence in val_sentences:
            f.write(sentence + '\n')
    
    # Compute statistics
    train_lengths = [len(s) for s in train_sentences]
    val_lengths = [len(s) for s in val_sentences]
    
    train_avg = sum(train_lengths) / len(train_lengths) if train_lengths else 0
    val_avg = sum(val_lengths) / len(val_lengths) if val_lengths else 0
    
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    print(f"Training set:")
    print(f"  - Sentences: {len(train_sentences)} ({len(train_sentences)/total:.1%})")
    print(f"  - Avg length: {train_avg:.1f} characters")
    print(f"  - Saved to: {train_path}")
    print()
    print(f"Validation set:")
    print(f"  - Sentences: {len(val_sentences)} ({len(val_sentences)/total:.1%})")
    print(f"  - Avg length: {val_avg:.1f} characters")
    print(f"  - Saved to: {val_path}")
    print("="*60)
    print(f"Split Completed: {datetime.now()}")
    print("="*60)
    
    return train_size, val_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split sentences dataset into train/val sets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_path',
                        default='/info/raid-etu/m2/s2405959/VO2/Agent/DB/sentences.txt',
                        help='Path to input sentences file')
    parser.add_argument('--train_path',
                        default='/info/raid-etu/m2/s2405959/VO2/Agent/DB/sentences_train.txt',
                        help='Path to output training sentences')
    parser.add_argument('--val_path',
                        default='/info/raid-etu/m2/s2405959/VO2/Agent/DB/sentences_val.txt',
                        help='Path to output validation sentences')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate train ratio
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {args.train_ratio}")
    
    # Create output directories if needed
    os.makedirs(os.path.dirname(args.train_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.val_path), exist_ok=True)
    
    split_dataset(args.input_path, args.train_path, args.val_path, 
                 args.train_ratio, args.seed)
