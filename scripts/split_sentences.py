"""
Split sentences.txt into train/validation sets for proper evaluation.

Creates:
- DB/sentences_train.txt (80% of sentences)
- DB/sentences_val.txt (20% of sentences)

Usage:
    python scripts/split_sentences.py --input DB/sentences.txt --train_ratio 0.8
"""

import argparse
import random
import os
from pathlib import Path


def split_sentences(input_path, train_ratio=0.8, shuffle=True, seed=42):
    """
    Split sentences into train and validation sets.
    
    Args:
        input_path (str): Path to input sentences.txt file.
        train_ratio (float): Ratio of training data (0.0 to 1.0).
        shuffle (bool): Whether to shuffle sentences before splitting.
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (train_sentences, val_sentences)
    """
    # Read all sentences
    with open(input_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(sentences)} sentences from {input_path}")
    
    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(sentences)
        print(f"Shuffled sentences with seed={seed}")
    
    # Split
    split_idx = int(len(sentences) * train_ratio)
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]
    
    print(f"Split: {len(train_sentences)} train, {len(val_sentences)} validation")
    
    return train_sentences, val_sentences


def save_split(sentences, output_path):
    """Save sentences to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    
    print(f"Saved {len(sentences)} sentences to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split sentences.txt into train/validation sets."
    )
    parser.add_argument(
        '--input',
        type=str,
        default='DB/sentences.txt',
        help='Path to input sentences.txt (default: DB/sentences.txt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='DB',
        help='Output directory (default: DB)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of training data (default: 0.8 = 80%%)'
    )
    parser.add_argument(
        '--no_shuffle',
        action='store_true',
        help='Do not shuffle sentences before splitting'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if not 0.0 < args.train_ratio < 1.0:
        print(f"Error: train_ratio must be between 0 and 1, got {args.train_ratio}")
        return 1
    
    # Split sentences
    train_sentences, val_sentences = split_sentences(
        args.input,
        train_ratio=args.train_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )
    
    if len(val_sentences) == 0:
        print("Warning: Validation set is empty! Consider using a smaller train_ratio.")
    
    # Save splits
    train_path = os.path.join(args.output_dir, 'sentences_train.txt')
    val_path = os.path.join(args.output_dir, 'sentences_val.txt')
    
    save_split(train_sentences, train_path)
    save_split(val_sentences, val_path)
    
    # Print summary
    print("\n" + "="*60)
    print("SPLIT SUMMARY")
    print("="*60)
    print(f"Input:      {args.input}")
    print(f"Total:      {len(train_sentences) + len(val_sentences)} sentences")
    print(f"Training:   {len(train_sentences)} sentences ({args.train_ratio*100:.1f}%)")
    print(f"            → {train_path}")
    print(f"Validation: {len(val_sentences)} sentences ({(1-args.train_ratio)*100:.1f}%)")
    print(f"            → {val_path}")
    print(f"Shuffle:    {'Yes' if not args.no_shuffle else 'No'}")
    print(f"Seed:       {args.seed}")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())
