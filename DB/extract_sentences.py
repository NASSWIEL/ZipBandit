#!/usr/bin/env python3
"""
Extract sentences from TSV file for training.
This script reads the NEB_test_clean.tsv file and extracts the first N sentences
to be used for training the agent model on diverse prompts.

Enhanced with:
- Configurable extraction count (default: 1000)
- Detailed logging
- Statistics reporting
- Reproducible extraction with seed-based shuffling option
"""

import argparse
import csv
import os
import random
from datetime import datetime


def extract_sentences(tsv_path, output_path, num_sentences=1000, shuffle=False, seed=42):
    """
    Extract sentences from TSV file with optional shuffling.
    
    Args:
        tsv_path: Path to the input TSV file
        output_path: Path to save the extracted sentences
        num_sentences: Number of sentences to extract (default: 1000)
        shuffle: Whether to shuffle before extracting (default: False)
        seed: Random seed for reproducibility (default: 42)
    """
    print("="*60)
    print(f"Sentence Extraction Started: {datetime.now()}")
    print("="*60)
    print(f"Input TSV: {tsv_path}")
    print(f"Output file: {output_path}")
    print(f"Target sentences: {num_sentences}")
    print(f"Shuffle: {shuffle}")
    if shuffle:
        print(f"Random seed: {seed}")
    print("="*60)
    
    sentences = []
    sentence_lengths = []
    
    try:
        # Read all sentences first
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                # The sentence text is in the last column (index -1)
                if len(row) > 0:
                    sentence = row[-1].strip()
                    if sentence:  # Only add non-empty sentences
                        sentences.append(sentence)
        
        print(f"Total sentences available in TSV: {len(sentences)}")
        
        # Shuffle if requested (for diversity)
        if shuffle:
            random.seed(seed)
            random.shuffle(sentences)
            print(f"Sentences shuffled with seed={seed}")
        
        # Take requested number of sentences
        sentences = sentences[:num_sentences]
        
        # Compute statistics
        for sent in sentences:
            sentence_lengths.append(len(sent))
        
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        min_length = min(sentence_lengths) if sentence_lengths else 0
        max_length = max(sentence_lengths) if sentence_lengths else 0
        
        # Write sentences to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        
        print("\n" + "="*60)
        print("EXTRACTION STATISTICS")
        print("="*60)
        print(f"Sentences extracted: {len(sentences)}")
        print(f"Average length: {avg_length:.1f} characters")
        print(f"Min length: {min_length} characters")
        print(f"Max length: {max_length} characters")
        print(f"Output saved to: {output_path}")
        print("="*60)
        print(f"Extraction Completed: {datetime.now()}")
        print("="*60)
        
        return len(sentences)
        
    except FileNotFoundError:
        print(f"ERROR: TSV file not found at {tsv_path}")
        raise
    except Exception as e:
        print(f"ERROR extracting sentences: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract sentences from TSV file for RL training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tsv_path', 
                        default='/info/corpus/Blizzard2023_segmented/segmented/NEB_train/reference_24khz/NEB_test_clean.tsv',
                        help='Path to input TSV file')
    parser.add_argument('--output_path',
                        default='/info/raid-etu/m2/s2405959/VO2/Agent/DB/sentences.txt',
                        help='Path to output text file')
    parser.add_argument('--num_sentences', type=int, default=1000,
                        help='Number of sentences to extract')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle sentences before extraction (for diversity)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    extract_sentences(args.tsv_path, args.output_path, args.num_sentences, 
                     args.shuffle, args.seed)
