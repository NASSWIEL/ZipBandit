#!/usr/bin/env python3
"""
Create a FILTERED subset FAISS index for constrained training.

This script creates a FAISS index containing only embeddings for sentences
with at least MIN_WORDS words. This is useful for:
1. Linguistic filtering - longer sentences may provide better context
2. Comparing filtered vs unfiltered training
3. Curriculum learning with linguistically constrained data

Usage:
    python create_filtered_subset_index.py --subset_size 5000 --min_words 6 --output_dir /path/to/output
"""

import os
import argparse
import pickle
import numpy as np
import faiss
import pandas as pd
from tqdm import tqdm

# Default paths
DEFAULT_SOURCE_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256"
DEFAULT_TSV_PATH = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/reference_24khz/NEB_test_clean.tsv"
DEFAULT_OUTPUT_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256_filtered_6words"


def count_words(text):
    """Count the number of words in a text string."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    # Split by whitespace and count non-empty tokens
    words = text.strip().split()
    return len(words)


def load_tsv_data(tsv_path):
    """Load the TSV file containing sentence data."""
    print(f"Loading TSV data from {tsv_path}...")
    # Format: {wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}
    df = pd.read_csv(tsv_path, sep='\t', header=None, 
                     names=['wav_name', 'prompt_transcription', 'prompt_wav', 'text'], 
                     quoting=3)
    return df


def create_filtered_subset_index(source_dir, tsv_path, output_dir, subset_size=5000, 
                                  min_words=6, filter_field='both', seed=42):
    """
    Create a filtered subset FAISS index from the original full index.
    
    Only includes sentences where:
    - prompt_transcription has >= min_words words, OR
    - text (target) has >= min_words words, OR
    - both have >= min_words words (configurable via filter_field)
    
    Args:
        source_dir: Directory containing the original FAISS index and metadata
        tsv_path: Path to the TSV file with text data
        output_dir: Directory to save the filtered subset index
        subset_size: Maximum number of embeddings to include (first N that pass filter)
        min_words: Minimum number of words required
        filter_field: Which field to filter on ('prompt', 'text', 'both', 'either')
        seed: Random seed for reproducibility
        
    Returns:
        Path to the created filtered subset index
    """
    np.random.seed(seed)
    
    # Load original index and metadata
    source_index_path = os.path.join(source_dir, "prompts.index")
    source_meta_path = os.path.join(source_dir, "prompts_metadata.pkl")
    
    print(f"Loading original index from {source_index_path}...")
    if not os.path.exists(source_index_path):
        raise FileNotFoundError(f"Source index not found: {source_index_path}")
    
    original_index = faiss.read_index(source_index_path)
    total_vectors = original_index.ntotal
    print(f"Original index contains {total_vectors} vectors")
    
    print(f"Loading metadata from {source_meta_path}...")
    with open(source_meta_path, 'rb') as f:
        original_metadata = pickle.load(f)
    
    # Load TSV data for text filtering
    df = load_tsv_data(tsv_path)
    print(f"Loaded {len(df)} rows from TSV")
    
    if len(df) != total_vectors:
        print(f"WARNING: TSV rows ({len(df)}) != index vectors ({total_vectors})")
        print("Proceeding with min(rows, vectors)...")
    
    # Filter by word count
    print(f"\nFiltering sentences with >= {min_words} words (field: {filter_field})...")
    
    # Count words in each field
    df['prompt_word_count'] = df['prompt_transcription'].apply(count_words)
    df['text_word_count'] = df['text'].apply(count_words)
    
    # Apply filter based on filter_field
    if filter_field == 'prompt':
        filter_mask = df['prompt_word_count'] >= min_words
        filter_desc = "prompt_transcription"
    elif filter_field == 'text':
        filter_mask = df['text_word_count'] >= min_words
        filter_desc = "text (target)"
    elif filter_field == 'both':
        filter_mask = (df['prompt_word_count'] >= min_words) & (df['text_word_count'] >= min_words)
        filter_desc = "both prompt AND text"
    elif filter_field == 'either':
        filter_mask = (df['prompt_word_count'] >= min_words) | (df['text_word_count'] >= min_words)
        filter_desc = "prompt OR text"
    else:
        raise ValueError(f"Unknown filter_field: {filter_field}")
    
    # Get indices that pass the filter
    filtered_indices = df[filter_mask].index.tolist()
    
    print(f"Sentences passing filter ({filter_desc} >= {min_words} words): {len(filtered_indices)}")
    
    # Distribution statistics
    print(f"\nWord count statistics (before filtering):")
    print(f"  Prompt transcription: min={df['prompt_word_count'].min()}, "
          f"max={df['prompt_word_count'].max()}, mean={df['prompt_word_count'].mean():.1f}")
    print(f"  Target text: min={df['text_word_count'].min()}, "
          f"max={df['text_word_count'].max()}, mean={df['text_word_count'].mean():.1f}")
    
    # Select first N filtered indices
    if len(filtered_indices) < subset_size:
        print(f"\nWarning: Only {len(filtered_indices)} sentences pass the filter")
        print(f"Using all {len(filtered_indices)} (requested: {subset_size})")
        subset_size = len(filtered_indices)
    
    # Take the first subset_size sentences that pass the filter
    selected_indices = np.array(filtered_indices[:subset_size])
    
    print(f"\nSelected {subset_size} sentences (first {subset_size} that pass filter)")
    print(f"Selected indices range: [{selected_indices[0]}, {selected_indices[-1]}]")
    
    # Show some examples
    print(f"\nExample sentences from selected subset:")
    for i in [0, subset_size//2, subset_size-1]:
        idx = selected_indices[i]
        row = df.iloc[idx]
        print(f"  [{i}] idx={idx}: prompt({row['prompt_word_count']} words): '{row['prompt_transcription'][:60]}...'")
        print(f"       text({row['text_word_count']} words): '{str(row['text'])[:60]}...'")
    
    # Extract vectors from original index
    print("\nExtracting vectors from original index...")
    subset_vectors = np.zeros((subset_size, 256), dtype=np.float32)
    
    for i, idx in enumerate(tqdm(selected_indices, desc="Reconstructing vectors")):
        if idx < total_vectors:
            subset_vectors[i] = original_index.reconstruct(int(idx))
        else:
            print(f"Warning: Index {idx} out of bounds, using zero vector")
    
    # Create subset metadata
    print("Creating subset metadata...")
    subset_metadata = []
    for i, idx in enumerate(selected_indices):
        if idx < len(original_metadata):
            meta = original_metadata[int(idx)].copy()
        else:
            meta = {
                'wav_name': '',
                'prompt_transcription': '',
                'prompt_wav': ''
            }
        
        # Add original index mapping and filter info
        meta['original_index'] = int(idx)
        meta['subset_index'] = i
        meta['prompt_word_count'] = int(df.iloc[idx]['prompt_word_count']) if idx < len(df) else 0
        meta['text_word_count'] = int(df.iloc[idx]['text_word_count']) if idx < len(df) else 0
        
        subset_metadata.append(meta)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create new FAISS index
    print("\nCreating filtered subset FAISS index...")
    d = 256  # Dimension
    subset_index = faiss.IndexFlatL2(d)
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(subset_vectors)
    subset_index.add(subset_vectors)
    
    print(f"Filtered subset index contains {subset_index.ntotal} vectors")
    
    # Save subset index
    subset_index_path = os.path.join(output_dir, "prompts.index")
    print(f"Saving filtered subset index to {subset_index_path}...")
    faiss.write_index(subset_index, subset_index_path)
    
    # Save subset metadata
    subset_meta_path = os.path.join(output_dir, "prompts_metadata.pkl")
    print(f"Saving filtered subset metadata to {subset_meta_path}...")
    with open(subset_meta_path, 'wb') as f:
        pickle.dump(subset_metadata, f)
    
    # Save index mapping for reference
    mapping_path = os.path.join(output_dir, "index_mapping.pkl")
    mapping = {
        'subset_to_original': {i: int(idx) for i, idx in enumerate(selected_indices)},
        'original_to_subset': {int(idx): i for i, idx in enumerate(selected_indices)},
        'filter_field': filter_field,
        'min_words': min_words,
        'subset_size': subset_size,
        'original_size': total_vectors,
        'total_passing_filter': len(filtered_indices),
        'seed': seed
    }
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"Saved index mapping to {mapping_path}")
    
    # Save info file for easy reference
    info_path = os.path.join(output_dir, "subset_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Filtered Subset FAISS Index Information\n")
        f.write(f"========================================\n\n")
        f.write(f"Source directory: {source_dir}\n")
        f.write(f"TSV data file: {tsv_path}\n\n")
        f.write(f"FILTERING CRITERIA:\n")
        f.write(f"  - Minimum words: {min_words}\n")
        f.write(f"  - Filter field: {filter_field}\n")
        f.write(f"  - Total passing filter: {len(filtered_indices)}\n\n")
        f.write(f"SUBSET STATISTICS:\n")
        f.write(f"  - Subset size: {subset_size}\n")
        f.write(f"  - Original size: {total_vectors}\n")
        f.write(f"  - Selection: first {subset_size} sentences passing filter\n")
        f.write(f"  - Random seed: {seed}\n")
        f.write(f"  - Dimension: {d}\n\n")
        f.write(f"Files created:\n")
        f.write(f"  - prompts.index (FAISS index)\n")
        f.write(f"  - prompts_metadata.pkl (metadata with word counts)\n")
        f.write(f"  - index_mapping.pkl (subset <-> original mapping + filter info)\n")
    
    print(f"\nFiltered subset index created successfully!")
    print(f"Output directory: {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create a filtered subset FAISS index (min word count filter)")
    parser.add_argument("--source_dir", type=str, default=DEFAULT_SOURCE_DIR,
                        help="Directory containing the original FAISS index")
    parser.add_argument("--tsv_path", type=str, default=DEFAULT_TSV_PATH,
                        help="Path to TSV file with text data")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save the filtered subset index")
    parser.add_argument("--subset_size", type=int, default=5000,
                        help="Maximum number of embeddings to include")
    parser.add_argument("--min_words", type=int, default=6,
                        help="Minimum number of words required in sentence")
    parser.add_argument("--filter_field", type=str, default='both',
                        choices=['prompt', 'text', 'both', 'either'],
                        help="Which field to filter: 'prompt', 'text', 'both', 'either'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    create_filtered_subset_index(
        source_dir=args.source_dir,
        tsv_path=args.tsv_path,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        min_words=args.min_words,
        filter_field=args.filter_field,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
