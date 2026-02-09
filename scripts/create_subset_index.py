#!/usr/bin/env python3
"""
Create a subset FAISS index for constrained training.

This script creates a smaller FAISS index containing only a subset of the 
original embeddings. This is useful for:
1. Reducing the action space in RL training
2. Faster convergence by limiting exploration
3. Curriculum learning (start with subset, expand later)

Usage:
    python create_subset_index.py --subset_size 5000 --output_dir /path/to/output
"""

import os
import argparse
import pickle
import numpy as np
import faiss
from tqdm import tqdm

# Default paths
DEFAULT_SOURCE_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256"
DEFAULT_OUTPUT_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256_subset"


def create_subset_index(source_dir, output_dir, subset_size=5000, selection_method='first', seed=42):
    """
    Create a subset FAISS index from the original full index.
    
    Args:
        source_dir: Directory containing the original FAISS index and metadata
        output_dir: Directory to save the subset index
        subset_size: Number of embeddings to include in the subset
        selection_method: How to select the subset ('first', 'random', 'stratified')
        seed: Random seed for reproducibility
        
    Returns:
        Path to the created subset index
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
    
    # Validate subset size
    if subset_size > total_vectors:
        print(f"Warning: Requested subset size ({subset_size}) > total vectors ({total_vectors})")
        print(f"Using all {total_vectors} vectors")
        subset_size = total_vectors
    
    # Select indices based on method
    print(f"Selecting {subset_size} vectors using '{selection_method}' method...")
    
    if selection_method == 'first':
        # Select first N vectors (deterministic, preserves original order)
        selected_indices = np.arange(subset_size)
        
    elif selection_method == 'random':
        # Randomly select N vectors
        selected_indices = np.random.choice(total_vectors, size=subset_size, replace=False)
        selected_indices = np.sort(selected_indices)  # Sort for efficient reconstruction
        
    elif selection_method == 'stratified':
        # Select vectors evenly distributed across the index
        # This ensures coverage across different parts of the embedding space
        step = total_vectors / subset_size
        selected_indices = np.array([int(i * step) for i in range(subset_size)])
        
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    print(f"Selected indices range: [{selected_indices[0]}, {selected_indices[-1]}]")
    
    # Extract vectors from original index
    print("Extracting vectors from original index...")
    subset_vectors = np.zeros((subset_size, 256), dtype=np.float32)
    
    for i, idx in enumerate(tqdm(selected_indices, desc="Reconstructing vectors")):
        subset_vectors[i] = original_index.reconstruct(int(idx))
    
    # Create subset metadata
    print("Creating subset metadata...")
    subset_metadata = [original_metadata[int(idx)] for idx in selected_indices]
    
    # Add original index mapping to metadata (important for tracking)
    for i, idx in enumerate(selected_indices):
        subset_metadata[i]['original_index'] = int(idx)
        subset_metadata[i]['subset_index'] = i
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create new FAISS index
    print("Creating subset FAISS index...")
    d = 256  # Dimension
    subset_index = faiss.IndexFlatL2(d)
    
    # Vectors should already be normalized from original index
    # But let's ensure they are normalized
    faiss.normalize_L2(subset_vectors)
    subset_index.add(subset_vectors)
    
    print(f"Subset index contains {subset_index.ntotal} vectors")
    
    # Save subset index
    subset_index_path = os.path.join(output_dir, "prompts.index")
    print(f"Saving subset index to {subset_index_path}...")
    faiss.write_index(subset_index, subset_index_path)
    
    # Save subset metadata
    subset_meta_path = os.path.join(output_dir, "prompts_metadata.pkl")
    print(f"Saving subset metadata to {subset_meta_path}...")
    with open(subset_meta_path, 'wb') as f:
        pickle.dump(subset_metadata, f)
    
    # Save index mapping for reference
    mapping_path = os.path.join(output_dir, "index_mapping.pkl")
    mapping = {
        'subset_to_original': {i: int(idx) for i, idx in enumerate(selected_indices)},
        'original_to_subset': {int(idx): i for i, idx in enumerate(selected_indices)},
        'selection_method': selection_method,
        'subset_size': subset_size,
        'original_size': total_vectors,
        'seed': seed
    }
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"Saved index mapping to {mapping_path}")
    
    # Save info file for easy reference
    info_path = os.path.join(output_dir, "subset_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Subset FAISS Index Information\n")
        f.write(f"==============================\n")
        f.write(f"Source directory: {source_dir}\n")
        f.write(f"Subset size: {subset_size}\n")
        f.write(f"Original size: {total_vectors}\n")
        f.write(f"Selection method: {selection_method}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Dimension: {d}\n")
        f.write(f"\nFiles created:\n")
        f.write(f"  - prompts.index (FAISS index)\n")
        f.write(f"  - prompts_metadata.pkl (metadata)\n")
        f.write(f"  - index_mapping.pkl (subset <-> original mapping)\n")
    
    print(f"\nSubset index created successfully!")
    print(f"Output directory: {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Create a subset FAISS index for constrained training")
    parser.add_argument("--source_dir", type=str, default=DEFAULT_SOURCE_DIR,
                        help="Directory containing the original FAISS index")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save the subset index")
    parser.add_argument("--subset_size", type=int, default=5000,
                        help="Number of embeddings to include in subset")
    parser.add_argument("--selection_method", type=str, default='first',
                        choices=['first', 'random', 'stratified'],
                        help="Method to select subset: 'first' (first N), 'random', 'stratified' (evenly spaced)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    create_subset_index(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        selection_method=args.selection_method,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
