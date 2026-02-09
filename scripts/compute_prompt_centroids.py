#!/usr/bin/env python3
"""
Compute k-means centroids from prompt embeddings for guided exploration.

This script pre-computes cluster centroids from the FAISS index to enable
manifold-aware exploration instead of random hypersphere sampling.

Usage:
    python scripts/compute_prompt_centroids.py --n_clusters 64 --output model/prompt_centroids.npy
"""

import numpy as np
import faiss
import argparse
import os
from sklearn.cluster import KMeans

# Configuration
DEFAULT_VECTORS_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256"


def extract_vectors_from_faiss(index_path):
    """Extract all vectors from a FAISS index."""
    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    
    n_vectors = index.ntotal
    dim = index.d
    print(f"Index contains {n_vectors} vectors of dimension {dim}")
    
    # Reconstruct all vectors
    vectors = np.zeros((n_vectors, dim), dtype=np.float32)
    for i in range(n_vectors):
        vectors[i] = index.reconstruct(i)
    
    return vectors


def compute_centroids(vectors, n_clusters=64, random_state=42):
    """Compute k-means centroids from vectors."""
    print(f"Computing {n_clusters} k-means centroids...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
        verbose=1
    )
    
    kmeans.fit(vectors)
    
    centroids = kmeans.cluster_centers_
    
    # L2 normalize centroids to match FAISS index normalization
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_normalized = centroids / (norms + 1e-8)
    
    print(f"Computed {n_clusters} centroids with shape {centroids_normalized.shape}")
    print(f"Inertia (sum of squared distances): {kmeans.inertia_:.4f}")
    
    # Compute cluster statistics
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Cluster size statistics: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    return centroids_normalized, kmeans


def main():
    parser = argparse.ArgumentParser(description="Compute prompt manifold centroids for guided exploration")
    parser.add_argument("--vectors_dir", type=str, default=DEFAULT_VECTORS_DIR,
                        help="Directory containing FAISS index")
    parser.add_argument("--n_clusters", type=int, default=64,
                        help="Number of k-means clusters (default: 64)")
    parser.add_argument("--output", type=str, default="model/prompt_centroids.npy",
                        help="Output path for centroids")
    
    args = parser.parse_args()
    
    # Paths
    index_path = os.path.join(args.vectors_dir, "prompts.index")
    
    if not os.path.exists(index_path):
        print(f"Error: FAISS index not found at {index_path}")
        return 1
    
    # Extract vectors
    vectors = extract_vectors_from_faiss(index_path)
    
    # Compute centroids
    centroids, kmeans = compute_centroids(vectors, n_clusters=args.n_clusters)
    
    # Save centroids
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, centroids)
    print(f"Centroids saved to {output_path}")
    
    # Also save as PyTorch tensor for direct loading
    import torch
    torch.save(torch.from_numpy(centroids).float(), output_path.replace('.npy', '.pt'))
    print(f"PyTorch tensor saved to {output_path.replace('.npy', '.pt')}")
    
    return 0


if __name__ == "__main__":
    exit(main())
