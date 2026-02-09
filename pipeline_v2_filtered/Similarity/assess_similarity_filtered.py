"""
Similarity search module for FILTERED pipeline (min 6 words).

This is an ISOLATED copy of the similarity search module that uses
the FILTERED 6-word minimum embeddings index.

This module is completely independent from the main pipeline.
Changes here do not affect the baseline pipeline.
"""

import faiss
import pickle
import numpy as np
import os
import argparse
import torch

# --- CONFIGURATION ---
# FILTERED INDEX: Only sentences with >= 6 words (both prompt and text)
FILTERED_VECTORS_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256_filtered_6words"

# Fallback to unfiltered if filtered not available
FALLBACK_VECTORS_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256"

def get_index_paths():
    """Get the appropriate index and metadata paths for filtered embeddings."""
    if os.path.exists(os.path.join(FILTERED_VECTORS_DIR, "prompts.index")):
        base_dir = FILTERED_VECTORS_DIR
        print(f"Using FILTERED index (6+ words, 5K prompts) from {base_dir}")
    else:
        base_dir = FALLBACK_VECTORS_DIR
        print(f"WARNING: Filtered index not found! Falling back to FULL index from {base_dir}")
        print(f"Run: python scripts/create_filtered_subset_index.py to generate filtered index")
    
    index_path = os.path.join(base_dir, "prompts.index")
    meta_path = os.path.join(base_dir, "prompts_metadata.pkl")
    
    return index_path, meta_path, base_dir

def load_database():
    """
    Loads the FILTERED FAISS index and Pickle metadata into memory.
    
    Returns:
        index: FAISS index
        metadata: List of metadata dictionaries
    """
    index_path, meta_path, _ = get_index_paths()
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    index = faiss.read_index(index_path)
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded FILTERED index with {index.ntotal} vectors")
    return index, metadata

def find_best_match(query_vector_256, index, metadata):
    """
    Takes a vector (1, 256) and returns the corresponding reference.
    Uses Cosine Similarity (via L2 normalization + Inner Product search).
    """
    # 1. Conversion and Normalization
    query = query_vector_256.astype(np.float32)
    
    # Reshape if necessary
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
        
    # Normalize the query vector to unit length
    faiss.normalize_L2(query)
    
    D, I = index.search(query, 1)
    
    best_index = I[0][0]
    l2_distance = D[0][0]
    
    # Convert L2 distance to Cosine Similarity
    # Relation: L2^2 = 2 - 2*cos_sim  =>  cos_sim = 1 - L2^2 / 2
    similarity_score = 1 - (l2_distance / 2)
    
    # 3. Retrieve actual info
    result_meta = metadata[best_index]
    
    # Retrieve the actual vector from the index
    try:
        retrieved_vector = index.reconstruct(int(best_index))
    except Exception as e:
        retrieved_vector = np.zeros(256)
        print(f"Warning: Could not reconstruct vector from index. Error: {e}")

    return result_meta, best_index, similarity_score, l2_distance, retrieved_vector

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description="Find best match for a given vector (FILTERED 6+ words index).")
    parser.add_argument("--vector", type=str, help="Path to the input vector file (.npy or .pt)")
    parser.add_argument("--output", type=str, help="Path to save the result as JSON.")
    parser.add_argument("--output_vector", type=str, help="Path to save the retrieved vector (.npy).")
    args = parser.parse_args()

    try:
        idx, meta = load_database()
        
        if args.vector:
            if args.vector.endswith('.npy'):
                query_vector = np.load(args.vector)
            elif args.vector.endswith('.pt'):
                query_vector = torch.load(args.vector)
                if isinstance(query_vector, torch.Tensor):
                    query_vector = query_vector.detach().cpu().numpy()
            else:
                try:
                    query_vector = np.loadtxt(args.vector)
                except:
                    raise ValueError("Unsupported file format. Please use .npy, .pt, or text file.")
            
            print(f"Loaded vector from {args.vector} with shape {query_vector.shape}")
        else:
            print("No vector file provided. Generating random vector for simulation.")
            query_vector = np.random.randn(1, 256) 
        
        result, idx_found, score, dist, retrieved_vec = find_best_match(query_vector, idx, meta)
        
        # Prepare result dictionary
        output_data = {
            "index": int(idx_found),
            "l2_distance": float(dist),
            "cosine_similarity": float(score),
            "prompt_wav": result.get('prompt_wav', ''),
            "prompt_transcription": result.get('prompt_transcription', ''),
            "prompt_word_count": result.get('prompt_word_count', 'N/A'),
            "text_word_count": result.get('text_word_count', 'N/A'),
            "original_index": result.get('original_index', 'N/A')
        }

        print(f"Result Found (Index {idx_found})")
        print(f"Cosine Similarity: {score:.4f}")
        print(f"WAV File: {result.get('prompt_wav', 'Key not found')}")
        print(f"Transcription: {result.get('prompt_transcription', 'Key not found')}")
        print(f"Word counts - Prompt: {result.get('prompt_word_count', '?')}, Text: {result.get('text_word_count', '?')}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"Result saved to {args.output}")
            
        if args.output_vector:
            np.save(args.output_vector, retrieved_vec)
            print(f"Retrieved vector saved to {args.output_vector}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
