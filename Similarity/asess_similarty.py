import faiss
import pickle
import numpy as np
import os
import argparse
import torch

# --- CONFIGURATION ---
# Using 256-dim vectors directory with FULL 63K prompts dataset
BASE_VECTORS_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256"

# Support for subset index (5K embeddings for faster RL training)
# The subset index reduces the action space from 63K to 5K prompts
SUBSET_VECTORS_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256_subset"

# Default to subset for training (can be overridden via --use_full_index)
USE_SUBSET_BY_DEFAULT = True

def get_index_paths(use_subset=True):
    """Get the appropriate index and metadata paths based on configuration."""
    if use_subset and os.path.exists(os.path.join(SUBSET_VECTORS_DIR, "prompts.index")):
        base_dir = SUBSET_VECTORS_DIR
        print(f"Using SUBSET index (5K prompts) from {base_dir}")
    else:
        base_dir = BASE_VECTORS_DIR
        if use_subset:
            print(f"Warning: Subset index not found, falling back to FULL index")
        print(f"Using FULL index (63K prompts) from {base_dir}")
    
    index_path = os.path.join(base_dir, "prompts.index")
    meta_path = os.path.join(base_dir, "prompts_metadata.pkl")
    
    return index_path, meta_path, base_dir

# Legacy paths for backward compatibility
INDEX_PATH = os.path.join(BASE_VECTORS_DIR, "prompts.index")
META_PATH = os.path.join(BASE_VECTORS_DIR, "prompts_metadata.pkl")

def load_database(use_subset=None):
    """
    Loads the FAISS index and Pickle metadata into memory.
    
    Args:
        use_subset: If True, use subset index. If False, use full index.
                   If None, use the default (USE_SUBSET_BY_DEFAULT).
    
    Returns:
        index: FAISS index
        metadata: List of metadata dictionaries
    """
    if use_subset is None:
        use_subset = USE_SUBSET_BY_DEFAULT
    
    index_path, meta_path, _ = get_index_paths(use_subset)
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    index = faiss.read_index(index_path)
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded index with {index.ntotal} vectors")
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
    # Since the database vectors are already normalized (during creation),
    # the dot product (Inner Product) of two normalized vectors is equal to their Cosine Similarity.
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
        # Ensure index is a standard Python int for FAISS SWIG wrapper
        retrieved_vector = index.reconstruct(int(best_index))
    except Exception as e:
        # Fallback if reconstruct is not supported
        retrieved_vector = np.zeros(256)
        print(f"Warning: Could not reconstruct vector from index. Error: {e}")

    return result_meta, best_index, similarity_score, l2_distance, retrieved_vector

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description="Find best match for a given vector.")
    parser.add_argument("--vector", type=str, help="Path to the input vector file (.npy or .pt)")
    parser.add_argument("--output", type=str, help="Path to save the result as JSON.")
    parser.add_argument("--output_vector", type=str, help="Path to save the retrieved vector (.npy).")
    parser.add_argument("--use_full_index", action="store_true", 
                        help="Use the full 63K index instead of the subset 5K index")
    args = parser.parse_args()

    try:
        # Determine which index to use
        use_subset = not args.use_full_index
        idx, meta = load_database(use_subset=use_subset)
        
        if args.vector:
            if args.vector.endswith('.npy'):
                query_vector = np.load(args.vector)
            elif args.vector.endswith('.pt'):
                query_vector = torch.load(args.vector)
                if isinstance(query_vector, torch.Tensor):
                    query_vector = query_vector.detach().cpu().numpy()
            else:
                # Fallback: try loading as text
                try:
                    query_vector = np.loadtxt(args.vector)
                except:
                    raise ValueError("Unsupported file format. Please use .npy, .pt, or text file.")
            
            print(f"Loaded vector from {args.vector} with shape {query_vector.shape}")
        else:
            print("No vector file provided. Generating random vector for simulation.")
            # Simulation: Agent generates a random 256-dim vector
            query_vector = np.random.randn(1, 256) 
        
        result, idx_found, score, dist, retrieved_vec = find_best_match(query_vector, idx, meta)
        
        # Prepare result dictionary
        output_data = {
            "index": int(idx_found),
            "l2_distance": float(dist),
            "cosine_similarity": float(score),
            "prompt_wav": result.get('prompt_wav', ''),
            "prompt_transcription": result.get('prompt_transcription', '')
        }

        print(f"Result Found (Index {idx_found})")
        # print(f"L2 Distance: {dist:.4f}")
        print(f"Cosine Similarity: {score:.4f}")
        print(f"WAV File: {result.get('prompt_wav', 'Key not found')}")
        print(f"Transcription: {result.get('prompt_transcription', 'Key not found')}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"Result saved to {args.output}")
            
        if args.output_vector:
            np.save(args.output_vector, retrieved_vec)
            print(f"Retrieved vector saved to {args.output_vector}")
        
    except Exception as e:
        print(f"Error: {e}")
