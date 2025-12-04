import faiss
import pickle
import numpy as np
import os
import argparse
import torch

BASE_VECTORS_DIR = "/info/raid-etu/m2/s2405959/VO2/Agent/DB/vectors"

INDEX_PATH = os.path.join(BASE_VECTORS_DIR, "prompts.index")
META_PATH = os.path.join(BASE_VECTORS_DIR, "prompts_metadata.pkl")

def load_database():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)

    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")
    with open(META_PATH, 'rb') as f:
        metadata = pickle.load(f)

    return index, metadata

def find_best_match(query_vector_100, index, metadata):
    query = query_vector_100.astype(np.float32)

    if len(query.shape) == 1:
        query = query.reshape(1, -1)

    faiss.normalize_L2(query)

    D, I = index.search(query, 1)

    best_index = I[0][0]
    l2_distance = D[0][0]

    similarity_score = 1 - (l2_distance / 2)

    result_meta = metadata[best_index]

    try:
        retrieved_vector = index.reconstruct(int(best_index))
    except Exception as e:
        retrieved_vector = np.zeros(100)
        print(f"Warning: Could not reconstruct vector from index. Error: {e}")

    return result_meta, best_index, similarity_score, l2_distance, retrieved_vector

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description="Find best match for a given vector.")
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
            query_vector = np.random.randn(1, 100)

        result, idx_found, score, dist, retrieved_vec = find_best_match(query_vector, idx, meta)

        output_data = {
            "index": int(idx_found),
            "l2_distance": float(dist),
            "cosine_similarity": float(score),
            "prompt_wav": result.get('prompt_wav', ''),
            "prompt_transcription": result.get('prompt_transcription', '')
        }

        print(f"Result Found (Index {idx_found})")
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
