#!/usr/bin/env python3
"""
Embedding Space Validation Script

This script validates whether the agent's output embedding space is well-aligned
with the FAISS database by checking if top-k retrievals for the same input are
consistent (low variance) or highly variable (problematic).

If top-k results vary significantly, it suggests the embedding space is not
well-structured for the task.
"""

import faiss
import pickle
import numpy as np
import os
import sys
import argparse
import torch
from typing import List, Tuple, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(CURRENT_DIR)

# Add paths for imports
if AGENT_DIR not in sys.path:
    sys.path.append(AGENT_DIR)

from model.agent_model import SonarAgent

# Vector database paths
VECTORS_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256"
INDEX_PATH = os.path.join(VECTORS_DIR, "prompts.index")
META_PATH = os.path.join(VECTORS_DIR, "prompts_metadata.pkl")


def load_database():
    """Load FAISS index and metadata."""
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata


def get_topk_results(query_vector: np.ndarray, index, metadata, k: int = 10) -> List[Dict]:
    """
    Get top-k nearest neighbors for a query vector.
    
    Returns:
        List of dicts with index, distance, similarity, and metadata
    """
    query = query_vector.astype(np.float32)
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    
    faiss.normalize_L2(query)
    
    D, I = index.search(query, k)
    
    results = []
    for i in range(k):
        idx = I[0][i]
        l2_dist = D[0][i]
        cos_sim = 1 - (l2_dist / 2)
        
        results.append({
            'rank': i + 1,
            'index': int(idx),
            'l2_distance': float(l2_dist),
            'cosine_similarity': float(cos_sim),
            'transcription': metadata[idx].get('prompt_transcription', '')[:50]
        })
    
    return results


def analyze_topk_stability(model, sentences: List[str], index, metadata, 
                          k: int = 10, n_runs: int = 5) -> Dict:
    """
    Analyze how stable top-k retrievals are across multiple runs with noise.
    
    For a well-learned embedding space:
    - Top-1 should be consistent (low variance in indices)
    - Top-k similarities should be well-separated (not all similar)
    
    Args:
        model: The SonarAgent model
        sentences: List of test sentences
        index: FAISS index
        metadata: Prompt metadata
        k: Number of top results to analyze
        n_runs: Number of runs with different noise samples
        
    Returns:
        Dict with stability metrics
    """
    print("\n" + "="*60)
    print("EMBEDDING SPACE VALIDATION")
    print("="*60)
    
    all_metrics = {
        'top1_consistency': [],  # How often top-1 is the same across runs
        'sim_spread': [],  # Difference between top-1 and top-k similarity
        'mean_top1_sim': [],
        'mean_topk_sim': []
    }
    
    model.eval()
    
    # Initialize text encoder once (with caching)
    from model.text_encoder import TextEncoder
    text_encoder = TextEncoder(use_cache=True)
    
    for sent_idx, sentence in enumerate(sentences[:10]):  # Limit to 10 sentences
        print(f"\n[{sent_idx+1}] Sentence: \"{sentence[:40]}...\"")
        
        # Encode text to 1024-dim
        try:
            text_vec = text_encoder.encode(sentence)
            text_tensor = torch.from_numpy(text_vec).float()
            if len(text_tensor.shape) == 1:
                text_tensor = text_tensor.unsqueeze(0)
        except Exception as e:
            print(f"    Skipping: {e}")
            continue
        
        top1_indices = []
        all_top1_sims = []
        all_topk_sims = []
        
        for run in range(n_runs):
            # Run with different noise
            with torch.no_grad():
                if run == 0:
                    # First run: no noise (deterministic)
                    output = model(text_tensor, add_noise=False, epsilon=0.0)
                else:
                    # Subsequent runs: with noise
                    output = model(text_tensor, add_noise=True, noise_std=0.15, epsilon=0.0)
            
            output_np = output.numpy()
            
            # Get top-k
            topk = get_topk_results(output_np, index, metadata, k)
            
            top1_indices.append(topk[0]['index'])
            all_top1_sims.append(topk[0]['cosine_similarity'])
            all_topk_sims.append(topk[-1]['cosine_similarity'])
        
        # Compute stability metrics
        unique_top1 = len(set(top1_indices))
        consistency = 1.0 - (unique_top1 - 1) / (n_runs - 1) if n_runs > 1 else 1.0
        sim_spread = np.mean(all_top1_sims) - np.mean(all_topk_sims)
        
        all_metrics['top1_consistency'].append(consistency)
        all_metrics['sim_spread'].append(sim_spread)
        all_metrics['mean_top1_sim'].append(np.mean(all_top1_sims))
        all_metrics['mean_topk_sim'].append(np.mean(all_topk_sims))
        
        print(f"    Top-1 Consistency: {consistency:.2f} (unique indices: {unique_top1}/{n_runs})")
        print(f"    Similarity Spread (top1 - top{k}): {sim_spread:.4f}")
        print(f"    Mean Top-1 Sim: {np.mean(all_top1_sims):.4f}")
        
        # Show top-3 for first run
        if sent_idx < 3:
            topk_display = get_topk_results(output.numpy(), index, metadata, 3)
            for r in topk_display:
                print(f"    Rank {r['rank']}: sim={r['cosine_similarity']:.4f}, \"{r['transcription']}\"")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    summary = {
        'mean_top1_consistency': np.mean(all_metrics['top1_consistency']),
        'mean_sim_spread': np.mean(all_metrics['sim_spread']),
        'mean_top1_similarity': np.mean(all_metrics['mean_top1_sim']),
        'mean_topk_similarity': np.mean(all_metrics['mean_topk_sim'])
    }
    
    print(f"Mean Top-1 Consistency: {summary['mean_top1_consistency']:.2f}")
    print(f"  - 1.0 = perfect (same result every time)")
    print(f"  - <0.5 = problematic (results vary significantly)")
    print(f"Mean Similarity Spread: {summary['mean_sim_spread']:.4f}")
    print(f"  - >0.1 = good (clear separation between ranks)")
    print(f"  - <0.05 = poor (all results equally similar)")
    print(f"Mean Top-1 Similarity: {summary['mean_top1_similarity']:.4f}")
    print(f"Mean Top-{k} Similarity: {summary['mean_topk_similarity']:.4f}")
    
    # Diagnose issues
    print("\n" + "-"*60)
    print("DIAGNOSIS:")
    if summary['mean_top1_consistency'] < 0.5:
        print("[WARNING] LOW CONSISTENCY: Model outputs are sensitive to noise.")
        print("    Recommendations: Reduce exploration noise, increase training stability.")
    if summary['mean_sim_spread'] < 0.05:
        print("[WARNING] LOW SPREAD: All top-k results have similar scores.")
        print("    This suggests the embedding space is not well-structured.")
        print("    Recommendations: Check if PCA reduction lost important information.")
    if summary['mean_top1_similarity'] < 0.25:
        print("[WARNING] LOW SIMILARITY: Agent outputs don't align well with database.")
        print("    Recommendations: The model may need more training or architecture changes.")
    
    if (summary['mean_top1_consistency'] >= 0.5 and 
        summary['mean_sim_spread'] >= 0.05 and 
        summary['mean_top1_similarity'] >= 0.25):
        print("[OK] Embedding space appears reasonably structured.")
    
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate embedding space quality")
    parser.add_argument("--model_path", type=str, 
                        default=os.path.join(AGENT_DIR, "model", "agent_model.pth"),
                        help="Path to model weights")
    parser.add_argument("--sentences_file", type=str,
                        default=os.path.join(AGENT_DIR, "DB", "sentences_val.txt"),
                        help="Path to sentences file")
    parser.add_argument("--k", type=int, default=10, help="Number of top results to analyze")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of runs per sentence")
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device("cpu")
    model = SonarAgent().to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"WARNING: Model not found, using random initialization")
    
    # Load database
    print("Loading FAISS index...")
    index, metadata = load_database()
    print(f"Loaded {index.ntotal} vectors from database")
    
    # Load sentences
    with open(args.sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(sentences)} validation sentences")
    
    # Run analysis
    results = analyze_topk_stability(model, sentences, index, metadata, 
                                     k=args.k, n_runs=args.n_runs)
    
    return results


if __name__ == "__main__":
    main()
