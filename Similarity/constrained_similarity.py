#!/usr/bin/env python3
"""
Constrained similarity search: select best prompt ONLY from a provided list.

Given a 256-d query vector and a list of 10 prompt texts, this script:
- maps prompt texts to global indices using the metadata
- reconstructs vectors from the FAISS index
- computes cosine similarity to the query
- returns the best prompt with metadata (prompt_wav, prompt_transcription)
"""

import argparse
import json
import os
import pickle

import faiss
import numpy as np

# --- CONFIGURATION ---
BASE_VECTORS_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256"
INDEX_PATH = os.path.join(BASE_VECTORS_DIR, "prompts.index")
META_PATH = os.path.join(BASE_VECTORS_DIR, "prompts_metadata.pkl")


def load_database():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def build_prompt_lookup(metadata):
    lookup = {}
    for idx, item in enumerate(metadata):
        text = item.get("prompt_transcription", "")
        if text:
            lookup.setdefault(text, []).append(idx)
    return lookup


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def parse_prompts(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        ordered = []
        for i in range(1, 11):
            key = f"prompt_{i}"
            if key not in obj:
                raise ValueError(f"Missing {key} in prompts object")
            ordered.append(obj[key])
        return ordered
    raise ValueError("Prompts must be a list or dict")


def main():
    parser = argparse.ArgumentParser(description="Constrained similarity search")
    parser.add_argument("--vector", type=str, required=True, help="Path to input 256-d vector (.npy)")
    parser.add_argument("--prompts_json", type=str, required=True, help="Path to prompts JSON (list or dict)")
    parser.add_argument("--output", type=str, required=True, help="Path to save JSON result")
    parser.add_argument("--output_vector", type=str, default=None, help="Path to save retrieved vector (.npy)")
    args = parser.parse_args()

    query = np.load(args.vector)
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    query = normalize(query.astype(np.float32))[0]

    with open(args.prompts_json, "r", encoding="utf-8") as f:
        prompts_obj = json.load(f)
    prompts = parse_prompts(prompts_obj)

    if len(prompts) != 10:
        raise ValueError(f"Expected 10 prompts, got {len(prompts)}")

    index, metadata = load_database()
    lookup = build_prompt_lookup(metadata)

    prompt_indices = []
    prompt_vectors = []
    for p in prompts:
        if p not in lookup:
            raise ValueError(f"Prompt text not found in metadata: {p}")
        idx = lookup[p][0]
        vec = index.reconstruct(int(idx)).astype(np.float32)
        vec = normalize(vec)
        prompt_indices.append(int(idx))
        prompt_vectors.append(vec)

    prompt_vectors = np.vstack(prompt_vectors)
    similarities = np.dot(prompt_vectors, query)
    best_i = int(np.argmax(similarities))
    best_idx = prompt_indices[best_i]
    best_meta = metadata[best_idx]

    output_data = {
        "index": int(best_idx),
        "cosine_similarity": float(similarities[best_i]),
        "prompt_wav": best_meta.get("prompt_wav", ""),
        "prompt_transcription": best_meta.get("prompt_transcription", ""),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    if args.output_vector:
        np.save(args.output_vector, prompt_vectors[best_i])


if __name__ == "__main__":
    main()