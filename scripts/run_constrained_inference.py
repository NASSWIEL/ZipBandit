#!/usr/bin/env python3
"""
Run constrained inference using the latest RL checkpoint.

This script enforces prompt selection from a fixed set of 10 prompts per target
provided in a JSON mapping file.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import faiss
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(CURRENT_DIR)

if AGENT_DIR not in sys.path:
    sys.path.append(AGENT_DIR)

from model.agent_model import SonarAgent, FAISS_INDEX_PATH, METADATA_PATH
from model.text_encoder import TextEncoder


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint by highest sentence number, else by mtime."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    candidates = list(ckpt_dir.glob("agent_model_sentence_*.pth"))
    if not candidates:
        return None

    def extract_num(path):
        match = re.search(r"agent_model_sentence_(\d+)\.pth", path.name)
        return int(match.group(1)) if match else -1

    candidates.sort(key=lambda p: (extract_num(p), p.stat().st_mtime), reverse=True)
    return str(candidates[0])


def load_prompt_index_and_metadata():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def build_prompt_lookup(metadata):
    lookup = {}
    for idx, item in enumerate(metadata):
        text = item.get("prompt_transcription", "")
        if not text:
            continue
        lookup.setdefault(text, []).append(idx)
    return lookup


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def parse_input_json(data):
    """Return list of (target_text, prompts_list)."""
    items = []
    if isinstance(data, dict):
        for target_text, prompts in data.items():
            items.append((target_text, prompts))
    elif isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            target_text = entry.get("target_text") or entry.get("target")
            prompts = entry.get("prompts") or entry.get("prompt_texts")
            if target_text is not None and prompts is not None:
                items.append((target_text, prompts))
    else:
        raise ValueError("Unsupported JSON structure. Expected dict or list.")
    return items


def main():
    parser = argparse.ArgumentParser(description="Constrained inference from target->prompts JSON")
    parser.add_argument("--input_json", required=True, help="Path to target->prompts JSON file")
    parser.add_argument("--output_json", default=os.path.join(AGENT_DIR, "logs_agent", "inference_constrained_results.json"),
                        help="Path to save JSON results")
    parser.add_argument("--output_csv", default=os.path.join(AGENT_DIR, "logs_agent", "inference_constrained_results.csv"),
                        help="Path to save CSV results")
    parser.add_argument("--model_path", default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--checkpoint_dir", default=os.path.join(AGENT_DIR, "checkpoint"),
                        help="Checkpoint directory to auto-select latest")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--max_targets", type=int, default=None, help="Limit number of targets")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = parse_input_json(data)
    if args.max_targets:
        items = items[:args.max_targets]

    if not items:
        raise ValueError("No target/prompt entries found in input JSON.")

    model_path = args.model_path
    if model_path is None:
        model_path = find_latest_checkpoint(args.checkpoint_dir)
        if model_path is None:
            model_path = os.path.join(AGENT_DIR, "model", "agent_model.pth")

    device = torch.device(args.device)

    model = SonarAgent().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    model.eval()

    text_encoder = TextEncoder(device=str(device))

    index, metadata = load_prompt_index_and_metadata()
    prompt_lookup = build_prompt_lookup(metadata)

    results = []

    for i, (target_text, prompts) in enumerate(items):
        if isinstance(prompts, dict):
            ordered = []
            for i in range(1, 11):
                key = f"prompt_{i}"
                if key not in prompts:
                    raise ValueError(f"Missing {key} for target: {target_text[:50]}")
                ordered.append(prompts[key])
            prompts = ordered
        if not isinstance(prompts, list):
            raise ValueError(f"Prompts for target must be a list or dict. Target: {target_text[:50]}")
        if len(prompts) != 10:
            raise ValueError(f"Expected 10 prompts for target, got {len(prompts)}. Target: {target_text[:50]}")

        prompt_indices = []
        for p in prompts:
            if p not in prompt_lookup:
                raise ValueError(f"Prompt text not found in metadata: {p}")
            prompt_indices.append(prompt_lookup[p][0])

        prompt_vectors = []
        for idx in prompt_indices:
            vec = index.reconstruct(int(idx))
            vec = normalize(vec.astype(np.float32))
            prompt_vectors.append(vec)

        prompt_vectors = np.vstack(prompt_vectors)

        target_emb = text_encoder.encode(target_text)
        target_tensor = torch.from_numpy(target_emb).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action_vec = model(target_tensor, add_noise=False, epsilon=0.0)
        action_vec = action_vec.detach().cpu().numpy().reshape(-1)
        action_vec = normalize(action_vec.astype(np.float32))

        similarities = np.dot(prompt_vectors, action_vec)
        best_idx = int(np.argmax(similarities))

        results.append({
            "target_text": target_text,
            "selected_prompt_text": prompts[best_idx],
            "selected_prompt_global_idx": int(prompt_indices[best_idx]),
            "selected_prompt_similarity": float(similarities[best_idx]),
            "all_prompt_texts": prompts,
            "all_prompt_global_indices": [int(x) for x in prompt_indices],
            "all_prompt_similarities": [float(x) for x in similarities.tolist()],
        })

        print(f"[{i+1}/{len(items)}] Selected prompt {best_idx+1}/10 (sim={similarities[best_idx]:.4f})")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"model_path": model_path, "results": results}, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8") as f:
        f.write("target_text\tselected_prompt_text\tselected_prompt_global_idx\tselected_prompt_similarity\n")
        for r in results:
            f.write(
                f"{r['target_text']}\t{r['selected_prompt_text']}\t{r['selected_prompt_global_idx']}\t{r['selected_prompt_similarity']:.6f}\n"
            )

    print(f"Results saved to: {args.output_json}")
    print(f"CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()