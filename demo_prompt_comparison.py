#!/usr/bin/env python3
"""
Demo Script: Prompt Selection Impact on CER

This script demonstrates how different prompt selections affect the Character Error Rate (CER)
for the same target text. It samples 5 random prompts from the available prompt database
and generates audio for each, comparing their CER values.

The goal is to showcase that prompt selection significantly impacts voice cloning quality.
"""

import os
import sys
import json
import pickle
import argparse
import subprocess
import numpy as np
import torch
import faiss
from datetime import datetime

# Add parent directories to path
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

# Configuration
BASE_VECTORS_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256"
INDEX_PATH = os.path.join(BASE_VECTORS_DIR, "prompts.index")
META_PATH = os.path.join(BASE_VECTORS_DIR, "prompts_metadata.pkl")
TEMP_DIR = os.path.join(AGENT_DIR, "temp_demo")
RESULTS_DIR = os.path.join(AGENT_DIR, "demo_results")

# Representative target texts for demo (interesting, clear, meaningful)
DEMO_TEXTS = [
    "La science et la technologie transforment notre monde à une vitesse vertigineuse.",
    "L'intelligence artificielle ouvre de nouvelles perspectives fascinantes pour l'humanité.",
    "La musique est un langage universel qui transcende toutes les frontières.",
    "Le développement durable est essentiel pour préserver notre planète.",
    "L'éducation est la clé du progrès et de l'épanouissement personnel.",
]


def load_prompt_database():
    """Load the FAISS index and metadata for prompts."""
    print(f"Loading prompt database from {BASE_VECTORS_DIR}...")
    
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")
    
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")
    
    index = faiss.read_index(INDEX_PATH)
    
    with open(META_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded database with {index.ntotal} prompts")
    return index, metadata


def sample_random_prompts(index, metadata, n_samples=5):
    """Sample n random prompts from the database."""
    total_prompts = index.ntotal
    
    # Sample random indices
    random_indices = np.random.choice(total_prompts, size=n_samples, replace=False)
    
    sampled_prompts = []
    for idx in random_indices:
        # Retrieve vector from index
        try:
            vector = index.reconstruct(int(idx))
            meta = metadata[int(idx)]
            sampled_prompts.append({
                'index': int(idx),
                'vector': vector,
                'metadata': meta,
                'prompt_wav': meta.get('prompt_wav', 'N/A'),
                'prompt_transcription': meta.get('prompt_transcription', 'N/A')
            })
        except Exception as e:
            print(f"Warning: Could not retrieve prompt {idx}: {e}")
            continue
    
    return sampled_prompts


def encode_target_text(target_text):
    """Encode target text using the text encoder."""
    output_path = os.path.join(TEMP_DIR, "target_vec_1024.npy")
    
    cmd = [
        "python3",
        os.path.join(AGENT_DIR, "model", "text_encoder.py"),
        "--sentence", target_text,
        "--output", output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Text encoding failed: {result.stderr}")
    
    if not os.path.exists(output_path):
        raise RuntimeError("Text encoding did not produce output file")
    
    return np.load(output_path)


def generate_audio_with_prompt(target_text, prompt_vector, prompt_metadata, iteration):
    """Generate audio using a specific prompt vector."""
    print(f"\n  [Step 1/{iteration}] Generating audio with prompt {iteration}...")
    
    # Save the prompt vector temporarily
    prompt_vec_path = os.path.join(TEMP_DIR, f"prompt_vec_{iteration}.npy")
    np.save(prompt_vec_path, prompt_vector)
    
    # Create similarity output JSON that mimics the normal pipeline
    # generate_with_zipVoice.py expects 'prompt_wav' and 'prompt_transcription'
    similarity_output = {
        'cosine_similarity': 1.0,  # Perfect match since we're using the exact prompt
        'nearest_idx': int(prompt_metadata.get('original_index', -1)),
        'prompt_wav': prompt_metadata.get('prompt_wav', ''),
        'prompt_transcription': prompt_metadata.get('prompt_transcription', ''),
        'l2_distance': 0.0
    }
    
    sim_output_path = os.path.join(TEMP_DIR, f"similarity_result_{iteration}.json")
    with open(sim_output_path, 'w') as f:
        json.dump(similarity_output, f, indent=2)
    
    # Generate audio
    audio_path_file = os.path.join(TEMP_DIR, f"generated_audio_path_{iteration}.txt")
    
    cmd = [
        "python3",
        os.path.join(AGENT_DIR, "generate_audio", "generate_with_zipVoice.py"),
        "--similarity_output", sim_output_path,
        "--target_text", target_text,
        "--output_path_file", audio_path_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Audio generation failed: {result.stderr}")
    
    if not os.path.exists(audio_path_file):
        raise RuntimeError("Audio generation did not produce output file")
    
    with open(audio_path_file, 'r') as f:
        audio_path = f.read().strip()
    
    return audio_path


def calculate_cer(target_text, audio_path):
    """Calculate CER for generated audio."""
    cer_output = os.path.join(TEMP_DIR, "cer_temp.txt")
    
    cmd = [
        "python3",
        os.path.join(AGENT_DIR, "assess_CER", "calculate_cer.py"),
        target_text,
        audio_path,
        "--output_cer", cer_output
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"CER calculation failed: {result.stderr}")
    
    if not os.path.exists(cer_output):
        raise RuntimeError("CER calculation did not produce output file")
    
    with open(cer_output, 'r') as f:
        cer_value = float(f.read().strip())
    
    return cer_value


def run_demo(target_text, n_prompts=5, save_results=True):
    """Run the demo comparing different prompts for the same target text."""
    
    print("="*70)
    print("DEMO: Impact of Prompt Selection on CER")
    print("="*70)
    print(f"\nTarget text: \"{target_text}\"")
    print(f"Number of prompts to test: {n_prompts}\n")
    
    # Create temporary directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load prompt database
    index, metadata = load_prompt_database()
    
    # Sample random prompts
    print(f"\nSampling {n_prompts} random prompts from database...")
    sampled_prompts = sample_random_prompts(index, metadata, n_samples=n_prompts)
    
    if len(sampled_prompts) < n_prompts:
        print(f"Warning: Could only sample {len(sampled_prompts)} prompts")
    
    # Display sampled prompts
    print("\nSampled Prompts:")
    for i, prompt in enumerate(sampled_prompts, 1):
        prompt_text = prompt.get('prompt_transcription', 'N/A')
        if len(prompt_text) > 80:
            prompt_text = prompt_text[:80] + "..."
        print(f"  {i}. Index {prompt['index']}: {prompt_text}")
    
    # Process each prompt
    results = []
    
    for i, prompt in enumerate(sampled_prompts, 1):
        print(f"\n{'-'*70}")
        print(f"Testing Prompt {i}/{len(sampled_prompts)}")
        print(f"{'-'*70}")
        
        try:
            # Generate audio with this prompt
            audio_path = generate_audio_with_prompt(
                target_text, 
                prompt['vector'], 
                prompt['metadata'], 
                i
            )
            print(f"  [Step 2/{i}] Audio generated: {audio_path}")
            
            # Calculate CER
            cer_value = calculate_cer(target_text, audio_path)
            print(f"  [Step 3/{i}] CER calculated: {cer_value:.4f}")
            
            # Store results
            result = {
                'prompt_index': prompt['index'],
                'prompt_transcription': prompt.get('prompt_transcription', 'N/A'),
                'prompt_wav': prompt.get('prompt_wav', 'N/A'),
                'generated_audio_path': audio_path,
                'cer': cer_value
            }
            results.append(result)
            
            print(f"  [OK] Prompt {i} completed successfully")
            
        except Exception as e:
            print(f"  [ERROR] Error processing prompt {i}: {e}")
            results.append({
                'prompt_index': prompt['index'],
                'prompt_transcription': prompt.get('prompt_transcription', 'N/A'),
                'error': str(e)
            })
    
    # Display summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nTarget text: \"{target_text}\"\n")
    
    successful_results = [r for r in results if 'cer' in r]
    
    if not successful_results:
        print("No successful results to display.")
        return results
    
    # Sort by CER (best to worst)
    successful_results.sort(key=lambda x: x['cer'])
    
    print(f"{'Rank':<6} {'Prompt Index':<15} {'CER':<10} {'Prompt Text'}")
    print("-"*70)
    
    for rank, result in enumerate(successful_results, 1):
        prompt_text = result.get('prompt_transcription', 'N/A')
        if len(prompt_text) > 40:
            prompt_text = prompt_text[:40] + "..."
        print(f"{rank:<6} {result['prompt_index']:<15} {result['cer']:<10.4f} {prompt_text}")
    
    # Calculate statistics
    cer_values = [r['cer'] for r in successful_results]
    min_cer = min(cer_values)
    max_cer = max(cer_values)
    mean_cer = np.mean(cer_values)
    std_cer = np.std(cer_values)
    
    print("\n" + "-"*70)
    print(f"Best CER:    {min_cer:.4f}")
    print(f"Worst CER:   {max_cer:.4f}")
    print(f"Mean CER:    {mean_cer:.4f}")
    print(f"Std Dev:     {std_cer:.4f}")
    print(f"Range:       {max_cer - min_cer:.4f}")
    print(f"Improvement: {((max_cer - min_cer) / max_cer * 100):.1f}% (best vs worst)")
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(RESULTS_DIR, f"demo_results_{timestamp}.json")
        
        full_results = {
            'target_text': target_text,
            'n_prompts': n_prompts,
            'timestamp': timestamp,
            'statistics': {
                'min_cer': float(min_cer),
                'max_cer': float(max_cer),
                'mean_cer': float(mean_cer),
                'std_cer': float(std_cer),
                'range': float(max_cer - min_cer),
                'improvement_percent': float((max_cer - min_cer) / max_cer * 100)
            },
            'results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Compare CER across different prompts for the same target text"
    )
    parser.add_argument(
        "--target_text",
        type=str,
        help="Target text to use for demo (if not provided, will use a preset)"
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=5,
        help="Number of random prompts to sample (default: 5)"
    )
    parser.add_argument(
        "--preset_index",
        type=int,
        choices=range(len(DEMO_TEXTS)),
        help=f"Use preset demo text (0-{len(DEMO_TEXTS)-1})"
    )
    parser.add_argument(
        "--list_presets",
        action="store_true",
        help="List available preset demo texts and exit"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results to file"
    )
    
    args = parser.parse_args()
    
    # List presets if requested
    if args.list_presets:
        print("\nAvailable preset demo texts:")
        for i, text in enumerate(DEMO_TEXTS):
            print(f"  {i}: {text}")
        return
    
    # Determine target text
    if args.target_text:
        target_text = args.target_text
    elif args.preset_index is not None:
        target_text = DEMO_TEXTS[args.preset_index]
    else:
        # Use first preset by default
        target_text = DEMO_TEXTS[0]
        print(f"No target text specified. Using preset: \"{target_text}\"")
    
    # Run demo
    try:
        results = run_demo(
            target_text=target_text,
            n_prompts=args.n_prompts,
            save_results=not args.no_save
        )
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
