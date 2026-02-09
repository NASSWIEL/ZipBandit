#!/usr/bin/env python3
"""
Validation script for the SonarAgent model.
Evaluates the agent on a held-out validation set WITHOUT exploration.
This provides a clean measure of the learned policy's performance.

Usage:
    python validate_agent.py --val_sentences_file ../DB/sentences_val.txt --model_path agent_model.pth
"""

import torch
import numpy as np
import argparse
import os
import sys
import json
import subprocess
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(CURRENT_DIR)

# Add paths for imports
if AGENT_DIR not in sys.path:
    sys.path.append(AGENT_DIR)

from model.agent_model import SonarAgent


def run_text_encoder(sentence, output_path):
    """Run SONAR text encoder on a sentence."""
    cmd = [
        "python3", os.path.join(CURRENT_DIR, "text_encoder.py"),
        "--sentence", sentence,
        "--output", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Text encoder error: {result.stderr}")
        return False
    return True


def run_similarity_search(vec_path, output_json, output_vec):
    """Run FAISS similarity search."""
    cmd = [
        "python3", os.path.join(AGENT_DIR, "Similarity", "asess_similarty.py"),
        "--vector", vec_path,
        "--output", output_json,
        "--output_vector", output_vec
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Similarity search error: {result.stderr}")
        return None
    
    with open(output_json, 'r') as f:
        return json.load(f)


def run_zipvoice(sim_result, target_text, output_audio_path):
    """Run ZipVoice TTS."""
    cmd = [
        "python3", os.path.join(AGENT_DIR, "generate_audio", "generate_with_zipVoice.py"),
        "--similarity_output_data", json.dumps(sim_result),
        "--target_text", target_text,
        "--output_wav", output_audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def calculate_cer(target_text, audio_path):
    """Calculate CER using Whisper ASR."""
    cmd = [
        "python3", os.path.join(AGENT_DIR, "assess_CER", "calculate_cer.py"),
        target_text, audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CER calculation error: {result.stderr}")
        return None
    
    # Parse CER from output
    for line in result.stdout.strip().split('\n'):
        if 'CER:' in line:
            try:
                cer = float(line.split(':')[1].strip())
                return cer
            except:
                pass
    return None


def validate_agent(model_path, val_sentences_file, max_sentences=10, epsilon=0.0, 
                   temp_dir=None, verbose=True):
    """
    Validate the agent on a held-out set without exploration.
    
    Args:
        model_path: Path to the trained model
        val_sentences_file: Path to validation sentences (one per line)
        max_sentences: Maximum number of sentences to validate
        epsilon: Exploration rate (should be 0.0 for validation)
        temp_dir: Temporary directory for intermediate files
        verbose: Whether to print detailed progress
        
    Returns:
        dict: Validation metrics including mean CER, reward, and similarity
    """
    device = torch.device("cpu")
    
    # Load model
    model = SonarAgent().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        if verbose:
            print(f"Loaded model from {model_path}")
    else:
        print(f"WARNING: Model not found at {model_path}, using random initialization")
    
    model.eval()  # Set to evaluation mode
    
    # Setup temp directory
    if temp_dir is None:
        temp_dir = os.path.join(AGENT_DIR, "temp_validation")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load validation sentences
    with open(val_sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    sentences = sentences[:max_sentences]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION: Evaluating on {len(sentences)} sentences (epsilon={epsilon})")
        print(f"{'='*60}\n")
    
    # Metrics collection
    cers = []
    rewards = []
    similarities = []
    
    for i, sentence in enumerate(sentences):
        if verbose:
            print(f"[{i+1}/{len(sentences)}] Validating: \"{sentence[:50]}...\"")
        
        try:
            # Step 1: Text encoding
            vec_1024_path = os.path.join(temp_dir, "val_vec_1024.npy")
            if not run_text_encoder(sentence, vec_1024_path):
                continue
            
            # Step 2: Agent prediction (NO exploration)
            input_vec = np.load(vec_1024_path)
            input_tensor = torch.from_numpy(input_vec).float()
            if len(input_tensor.shape) == 1:
                input_tensor = input_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output_tensor = model(input_tensor, add_noise=False, epsilon=0.0)
            
            vec_256_path = os.path.join(temp_dir, "val_vec_256.npy")
            np.save(vec_256_path, output_tensor.numpy())
            
            # Step 3: Similarity search
            sim_json_path = os.path.join(temp_dir, "val_sim_result.json")
            retrieved_vec_path = os.path.join(temp_dir, "val_retrieved_vec.npy")
            
            sim_result = run_similarity_search(vec_256_path, sim_json_path, retrieved_vec_path)
            if sim_result is None:
                continue
            
            similarity = sim_result.get('cosine_similarity', 0.0)
            similarities.append(similarity)
            
            # Step 4: Generate audio (optional - can be slow)
            # For quick validation, we can skip TTS and just use similarity as proxy
            # Uncomment below for full validation with CER
            
            # audio_path = os.path.join(temp_dir, f"val_audio_{i}.wav")
            # if run_zipvoice(sim_result, sentence, audio_path):
            #     cer = calculate_cer(sentence, audio_path)
            #     if cer is not None:
            #         cers.append(cer)
            #         rewards.append(max(0.0, 1.0 - cer))
            
            if verbose:
                print(f"    Similarity: {similarity:.4f}")
                
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Compute statistics
    results = {
        'num_validated': len(similarities),
        'mean_similarity': np.mean(similarities) if similarities else 0.0,
        'std_similarity': np.std(similarities) if similarities else 0.0,
        'min_similarity': np.min(similarities) if similarities else 0.0,
        'max_similarity': np.max(similarities) if similarities else 0.0,
    }
    
    if cers:
        results['mean_cer'] = np.mean(cers)
        results['std_cer'] = np.std(cers)
        results['mean_reward'] = np.mean(rewards)
        results['std_reward'] = np.std(rewards)
    
    if verbose:
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Sentences validated: {results['num_validated']}")
        print(f"Mean Similarity: {results['mean_similarity']:.4f} ± {results['std_similarity']:.4f}")
        print(f"Similarity Range: [{results['min_similarity']:.4f}, {results['max_similarity']:.4f}]")
        if 'mean_cer' in results:
            print(f"Mean CER: {results['mean_cer']:.4f} ± {results['std_cer']:.4f}")
            print(f"Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate Agent Model on held-out set")
    parser.add_argument("--val_sentences_file", type=str, required=True,
                        help="Path to validation sentences file (one per line)")
    parser.add_argument("--model_path", type=str, 
                        default=os.path.join(CURRENT_DIR, "agent_model.pth"),
                        help="Path to model weights")
    parser.add_argument("--max_sentences", type=int, default=10,
                        help="Maximum number of sentences to validate")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Exploration rate (should be 0.0 for validation)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save validation results as JSON")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.val_sentences_file):
        print(f"ERROR: Validation file not found: {args.val_sentences_file}")
        print("Create it by running: python DB/split_dataset.py")
        sys.exit(1)
    
    results = validate_agent(
        model_path=args.model_path,
        val_sentences_file=args.val_sentences_file,
        max_sentences=args.max_sentences,
        epsilon=args.epsilon
    )
    
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")
    
    return results


if __name__ == "__main__":
    main()
