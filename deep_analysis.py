#!/usr/bin/env python3
"""
Deep analysis of three RL training logs.
"""

import re
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_log_comprehensive(log_path):
    """Parse a log file and extract all possible metrics."""
    metrics = {
        'sentence_num': [],
        'avg_loss': [],
        'policy_loss': [],
        'contrastive_loss': [],
        'value_loss': [],
        'entropy': [],
        'reward': [],
        'cer': [],
        'epsilon': [],
        'target_text': [],
        'selected_prompt': [],
        'prompt_lengths': [],
        'target_lengths': [],
        'num_candidates': [],
    }
    
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    current_sentence = None
    current_epsilon = None
    current_target = None
    current_prompt = None
    
    for i, line in enumerate(lines):
        # Sentence number
        m = re.search(r'Processing Sentence (\d+)\s*/', line)
        if m:
            current_sentence = int(m.group(1))
        
        # Epsilon
        m = re.search(r'Epsilon \(exploration\): ([\d.]+)', line)
        if m:
            current_epsilon = float(m.group(1))
        
        # Target text
        m = re.search(r'Target text: (.+)', line)
        if m:
            current_target = m.group(1).strip()
        
        # Selected prompt
        m = re.search(r'Selected prompt[^:]*: (.+)', line)
        if not m:
            m = re.search(r'Best prompt[^:]*: (.+)', line)
        if m:
            current_prompt = m.group(1).strip()
        
        # CER
        m = re.search(r'^CER: ([\d.]+)', line)
        if m:
            cer_val = float(m.group(1))
            metrics['cer'].append(cer_val)
            if current_target:
                metrics['target_lengths'].append(len(current_target.split()))
                metrics['target_text'].append(current_target)
            if current_prompt:
                metrics['prompt_lengths'].append(len(current_prompt.split()))
                metrics['selected_prompt'].append(current_prompt)
        
        # Reward
        m = re.search(r'Sentence \d+ Reward: ([\d.]+)', line)
        if m:
            metrics['reward'].append(float(m.group(1)))
        
        # Training metrics  
        m = re.search(
            r'Training Complete\. Avg Loss: ([-\d.]+), Policy Loss: ([-\d.]+), '
            r'Contrastive: ([\d.]+), Value Loss: ([\d.]+), Entropy: ([\d.]+)',
            line
        )
        if m and current_sentence is not None:
            metrics['sentence_num'].append(current_sentence)
            metrics['avg_loss'].append(float(m.group(1)))
            metrics['policy_loss'].append(float(m.group(2)))
            metrics['contrastive_loss'].append(float(m.group(3)))
            metrics['value_loss'].append(float(m.group(4)))
            metrics['entropy'].append(float(m.group(5)))
            if current_epsilon is not None:
                metrics['epsilon'].append(current_epsilon)
        
        # Number of candidates
        m = re.search(r'(\d+) candidate prompts?', line)
        if m:
            metrics['num_candidates'].append(int(m.group(1)))
    
    return metrics


def compute_block_stats(values, block_size=100):
    """Compute statistics per block."""
    blocks = []
    for i in range(0, len(values), block_size):
        block = values[i:i+block_size]
        if len(block) > 0:
            blocks.append({
                'range': f"{i+1}-{i+len(block)}",
                'mean': np.mean(block),
                'std': np.std(block),
                'min': np.min(block),
                'max': np.max(block),
                'median': np.median(block),
                'count': len(block)
            })
    return blocks


def analyze_oscillation(values):
    """Analyze oscillation characteristics."""
    if len(values) < 3:
        return {}
    arr = np.array(values)
    diffs = np.diff(arr)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    
    # Running std with window
    window = min(20, len(arr) // 3)
    if window < 3:
        return {'sign_changes': sign_changes, 'total_points': len(arr)}
    
    running_std = []
    for i in range(len(arr) - window + 1):
        running_std.append(np.std(arr[i:i+window]))
    
    return {
        'sign_changes': sign_changes,
        'total_points': len(arr),
        'oscillation_rate': sign_changes / max(1, len(diffs) - 1),
        'mean_running_std': np.mean(running_std),
        'std_running_std': np.std(running_std),
        'early_volatility': np.mean(running_std[:len(running_std)//3]) if len(running_std) > 3 else 0,
        'late_volatility': np.mean(running_std[-len(running_std)//3:]) if len(running_std) > 3 else 0,
    }


def print_analysis(name, metrics):
    """Print comprehensive analysis for one experiment."""
    print(f"\n{'='*80}")
    print(f"  ANALYSIS: {name}")
    print(f"{'='*80}")
    
    n_training = len(metrics['sentence_num'])
    n_rewards = len(metrics['reward'])
    n_cer = len(metrics['cer'])
    
    print(f"\n--- GLOBAL OVERVIEW ---")
    print(f"  Training iterations: {n_training}")
    print(f"  Rewards collected: {n_rewards}")
    print(f"  CER measurements: {n_cer}")
    if metrics['epsilon']:
        print(f"  Epsilon range: {metrics['epsilon'][0]:.4f} -> {metrics['epsilon'][-1]:.4f}")
    if metrics['sentence_num']:
        print(f"  Sentence range: {metrics['sentence_num'][0]} -> {metrics['sentence_num'][-1]}")
    
    # ---- LOSS ANALYSIS ----
    print(f"\n--- LOSS ANALYSIS ---")
    if metrics['avg_loss']:
        avg = np.array(metrics['avg_loss'])
        print(f"  Avg Loss:  mean={np.mean(avg):.4f}, std={np.std(avg):.4f}, "
              f"min={np.min(avg):.4f}, max={np.max(avg):.4f}")
        print(f"  First 10 avg: {np.mean(avg[:10]):.4f}")
        print(f"  Last 10 avg:  {np.mean(avg[-10:]):.4f}")
        print(f"  Trend (last-first): {np.mean(avg[-10:]) - np.mean(avg[:10]):.4f}")
        
        osc = analyze_oscillation(metrics['avg_loss'])
        if osc:
            print(f"  Oscillation rate: {osc.get('oscillation_rate', 0):.4f}")
            print(f"  Mean running std: {osc.get('mean_running_std', 0):.4f}")
            print(f"  Early volatility: {osc.get('early_volatility', 0):.4f}")
            print(f"  Late volatility:  {osc.get('late_volatility', 0):.4f}")
    
    if metrics['policy_loss']:
        pl = np.array(metrics['policy_loss'])
        print(f"\n  Policy Loss: mean={np.mean(pl):.4f}, std={np.std(pl):.4f}")
        print(f"  First 10: {np.mean(pl[:10]):.4f}, Last 10: {np.mean(pl[-10:]):.4f}")
    
    if metrics['contrastive_loss']:
        cl = np.array(metrics['contrastive_loss'])
        print(f"  Contrastive Loss: mean={np.mean(cl):.4f}, std={np.std(cl):.4f}")
        print(f"  First 10: {np.mean(cl[:10]):.4f}, Last 10: {np.mean(cl[-10:]):.4f}")
    
    if metrics['value_loss']:
        vl = np.array(metrics['value_loss'])
        print(f"  Value Loss: mean={np.mean(vl):.6f}, std={np.std(vl):.6f}")
        print(f"  First 10: {np.mean(vl[:10]):.6f}, Last 10: {np.mean(vl[-10:]):.6f}")
        if vl[0] > 0:
            print(f"  Reduction: {((vl[0] - vl[-1]) / vl[0] * 100):.2f}%")
    
    if metrics['entropy']:
        ent = np.array(metrics['entropy'])
        print(f"\n  Entropy: mean={np.mean(ent):.4f}, std={np.std(ent):.4f}")
        print(f"  First 10: {np.mean(ent[:10]):.4f}, Last 10: {np.mean(ent[-10:]):.4f}")
    
    # ---- REWARD ANALYSIS ----
    print(f"\n--- REWARD ANALYSIS ---")
    if metrics['reward']:
        rew = np.array(metrics['reward'])
        print(f"  Mean: {np.mean(rew):.4f}, Std: {np.std(rew):.4f}")
        print(f"  Min: {np.min(rew):.4f}, Max: {np.max(rew):.4f}")
        print(f"  Median: {np.median(rew):.4f}")
        
        # Block analysis
        blocks = compute_block_stats(metrics['reward'], 100)
        print(f"\n  Block-wise reward stats (per 100 sentences):")
        for b in blocks:
            print(f"    {b['range']:>10s}: mean={b['mean']:.4f}, std={b['std']:.4f}, "
                  f"min={b['min']:.4f}, max={b['max']:.4f}")
        
        osc = analyze_oscillation(metrics['reward'])
        if osc:
            print(f"\n  Reward oscillation rate: {osc.get('oscillation_rate', 0):.4f}")
            print(f"  Early volatility: {osc.get('early_volatility', 0):.4f}")
            print(f"  Late volatility:  {osc.get('late_volatility', 0):.4f}")
        
        # Distribution
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        hist, _ = np.histogram(rew, bins=bins)
        total = len(rew)
        print(f"\n  Reward distribution:")
        for i in range(len(hist)):
            pct = hist[i] / total * 100
            print(f"    [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:>4d} ({pct:.1f}%)")
    
    # ---- CER ANALYSIS ----
    print(f"\n--- CER ANALYSIS ---")
    if metrics['cer']:
        cer = np.array(metrics['cer'])
        print(f"  Mean: {np.mean(cer):.4f}, Std: {np.std(cer):.4f}")
        print(f"  Min: {np.min(cer):.4f}, Max: {np.max(cer):.4f}")
        print(f"  Median: {np.median(cer):.4f}")
        
        blocks = compute_block_stats(metrics['cer'], 100)
        print(f"\n  Block-wise CER stats (per 100 sentences):")
        for b in blocks:
            print(f"    {b['range']:>10s}: mean={b['mean']:.4f}, std={b['std']:.4f}")
        
        # CER > 0.5 ratio
        high_cer = np.sum(cer > 0.5) / len(cer) * 100
        low_cer = np.sum(cer < 0.2) / len(cer) * 100
        print(f"\n  CER > 0.5: {high_cer:.1f}%")
        print(f"  CER < 0.2: {low_cer:.1f}%")
    
    # ---- PROMPT LENGTH ANALYSIS ----
    print(f"\n--- PROMPT SELECTION ANALYSIS ---")
    if metrics['prompt_lengths']:
        pl = np.array(metrics['prompt_lengths'])
        print(f"  Mean prompt length: {np.mean(pl):.2f} words")
        print(f"  Std: {np.std(pl):.2f}")
        
        blocks = compute_block_stats(metrics['prompt_lengths'], 100)
        print(f"\n  Prompt length evolution (per 100 sentences):")
        for b in blocks:
            print(f"    {b['range']:>10s}: mean={b['mean']:.2f}, std={b['std']:.2f}")
        
        # Short vs long prompt selection
        short = np.sum(pl <= 7)
        long_ = np.sum(pl >= 12)
        print(f"\n  Short prompts (<=7 words): {short} ({short/len(pl)*100:.1f}%)")
        print(f"  Long prompts (>=12 words): {long_} ({long_/len(pl)*100:.1f}%)")
        if long_ > 0:
            print(f"  Short/Long ratio: {short/long_:.2f}x")
    
    # ---- TARGET LENGTH vs CER ----
    print(f"\n--- TARGET LENGTH vs CER ---")
    if metrics['target_lengths'] and metrics['cer']:
        min_len = min(len(metrics['target_lengths']), len(metrics['cer']))
        tl = np.array(metrics['target_lengths'][:min_len])
        cer = np.array(metrics['cer'][:min_len])
        
        categories = {
            'VERY SHORT (1-3)': (tl >= 1) & (tl <= 3),
            'SHORT (4-6)': (tl >= 4) & (tl <= 6),
            'MEDIUM (7-10)': (tl >= 7) & (tl <= 10),
            'LONG (11+)': (tl >= 11),
        }
        
        for cat_name, mask in categories.items():
            if np.sum(mask) > 0:
                print(f"  {cat_name}: n={np.sum(mask)}, CER mean={np.mean(cer[mask]):.4f}, "
                      f"std={np.std(cer[mask]):.4f}")
    
    return metrics


def compare_experiments(all_metrics):
    """Compare all experiments."""
    print(f"\n{'='*80}")
    print(f"  COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    names = list(all_metrics.keys())
    
    # Table header
    print(f"\n{'Metric':<30s}", end="")
    for name in names:
        short = name.split('/')[-1][:20]
        print(f"  {short:>20s}", end="")
    print()
    print("-" * (30 + 22 * len(names)))
    
    # Rows
    rows = [
        ("Training iterations", lambda m: len(m['sentence_num'])),
        ("Total rewards", lambda m: len(m['reward'])),
        ("Mean reward", lambda m: f"{np.mean(m['reward']):.4f}" if m['reward'] else "N/A"),
        ("Std reward", lambda m: f"{np.std(m['reward']):.4f}" if m['reward'] else "N/A"),
        ("Mean CER", lambda m: f"{np.mean(m['cer']):.4f}" if m['cer'] else "N/A"),
        ("Std CER", lambda m: f"{np.std(m['cer']):.4f}" if m['cer'] else "N/A"),
        ("Mean avg_loss", lambda m: f"{np.mean(m['avg_loss']):.4f}" if m['avg_loss'] else "N/A"),
        ("Mean policy_loss", lambda m: f"{np.mean(m['policy_loss']):.4f}" if m['policy_loss'] else "N/A"),
        ("Mean value_loss", lambda m: f"{np.mean(m['value_loss']):.6f}" if m['value_loss'] else "N/A"),
        ("Mean entropy", lambda m: f"{np.mean(m['entropy']):.4f}" if m['entropy'] else "N/A"),
        ("Mean prompt length", lambda m: f"{np.mean(m['prompt_lengths']):.2f}" if m['prompt_lengths'] else "N/A"),
        ("CER > 0.5 %", lambda m: f"{np.sum(np.array(m['cer']) > 0.5)/len(m['cer'])*100:.1f}%" if m['cer'] else "N/A"),
        ("CER < 0.2 %", lambda m: f"{np.sum(np.array(m['cer']) < 0.2)/len(m['cer'])*100:.1f}%" if m['cer'] else "N/A"),
    ]
    
    for label, fn in rows:
        print(f"{label:<30s}", end="")
        for name in names:
            val = fn(all_metrics[name])
            print(f"  {str(val):>20s}", end="")
        print()
    
    # Reward improvement analysis
    print(f"\n--- REWARD TREND COMPARISON ---")
    for name in names:
        rew = all_metrics[name]['reward']
        if len(rew) >= 20:
            first = np.mean(rew[:min(50, len(rew)//5)])
            last = np.mean(rew[-min(50, len(rew)//5):])
            delta = last - first
            pct = delta / first * 100 if first != 0 else 0
            short = name.split('/')[-1][:25]
            print(f"  {short}: first50={first:.4f}, last50={last:.4f}, "
                  f"delta={delta:+.4f} ({pct:+.1f}%)")
    
    # Oscillation comparison
    print(f"\n--- OSCILLATION COMPARISON ---")
    for name in names:
        short = name.split('/')[-1][:25]
        if all_metrics[name]['reward']:
            osc = analyze_oscillation(all_metrics[name]['reward'])
            print(f"  {short} reward osc_rate={osc.get('oscillation_rate',0):.4f}, "
                  f"early_vol={osc.get('early_volatility',0):.4f}, "
                  f"late_vol={osc.get('late_volatility',0):.4f}")
        if all_metrics[name]['avg_loss']:
            osc = analyze_oscillation(all_metrics[name]['avg_loss'])
            print(f"  {short} loss   osc_rate={osc.get('oscillation_rate',0):.4f}, "
                  f"early_vol={osc.get('early_volatility',0):.4f}, "
                  f"late_vol={osc.get('late_volatility',0):.4f}")
    
    # Value loss convergence comparison
    print(f"\n--- VALUE LOSS CONVERGENCE ---")
    for name in names:
        vl = all_metrics[name]['value_loss']
        if vl:
            short = name.split('/')[-1][:25]
            first = np.mean(vl[:10])
            last = np.mean(vl[-10:])
            red = (first - last) / first * 100 if first > 0 else 0
            print(f"  {short}: first10={first:.6f}, last10={last:.6f}, reduction={red:.1f}%")
    
    # Prompt length evolution comparison
    print(f"\n--- PROMPT LENGTH EVOLUTION ---")
    for name in names:
        pl = all_metrics[name]['prompt_lengths']
        if pl:
            short = name.split('/')[-1][:25]
            first_block = np.mean(pl[:min(100, len(pl))])
            last_block = np.mean(pl[-min(100, len(pl)):])
            print(f"  {short}: first100={first_block:.2f}, last100={last_block:.2f}, "
                  f"delta={last_block-first_block:+.2f}")


def main():
    logs = {
        'pipeline_261012 (filtered)': Path("/info/raid-etu/m2/s2405959/VO2/Agent/pipeline_v2_filtered/logs_agent/pipeline_261012.log"),
        'pipeline_261011 (unfiltered)': Path("/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/pipeline_261011.log"),
        'pipeline_260326 (BASELINE)': Path("/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/pipeline_260326.log"),
    }
    
    all_metrics = {}
    for name, path in logs.items():
        print(f"\nParsing {path}...")
        metrics = parse_log_comprehensive(path)
        all_metrics[name] = print_analysis(name, metrics)
    
    compare_experiments(all_metrics)
    
    print(f"\n{'='*80}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
