import re
import statistics
from collections import Counter

def parse_logs(file_path, limit=500):
    entries = []
    current_entry = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Start of a new sentence processing
            if "Processing Sentence" in line:
                if current_entry.get('prompt') and current_entry.get('cer') is not None:
                    # Save previous entry if complete (some might be partial/failed)
                     entries.append(current_entry)
                     if len(entries) >= limit:
                         break
                current_entry = {}
            
            # Extract Input
            if line.startswith('Input:'):
                current_entry['input'] = line.replace('Input:', '').strip().strip('"')
                
            # Extract Selected Prompt (Transcription in log context)
            if line.startswith('Transcription:'):
                current_entry['prompt'] = line.replace('Transcription:', '').strip()
                
            # Extract CER
            if line.startswith('CER:'):
                try:
                    current_entry['cer'] = float(line.replace('CER:', '').strip())
                except ValueError:
                    pass

            # Extract Reward
            if line.startswith('Reward:'):
                try:
                    current_entry['reward'] = float(line.replace('Reward:', '').strip())
                except ValueError:
                    pass
                    
    # Add the last entry if complete
    if len(entries) < limit and current_entry.get('prompt') and current_entry.get('cer') is not None:
        entries.append(current_entry)
        
    return entries

def analyze_entries(entries):
    short_prompts = []
    long_prompts = []
    
    # Define thresholds strictly
    SHORT_THRESHOLD = 7  # words
    LONG_THRESHOLD = 12   # words
    
    for entry in entries:
        prompt_words = entry['prompt'].split()
        word_count = len(prompt_words)
        entry['word_count'] = word_count
        
        if word_count <= SHORT_THRESHOLD:
            short_prompts.append(entry)
        elif word_count >= LONG_THRESHOLD:
            long_prompts.append(entry)
            
    # Calculate stats
    avg_cer_short = statistics.mean([e['cer'] for e in short_prompts]) if short_prompts else 0
    avg_cer_long = statistics.mean([e['cer'] for e in long_prompts]) if long_prompts else 0
    
    return {
        'total_analyzed': len(entries),
        'short_count': len(short_prompts),
        'long_count': len(long_prompts),
        'avg_cer_short': avg_cer_short,
        'avg_cer_long': avg_cer_long,
        'short_examples': short_prompts[:3],
        'long_examples': long_prompts[:3],
        'prompts': [e['prompt'] for e in entries] # For repetition analysis
    }

def print_report(data):
    print("ANALYSIS REPORT")
    print("===============")
    print(f"Total Sentences Analyzed: {data['total_analyzed']}")
    print(f"Short Prompts (<= 7 words): {data['short_count']}")
    print(f"Long Prompts (>= 12 words): {data['long_count']}")
    print(f"Average CER (Short): {data['avg_cer_short']:.4f}")
    print(f"Average CER (Long): {data['avg_cer_long']:.4f}")
    
    print("\nEXAMPLES: SHORT PROMPTS")
    for ex in data['short_examples']:
        print(f"Target: \"{ex['input']}\"")
        print(f"- Prompt: \"{ex['prompt']}\" (Length: {ex['word_count']}) -> CER: {ex['cer']:.4f}\n")
        
    print("\nEXAMPLES: LONG PROMPTS")
    for ex in data['long_examples']:
        print(f"Target: \"{ex['input']}\"")
        print(f"- Prompt: \"{ex['prompt']}\" (Length: {ex['word_count']}) -> CER: {ex['cer']:.4f}\n")

    # Repetition Check
    prompt_counts = Counter(data['prompts'])
    most_common = prompt_counts.most_common(5)
    print("\nMOST COMMON PROMPTS")
    for prompt, count in most_common:
        print(f"- \"{prompt}\": {count} times")

if __name__ == "__main__":
    LOG_FILE = "/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/pipeline_260326.log"
    entries = parse_logs(LOG_FILE, limit=500)
    analysis = analyze_entries(entries)
    print_report(analysis)
