import sys
import os
import argparse
import torch
import librosa
import numpy as np

# Add parent directory to import model_cache
AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

# Add WER_CER directory to path to import ASREngine
WER_CER_DIR = "/info/raid-etu/m2/s2405959/VO2/WER_CER"
if WER_CER_DIR not in sys.path:
    sys.path.append(WER_CER_DIR)

try:
    from asr_engine import ASREngine
except ImportError as e:
    print(f"Error importing ASREngine: {e}")
    sys.exit(1)

# Try to use model cache
try:
    from model.model_cache import ModelCache
    USE_CACHE = True
except:
    USE_CACHE = False

def load_audio_librosa(audio_path, target_sr=16000):
    """Load audio using librosa and resample to 16kHz"""
    try:
        # librosa.load returns (audio, sr) where audio is a numpy array
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Calculate CER for generated audio.")
    parser.add_argument("target_text", type=str, help="The target text (ground truth).")
    parser.add_argument("audio_path", type=str, help="Path to the generated audio file.")
    parser.add_argument("--output_cer", type=str, help="Path to save the CER value.")
    
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        sys.exit(1)

    # Configuration for ASREngine
    # Using Whisper V3 Large as requested
    config = {
        'asr_model': 'openai/whisper-large-v3',
        'language': 'fr',  # Assuming French based on previous context (ZipVoice/Blizzard)
        'cer_remove_spaces': False
    }

    # Determine device
    # Force CPU due to CUDA capability mismatch on current cluster node
    device = 'cpu' 
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")

    # Initialize ASR Engine
    # This will load the model (or get from cache)
    try:
        if USE_CACHE:
            # Try to get cached Whisper model
            cache = ModelCache()
            cache_status = cache.get_cache_status()
            if cache_status.get('whisper_model'):
                print("Using cached Whisper model")
            else:
                print("Loading Whisper model (will be cached for future use)...")
                cache.get_whisper_model(model_size='large-v3', device=device)
        
        asr_engine = ASREngine(config, device)
    except Exception as e:
        print(f"Error initializing ASREngine: {e}")
        sys.exit(1)

    # Load Audio
    audio = load_audio_librosa(args.audio_path)
    if audio is None:
        sys.exit(1)

    # Transcribe
    # transcribe_batch expects a list of numpy arrays
    try:
        # print("Transcribing...")
        transcriptions = asr_engine.transcribe_batch([audio])
        
        if not transcriptions:
            print("Error: No transcription generated.")
            sys.exit(1)
            
        prediction = transcriptions[0]
        # print(f"Transcription: {prediction}")

        # Calculate CER
        metrics = asr_engine.compute_metrics(prediction, args.target_text)
        
        cer = metrics['cer']
        
        # Final Output
        print(f"CER: {cer:.4f}")
        
        if args.output_cer:
            with open(args.output_cer, 'w') as f:
                f.write(f"{cer:.4f}")

        
        # Optional: Print details for debugging/logging (to stderr to keep stdout clean if needed)
        # print(f"Target: {metrics['reference_clean']}", file=sys.stderr)
        # print(f"Pred  : {metrics['prediction_clean']}", file=sys.stderr)

    except Exception as e:
        print(f"Error during transcription/calculation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
