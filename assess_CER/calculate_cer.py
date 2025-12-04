import sys
import os
import argparse
import torch
import librosa
import numpy as np

WER_CER_DIR = "/info/raid-etu/m2/s2405959/VO2/WER_CER"
if WER_CER_DIR not in sys.path:
    sys.path.append(WER_CER_DIR)

try:
    from asr_engine import ASREngine
except ImportError as e:
    print(f"Error importing ASREngine: {e}")
    sys.exit(1)

def load_audio_librosa(audio_path, target_sr=16000):
    try:
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

    config = {
        'asr_model': 'openai/whisper-large-v3',
        'language': 'fr',
        'cer_remove_spaces': False
    }

    device = 'cpu'

    try:
        asr_engine = ASREngine(config, device)
    except Exception as e:
        print(f"Error initializing ASREngine: {e}")
        sys.exit(1)

    audio = load_audio_librosa(args.audio_path)
    if audio is None:
        sys.exit(1)

    try:
        transcriptions = asr_engine.transcribe_batch([audio])

        if not transcriptions:
            print("Error: No transcription generated.")
            sys.exit(1)

        prediction = transcriptions[0]
        metrics = asr_engine.compute_metrics(prediction, args.target_text)
        cer = metrics['cer']
        print(f"CER: {cer:.4f}")

        if args.output_cer:
            with open(args.output_cer, 'w') as f:
                f.write(f"{cer:.4f}")

    except Exception as e:
        print(f"Error during transcription/calculation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
