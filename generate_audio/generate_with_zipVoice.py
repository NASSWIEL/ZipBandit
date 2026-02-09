import argparse
import json
import subprocess
import os
import sys
import re

def main():
    parser = argparse.ArgumentParser(description="Generate audio using ZipVoice.")
    parser.add_argument("--similarity_output", type=str, required=True, help="Path to the JSON output from assess_similarity.py")
    parser.add_argument("--target_text", type=str, required=True, help="The target text to synthesize.")
    parser.add_argument("--output_path_file", type=str, help="Path to save the generated audio filename.")
    args = parser.parse_args()

    # 1. Load Similarity Results
    if not os.path.exists(args.similarity_output):
        print(f"Error: Similarity output file not found: {args.similarity_output}")
        sys.exit(1)

    with open(args.similarity_output, 'r') as f:
        sim_data = json.load(f)

    prompt_wav = sim_data.get("prompt_wav")
    prompt_text = sim_data.get("prompt_transcription")

    if not prompt_wav or not prompt_text:
        print("Error: Missing prompt_wav or prompt_transcription in similarity output.")
        sys.exit(1)

    print(f"Selected Prompt WAV: {prompt_wav}")
    print(f"Selected Prompt Text: {prompt_text}")

    # 2. Setup Paths
    # Fixed output directory as per instructions
    output_dir = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/RL_train"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate informative filename
    # 1. Extract first two words from target text
    words = args.target_text.split()[:2]
    # Sanitize: keep only ascii letters and numbers (English characters)
    sanitized_words = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in words]
    # Filter out empty strings
    sanitized_words = [w for w in sanitized_words if w]
    
    if not sanitized_words:
        text_prefix = "generated"
    else:
        text_prefix = "_".join(sanitized_words)

    # 2. Extract prompt ID from wav path
    prompt_filename = os.path.basename(prompt_wav)
    prompt_id = os.path.splitext(prompt_filename)[0]

    filename = f"{text_prefix}_{prompt_id}.wav"
    output_wav_path = os.path.join(output_dir, filename)

    # ZipVoice environment path (allow override via env var)
    zipvoice_env_path = os.environ.get(
        "ZIPVOICE_ENV_PATH",
        "/info/etu/m2/s2405959/miniconda3/envs/zipvoice_clean",
    )
    # ZipVoice Python Executable
    zipvoice_python = os.path.join(zipvoice_env_path, "bin", "python")

    # ZipVoice Root Directory (to be added to PYTHONPATH)
    zipvoice_root = os.environ.get(
        "ZIPVOICE_ROOT",
        "/info/raid-etu/m2/s2405959/VO2/ZipVoice",
    )

    # Setup LD_LIBRARY_PATH for CUDA/cuDNN
    nvidia_path = os.path.join(zipvoice_env_path, "lib/python3.11/site-packages/nvidia")
    
    nvidia_libs = []
    if os.path.exists(nvidia_path):
        for root, dirs, files in os.walk(nvidia_path):
            if 'lib' in dirs:
                nvidia_libs.append(os.path.join(root, 'lib'))
            # Also check if current dir is a lib dir (sometimes structure varies)
            if os.path.basename(root) == 'lib':
                nvidia_libs.append(root)
    
    nvidia_lib_path = ":".join(nvidia_libs)

    # 3. Construct Command
    # python3 -m zipvoice.bin.infer_zipvoice ...
    cmd = [
        zipvoice_python,
        "-m", "zipvoice.bin.infer_zipvoice",
        "--model-name", "zipvoice",
        "--prompt-wav", prompt_wav,
        "--prompt-text", prompt_text,
        "--text", args.target_text,
        "--res-wav-path", output_wav_path
    ]

    # 4. Execute Command
    print("Running ZipVoice Inference...")
    print("Command:", " ".join(cmd))

    env = os.environ.copy()
    # Add ZipVoice root to PYTHONPATH
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{zipvoice_root}:{current_pythonpath}"
    
    # Add Nvidia libs to LD_LIBRARY_PATH
    current_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{current_ld_path}:{nvidia_lib_path}"

    # Force CPU execution for ZipVoice (disable CUDA visibility)
    env["CUDA_VISIBLE_DEVICES"] = ""

    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"Inference successful. Audio saved to: {output_wav_path}")
        
        # Save the output path to a file if requested
        if args.output_path_file:
            with open(args.output_path_file, 'w') as f:
                f.write(output_wav_path)
                
    except subprocess.CalledProcessError as e:
        print(f"Error running ZipVoice inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
