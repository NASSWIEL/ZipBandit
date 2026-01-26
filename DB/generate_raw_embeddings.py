import os
import pandas as pd
import torch
import numpy as np
import faiss
import pickle
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from tqdm import tqdm

# Paths
TSV_PATH = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/reference_24khz/NEB_test_clean.tsv"
DB_DIR = "/info/raid-etu/m2/s2405959/VO2/Agent/DB"
VECTORS_DIR = os.path.join(DB_DIR, "vectors")
RAW_INDEX_PATH = os.path.join(VECTORS_DIR, "prompts_raw.index")
RAW_NPY_PATH = os.path.join(VECTORS_DIR, "prompts_raw.npy")

def load_data(tsv_path):
    print(f"Loading data from {tsv_path}...")
    # Read TSV without header, assign column names
    # Format: {wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['wav_name', 'prompt_transcription', 'prompt_wav', 'text'], quoting=3)
    return df

def generate_embeddings(df, device='cuda'):
    print("Loading SONAR Speech model...")
    # Use Speech encoder to capture audio characteristics
    # 'sonar_speech_encoder_fra' is the French speech encoder
    s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_fra", device=torch.device(device))
    
    embeddings = []
    print("Generating embeddings from Audio Prompts...")
    batch_size = 32
    audio_paths = df['prompt_wav'].tolist()
    
    for i in tqdm(range(0, len(audio_paths), batch_size)):
        batch = audio_paths[i:i+batch_size]
        # SONAR Speech pipeline can take list of file paths
        emb = s2vec_model.predict(batch) 
        embeddings.append(emb.cpu().numpy())
        
    return np.vstack(embeddings)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if not os.path.exists(VECTORS_DIR):
        os.makedirs(VECTORS_DIR)
    
    df = load_data(TSV_PATH)
    print(f"Loaded {len(df)} prompts.")
    
    # Generate 1024-dim embeddings
    embeddings_1024 = generate_embeddings(df, device=device)
    print(f"Generated embeddings shape: {embeddings_1024.shape}")
    
    # Save as .npy for raw access
    print(f"Saving raw embeddings to {RAW_NPY_PATH}...")
    np.save(RAW_NPY_PATH, embeddings_1024)

    # Create FAISS index for raw vectors
    print("Creating FAISS index (1024 dim)...")
    d = 1024
    index = faiss.IndexFlatL2(d)
    
    # Note: We are NOT normalizing here to preserve the raw SONAR output.
    # If cosine similarity is desired later, these vectors should be normalized before query/add.
    # But for "raw storage", we keep them as is.
    
    index.add(embeddings_1024)
    print(f"Index contains {index.ntotal} vectors.")
    
    # Save Index
    print(f"Saving FAISS index to {RAW_INDEX_PATH}...")
    faiss.write_index(index, RAW_INDEX_PATH)
        
    print("Done.")

if __name__ == "__main__":
    main()
