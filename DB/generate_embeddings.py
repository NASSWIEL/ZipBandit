import os
import pandas as pd
import torch
import numpy as np
import faiss
import pickle
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sklearn.decomposition import PCA
from tqdm import tqdm

# Paths
TSV_PATH = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/reference_24khz/NEB_test_clean.tsv"
DB_DIR = "/info/raid-etu/m2/s2405959/VO2/Agent/DB"
# EXPERT FIX: New directory for 256-dim embeddings
VECTORS_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256"
FAISS_INDEX_PATH = os.path.join(VECTORS_DIR, "prompts.index")
METADATA_PATH = os.path.join(VECTORS_DIR, "prompts_metadata.pkl")
PCA_PATH = os.path.join(VECTORS_DIR, "pca_model.pkl")

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
    # Force CPU due to old GPU incompatibility (CUDA capability 5.2 < required 7.0)
    device = 'cpu'
    print(f"Using device: {device} (forced due to GPU compatibility issues)")
    
    if not os.path.exists(VECTORS_DIR):
        os.makedirs(VECTORS_DIR)
    
    df = load_data(TSV_PATH)
    print(f"Loaded {len(df)} prompts.")
    
    # Generate 1024-dim embeddings
    embeddings_1024 = generate_embeddings(df, device=device)
    print(f"Generated embeddings shape: {embeddings_1024.shape}")
    
    # EXPERT FIX: Apply PCA to reduce to 256 dims (instead of 100)
    print("Fitting PCA (1024 -> 256)...")
    pca = PCA(n_components=256)
    embeddings_256 = pca.fit_transform(embeddings_1024)
    print(f"Reduced embeddings shape: {embeddings_256.shape}")
    
    # Save PCA model
    print(f"Saving PCA model to {PCA_PATH}...")
    with open(PCA_PATH, 'wb') as f:
        pickle.dump(pca, f)
        
    # Create FAISS index (256 dims)
    print("Creating FAISS index...")
    d = 256  # EXPERT FIX: Updated from 100 to 256
    index = faiss.IndexFlatL2(d)
    
    # Normalize for Cosine Similarity (L2 distance on normalized vectors <=> Cosine Similarity)
    # faiss.normalize_L2 expects float32
    embeddings_256 = embeddings_256.astype(np.float32)
    faiss.normalize_L2(embeddings_256)
    
    index.add(embeddings_256)
    print(f"Index contains {index.ntotal} vectors.")
    
    # Save Index
    print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Save Metadata
    print(f"Saving metadata to {METADATA_PATH}...")
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            'wav_name': row['wav_name'],
            'prompt_transcription': row['prompt_transcription'],
            'prompt_wav': row['prompt_wav']
        })
        
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
        
    print("Done.")

if __name__ == "__main__":
    main()
