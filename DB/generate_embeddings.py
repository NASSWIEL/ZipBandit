import os
import pandas as pd
import torch
import numpy as np
import faiss
import pickle
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sklearn.decomposition import PCA
from tqdm import tqdm

TSV_PATH = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/reference_24khz/NEB_test_clean.tsv"
DB_DIR = "/info/raid-etu/m2/s2405959/VO2/Agent/DB"
VECTORS_DIR = os.path.join(DB_DIR, "vectors")
FAISS_INDEX_PATH = os.path.join(VECTORS_DIR, "prompts.index")
METADATA_PATH = os.path.join(VECTORS_DIR, "prompts_metadata.pkl")
PCA_PATH = os.path.join(VECTORS_DIR, "pca_model.pkl")

def load_data(tsv_path):
    print(f"Loading data from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['wav_name', 'prompt_transcription', 'prompt_wav', 'text'], quoting=3)
    return df

def generate_embeddings(df, device='cuda'):
    print("Loading SONAR Speech model...")
    s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_fra", device=torch.device(device))

    embeddings = []
    print("Generating embeddings from Audio Prompts...")
    batch_size = 32
    audio_paths = df['prompt_wav'].tolist()

    for i in tqdm(range(0, len(audio_paths), batch_size)):
        batch = audio_paths[i:i+batch_size]
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

    embeddings_1024 = generate_embeddings(df, device=device)
    print(f"Generated embeddings shape: {embeddings_1024.shape}")

    print("Fitting PCA (1024 -> 100)...")
    pca = PCA(n_components=100)
    embeddings_100 = pca.fit_transform(embeddings_1024)
    print(f"Reduced embeddings shape: {embeddings_100.shape}")

    print(f"Saving PCA model to {PCA_PATH}...")
    with open(PCA_PATH, 'wb') as f:
        pickle.dump(pca, f)

    print("Creating FAISS index...")
    d = 100
    index = faiss.IndexFlatL2(d)

    embeddings_100 = embeddings_100.astype(np.float32)
    faiss.normalize_L2(embeddings_100)

    index.add(embeddings_100)
    print(f"Index contains {index.ntotal} vectors.")

    print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, FAISS_INDEX_PATH)

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
