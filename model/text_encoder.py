import torch
import numpy as np
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

# Global cache for reusing the encoder across calls
_ENCODER_CACHE = None

class TextEncoder:
    def __init__(self, device=None, use_cache=True):
        """
        Initialize the SONAR Text Encoder with optional caching.
        
        Args:
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). 
                                    If None, automatically detects cuda.
            use_cache (bool): If True, reuse cached encoder to avoid reloading.
        """
        global _ENCODER_CACHE
        
        if device is None:
            # Force CPU due to CUDA capability mismatch on current cluster node (Titan X sm_52 vs PyTorch sm_70+)
            self.device = torch.device('cpu')
            # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Try to reuse cached encoder
        if use_cache and _ENCODER_CACHE is not None:
            # print("[CACHE] Reusing cached SONAR Text model")
            self.model = _ENCODER_CACHE
        else:
            print(f"Loading SONAR Text model on {self.device}...")
            # We use the multilingual SONAR text encoder
            # 'text_sonar_basic_encoder' is the standard multilingual text encoder
            self.model = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder", 
                tokenizer="text_sonar_basic_encoder",
                device=self.device
            )
            
            if use_cache:
                _ENCODER_CACHE = self.model
                # print("[CACHE] Cached SONAR Text model for future use")

    def encode(self, texts):
        """
        Encodes a list of text strings into 1024-dimensional embeddings.
        
        Args:
            texts (str or list[str]): The text(s) to encode.
            
        Returns:
            np.ndarray: 
                - If input is a single string: Array of shape (1024,)
                - If input is a list: Array of shape (n_texts, 1024)
        """
        is_single_input = isinstance(texts, str)
        if is_single_input:
            texts = [texts]
            
        # Predict embeddings
        # source_lang="fra_Latn" specifies French (Latin script)
        embeddings = self.model.predict(texts, source_lang="fra_Latn")
        
        result = embeddings.cpu().numpy()
        
        if is_single_input:
            return result[0]
            
        return result

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Encode text using SONAR Text Encoder.")
    parser.add_argument("--sentence", type=str, required=True, help="The text to encode.")
    parser.add_argument("--output", type=str, help="Path to save the output vector (.npy).")
    args = parser.parse_args()

    try:
        # print("Initializing TextEncoder...")
        encoder = TextEncoder()
        
        test_text = args.sentence
        # print(f"Encoding text: '{test_text}'")
        
        emb = encoder.encode(test_text)
        
        if args.output:
            np.save(args.output, emb)
            print(f"Vector saved to {args.output}")
        else:
            # Output the vector to stdout if no file specified
            print(emb.tolist())
            
    except Exception as e:
        print(f"Error: {e}")
