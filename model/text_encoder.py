import torch
import numpy as np
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

class TextEncoder:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        print(f"Loading SONAR Text model on {self.device}...")
        self.model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device
        )

    def encode(self, texts):
        is_single_input = isinstance(texts, str)
        if is_single_input:
            texts = [texts]

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
        encoder = TextEncoder()
        test_text = args.sentence

        emb = encoder.encode(test_text)

        if args.output:
            np.save(args.output, emb)
            print(f"Vector saved to {args.output}")
        else:
            print(emb.tolist())

    except Exception as e:
        print(f"Error: {e}")
