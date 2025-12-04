import torch
import torch.nn as nn
import os

DB_DIR = "/info/raid-etu/m2/s2405959/VO2/Agent/DB"
VECTORS_DIR = os.path.join(DB_DIR, "vectors")
FAISS_INDEX_PATH = os.path.join(VECTORS_DIR, "prompts.index")
METADATA_PATH = os.path.join(VECTORS_DIR, "prompts_metadata.pkl")
PCA_PATH = os.path.join(VECTORS_DIR, "pca_model.pkl")

class SonarAgent(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100, dropout=0.2):
        super(SonarAgent, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(

            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, output_dim)

        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Run Agent Model (1024 -> 100).")
    parser.add_argument("--input", type=str, help="Path to input 1024-dim vector (.npy).")
    parser.add_argument("--output", type=str, help="Path to save output 100-dim vector (.npy).")
    parser.add_argument("--model_path", type=str, help="Path to model weights (optional).")

    args = parser.parse_args()

    if args.input and args.output:
        try:
            input_vec = np.load(args.input)
            input_tensor = torch.from_numpy(input_vec).float()

            if len(input_tensor.shape) == 1:
                input_tensor = input_tensor.unsqueeze(0)

            model = SonarAgent()

            if args.model_path and os.path.exists(args.model_path):
                model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
                model.eval()
            else:
                model.eval()

            with torch.no_grad():
                output_tensor = model(input_tensor)

            output_vec = output_tensor.numpy()
            np.save(args.output, output_vec)
            print(f"Agent output saved to {args.output}")

        except Exception as e:
            print(f"Error during inference: {e}")
            exit(1)
    else:
        try:
            print("Initializing SonarAgent...")
            model = SonarAgent()
            print(model)

            batch_size = 4
            dummy_input = torch.randn(batch_size, 1024)
            print(f"Dummy input shape: {dummy_input.shape}")

            output = model(dummy_input)
            print(f"Output shape: {output.shape}")

            if output.shape == (batch_size, 100):
                print("Success: Output dimension is correct (100).")
            else:
                print(f"Error: Expected output shape ({batch_size}, 100), got {output.shape}")

        except Exception as e:
            print(f"Test failed: {e}")