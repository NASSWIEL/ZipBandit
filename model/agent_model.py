import torch
import torch.nn as nn
import os
import numpy as np

# Constants for paths (using 256-dim vectors for FULL 63K dataset)
VECTORS_DIR = "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256"
FAISS_INDEX_PATH = os.path.join(VECTORS_DIR, "prompts.index")
METADATA_PATH = os.path.join(VECTORS_DIR, "prompts_metadata.pkl")
PCA_PATH = os.path.join(VECTORS_DIR, "pca_model.pkl")
CENTROIDS_PATH = os.path.join(os.path.dirname(__file__), "prompt_centroids.pt")


class ResidualBlock(nn.Module):
    """Residual block with pre-normalization for better gradient flow."""
    
    def __init__(self, dim, dropout=0.1, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.net(x)


class SonarAgent(nn.Module):
    def __init__(self, input_dim=1024, output_dim=256, dropout=0.15, hidden_dim=512, n_residual_blocks=4):
        """
        Neural network agent with residual connections and value head.
        Maps SONAR text embeddings (1024d) to prompt embedding space (256d).
        
        Args:
            input_dim (int): Dimension of input SONAR text embeddings.
            output_dim (int): Dimension of target prompt embeddings.
            dropout (float): Dropout probability.
            hidden_dim (int): Hidden layer dimension.
            n_residual_blocks (int): Number of residual blocks.
        """
        super(SonarAgent, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection with skip connection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks for deep feature extraction
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) 
            for _ in range(n_residual_blocks)
        ])
        
        # Action head (policy): outputs prompt embedding
        # Use tanh to constrain output range before normalization
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Less dropout in action head
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Value head: 512 -> 256 -> 128 -> 64 -> 1 with skip connection
        self.value_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Deeper value network with intermediate layers
        self.value_hidden1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.value_hidden2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.value_output = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Constrain to [0, 1] to match reward range
        )
        # Skip connection for value head (projects 256 -> 64 for residual)
        self.value_skip = nn.Linear(256, 64)
        
        # Statistics for reward normalization
        self.register_buffer('reward_mean', torch.tensor([0.65]))
        self.register_buffer('reward_std', torch.tensor([0.2]))
        self.register_buffer('reward_count', torch.tensor([10.0]))  # Avoid cold start
        
        # Prompt manifold centroids for guided exploration (loaded separately)
        self.register_buffer('prompt_centroids', None)
        
        # Diversity tracking with decay
        self.prompt_usage_freq = {}  # prompt_idx -> count
        self.usage_decay = 0.99  # Decay factor to prevent infinite penalty growth
        
        # Initialize weights for better cosine similarity optimization
        self._init_weights()
        
        # Try to load pre-computed centroids
        self._load_centroids()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _load_centroids(self):
        """Load pre-computed prompt manifold centroids for guided exploration."""
        if os.path.exists(CENTROIDS_PATH):
            try:
                centroids = torch.load(CENTROIDS_PATH, map_location='cpu')
                self.register_buffer('prompt_centroids', centroids)
                print(f"Loaded {centroids.shape[0]} prompt centroids from {CENTROIDS_PATH}")
            except Exception as e:
                print(f"Warning: Could not load centroids: {e}")
                self.prompt_centroids = None
        else:
            print(f"No centroids file at {CENTROIDS_PATH} - using random exploration")
            self.prompt_centroids = None
    
    def _get_features(self, x):
        """Extract features through projection and residual blocks."""
        # Project input to hidden dimension
        features = self.input_proj(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            features = block(features)
        
        return features

    def forward(self, x, add_noise=False, noise_std=0.1, epsilon=0.0, return_value=False):
        """
        Forward pass with manifold-aware exploration and value estimation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            add_noise (bool): Whether to add Gaussian noise for exploration.
            noise_std (float): Standard deviation of exploration noise.
            epsilon (float): Probability of random exploration [0, 1].
            return_value (bool): If True, also return value estimate.
            
        Returns:
            torch.Tensor or tuple: 
                - If return_value=False: L2-normalized action tensor (batch_size, output_dim)
                - If return_value=True: (action, value) tuple
        """
        # Extract features through residual network
        features = self._get_features(x)
        
        # Get action (policy output)
        action = self.action_head(features)
        
        # L2 normalization to align with FAISS index
        action = torch.nn.functional.normalize(action, p=2, dim=1)
        
        # Value estimate through deeper pathway
        value_features = self.value_proj(features)
        v1 = self.value_hidden1(value_features)
        v2 = self.value_hidden2(v1)
        # Skip connection from value_proj output to value_hidden2 output
        v2_skip = v2 + self.value_skip(value_features)
        value = self.value_output(v2_skip)
        
        # Epsilon-greedy exploration during training
        if self.training and epsilon > 0:
            random_mask = (torch.rand(x.size(0)) < epsilon).to(x.device)
            if random_mask.any():
                n_random = random_mask.sum().item()
                
                    # Sample from centroids for manifold-aware exploration
                if self.prompt_centroids is not None and len(self.prompt_centroids) > 0:
                    # Sample random centroids for exploration
                    centroid_indices = torch.randint(
                        0, len(self.prompt_centroids), (n_random,), device=x.device
                    )
                    exploration_vectors = self.prompt_centroids[centroid_indices].to(x.device)
                    
                    # Add small noise around centroids (local exploration)
                    centroid_noise = torch.randn_like(exploration_vectors) * 0.1
                    exploration_vectors = exploration_vectors + centroid_noise
                    exploration_vectors = torch.nn.functional.normalize(exploration_vectors, p=2, dim=1)
                else:
                    # Fallback: random hypersphere sampling
                    exploration_vectors = torch.randn(n_random, self.output_dim, device=x.device)
                    exploration_vectors = torch.nn.functional.normalize(exploration_vectors, p=2, dim=1)
                
                action[random_mask] = exploration_vectors
        
        # Add exploration noise during training (small perturbation)
        if add_noise and self.training:
            noise = torch.randn_like(action) * noise_std
            action = action + noise
            action = torch.nn.functional.normalize(action, p=2, dim=1)
        
        if return_value:
            return action, value
        return action
    
    def update_reward_stats(self, reward):
        """Update running reward statistics using Welford's online algorithm."""
        if isinstance(reward, (int, float)):
            reward = torch.tensor(reward, device=self.reward_mean.device)
        
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        # Welford's online algorithm for variance
        m2 = self.reward_std ** 2 * (self.reward_count - 1) if self.reward_count > 1 else torch.zeros(1, device=reward.device)
        m2 = m2.to(delta.device)
        m2 += delta * delta2
        # Use sample variance (N-1) for unbiased estimator
        self.reward_std = torch.sqrt(m2 / max(1, self.reward_count - 1))
        # Clamp std to prevent too aggressive normalization
        self.reward_std = torch.clamp(self.reward_std, min=0.1, max=0.5)
    
    def normalize_reward(self, reward):
        """Normalize reward using running statistics."""
        if self.reward_count < 2:
            return reward
        # Use softer normalization to prevent extreme values
        normalized = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return torch.clamp(normalized, min=-3.0, max=3.0)
    
    def update_prompt_usage(self, prompt_idx):
        """Track prompt usage for diversity penalty with decay."""
        # Apply decay to all existing counts
        if hasattr(self, 'usage_decay'):
            for k in self.prompt_usage_freq:
                self.prompt_usage_freq[k] *= self.usage_decay
        
        if prompt_idx not in self.prompt_usage_freq:
            self.prompt_usage_freq[prompt_idx] = 0
        self.prompt_usage_freq[prompt_idx] += 1
    
    def get_diversity_penalty(self, prompt_idx, alpha=0.05):
        """Calculate diversity penalty based on prompt usage frequency.
        
        Args:
            prompt_idx (int): Index of the selected prompt.
            alpha (float): Penalty strength.
            
        Returns:
            float: Diversity penalty (higher for more frequently used prompts).
        """
        if not self.prompt_usage_freq:
            return 0.0
        
        usage_count = self.prompt_usage_freq.get(prompt_idx, 0)
        total_count = sum(self.prompt_usage_freq.values())
        
        if total_count == 0:
            return 0.0
        
        # Softer penalty using log to reduce extreme penalties
        frequency = usage_count / total_count
        # Use log(1 + frequency) for softer scaling
        penalty = alpha * np.log1p(frequency * 10)
        
        return min(penalty, 0.1)  # Cap maximum penalty
    
    def reset_diversity_tracking(self):
        """Reset diversity tracking for fresh training."""
        self.prompt_usage_freq = {}
    
    def compute_manifold_alignment_loss(self, predictions):
        """
        Compute manifold alignment loss to keep predictions close to prompt centroid space.
        
        This loss encourages the agent's outputs to stay within the learned prompt manifold,
        preventing outputs from falling into "dead zones" of the embedding space.
        
        Args:
            predictions: Agent predictions (batch_size, output_dim), should be L2 normalized
            
        Returns:
            torch.Tensor: Scalar loss value (mean distance to nearest centroid)
        """
        if self.prompt_centroids is None or len(self.prompt_centroids) == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # Ensure centroids are on the same device
        centroids = self.prompt_centroids.to(predictions.device)
        
        # Compute cosine similarity to all centroids (predictions and centroids are L2 normalized)
        # Shape: (batch_size, num_centroids)
        similarities = torch.mm(predictions, centroids.T)
        
        # Get maximum similarity for each prediction (closest centroid)
        max_similarities, _ = similarities.max(dim=1)
        
        # Loss is (1 - max_similarity), so we minimize distance to nearest centroid
        # When prediction is close to a centroid, similarity is high, loss is low
        alignment_loss = (1.0 - max_similarities).mean()
        
        return alignment_loss
    
    def compute_centroid_proximity_regularization(self, predictions, soft_constraint=True):
        """
        Output normalization with centroid proximity constraint.
        
        Ensures outputs are not just L2 normalized but also lie near the prompt manifold.
        
        Args:
            predictions: Agent predictions (batch_size, output_dim)
            soft_constraint: If True, return soft penalty. If False, project to nearest centroid.
            
        Returns:
            torch.Tensor: Regularization loss or projected predictions
        """
        if self.prompt_centroids is None or len(self.prompt_centroids) == 0:
            return torch.tensor(0.0, device=predictions.device) if soft_constraint else predictions
        
        centroids = self.prompt_centroids.to(predictions.device)
        
        # Compute distances to all centroids
        # Using L2 distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        # Since both are L2 normalized: = 2 - 2*cos_sim = 2*(1 - cos_sim)
        similarities = torch.mm(predictions, centroids.T)
        
        if soft_constraint:
            # Soft constraint: penalize predictions far from any centroid
            max_sim, nearest_idx = similarities.max(dim=1)
            
            # Compute variance of similarities (encourage being close to ONE centroid, not spread)
            sim_var = similarities.var(dim=1).mean()
            
            # Loss: encourage high max similarity and low variance (be close to one centroid)
            proximity_loss = (1.0 - max_sim).mean() + 0.1 * sim_var
            return proximity_loss
        else:
            # Hard constraint: project to nearest centroid (for inference)
            _, nearest_idx = similarities.max(dim=1)
            projected = centroids[nearest_idx]
            return projected

if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Run Agent Model (1024 -> 256).")
    parser.add_argument("--input", type=str, help="Path to input 1024-dim vector (.npy).")
    parser.add_argument("--output", type=str, help="Path to save output 256-dim vector (.npy).")
    parser.add_argument("--model_path", type=str, help="Path to model weights (optional).")
    parser.add_argument("--exploration_noise", type=float, default=0.0, help="Std dev of exploration noise.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon for epsilon-greedy exploration.")
    
    args = parser.parse_args()

    if args.input and args.output:
        try:
            # Load input
            input_vec = np.load(args.input)
            input_tensor = torch.from_numpy(input_vec).float()
            
            # Handle shape (ensure batch dimension)
            if len(input_tensor.shape) == 1:
                input_tensor = input_tensor.unsqueeze(0)
            
            # Initialize model
            model = SonarAgent()
            
            # Load weights if provided
            if args.model_path and os.path.exists(args.model_path):
                try:
                    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=False)
                except Exception as e:
                    print(f"Warning: Could not load model weights ({e}), using random initialization")
            
            # Set to training mode to enable epsilon-greedy exploration
            if args.epsilon > 0 or args.exploration_noise > 0:
                model.train()
            else:
                model.eval()
            
            # Inference with exploration (no torch.no_grad to allow epsilon-greedy)
            add_noise = args.exploration_noise > 0
            output_tensor = model(input_tensor, add_noise=add_noise, noise_std=args.exploration_noise, epsilon=args.epsilon)
            
            # Save output (detach if requires_grad)
            output_vec = output_tensor.detach().numpy()
            np.save(args.output, output_vec)
            print(f"Agent output saved to {args.output}")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            exit(1)
    else:
        # Simple test to verify dimensions if no args provided
        try:
            print("Initializing SonarAgent...")
            model = SonarAgent()
            print(model)
            
            batch_size = 4
            dummy_input = torch.randn(batch_size, 1024)
            print(f"Dummy input shape: {dummy_input.shape}")
            
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            
            if output.shape == (batch_size, 256):
                print("Success: Output dimension is correct (256).")
            else:
                print(f"Error: Expected output shape ({batch_size}, 256), got {output.shape}")
                
        except Exception as e:
            print(f"Test failed: {e}")