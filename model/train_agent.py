import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import sys
import shutil
from datetime import datetime

# Add the Agent directory to sys.path to allow imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(CURRENT_DIR) # Parent of model dir
if AGENT_DIR not in sys.path:
    sys.path.append(AGENT_DIR)

from model.agent_model import SonarAgent
from model.replay_buffer import ReplayBuffer

def train_step(model, optimizer, state, action, reward, prompt_idx=None, 
               entropy_coef=0.01, epsilon=0.0, diversity_penalty_weight=0.1):
    """
    Perform a single training step using CONTRASTIVE LEARNING (EXPERT FIX #1).
    
    NEW APPROACH - Contrastive Learning:
    - GOOD prompts (reward > 0.8): Pull prediction TOWARD action
    - BAD prompts (reward < 0.3): Push prediction AWAY from action
    - MEDIOCRE prompts (0.3 ≤ reward ≤ 0.8): No strong signal, smaller weight
    
    This fixes the fundamental issue: we now learn what to AVOID, not just what to copy.
    
    Args:
        model: The SonarAgent model.
        optimizer: The optimizer.
        state: Input tensor (batch_size, 1024).
        action: Retrieved prompt vector (batch_size, 100).
        reward: Scalar or tensor reward [0, 1].
        prompt_idx: Index of the selected prompt (for diversity tracking).
        entropy_coef: Coefficient for entropy regularization (default: 0.01).
        epsilon: Epsilon-greedy exploration probability (default: 0.0).
        diversity_penalty_weight: Weight for diversity penalty (default: 0.1).
        
    Returns:
        dict: Dictionary with loss, entropy, diversity penalty, and reward metrics.
    """
    model.train() 
    optimizer.zero_grad()
    
    # Forward pass with epsilon-greedy and increased exploration noise
    # Also get value estimate for advantage calculation
    prediction, value = model(state, add_noise=True, noise_std=0.15, epsilon=epsilon, return_value=True)
    
    # ============================================================
    # CONTRASTIVE LEARNING (EXPERT FIX #1)
    # ============================================================
    # Compute MSE between prediction and action
    mse_loss = nn.MSELoss(reduction='none')(prediction, action)
    loss_per_sample = mse_loss.mean(dim=1)
    
    # Convert reward to tensor if needed
    if isinstance(reward, (int, float)):
        reward_tensor = torch.tensor([reward], device=prediction.device)
    else:
        reward_tensor = reward if torch.is_tensor(reward) else torch.tensor(reward, device=prediction.device)
    
    # Contrastive weighting based on reward quality
    # HIGH reward (>0.8): Positive sample - pull toward (weight = +1.0)
    # LOW reward (<0.3): Negative sample - push away (weight = -0.5)
    # MEDIUM reward: Weak signal (weight = reward - 0.5, can be pos or neg)
    contrastive_weight = torch.where(
        reward_tensor > 0.8,
        torch.ones_like(reward_tensor),  # Good: pull toward
        torch.where(
            reward_tensor < 0.3,
            torch.full_like(reward_tensor, -0.5),  # Bad: push away
            reward_tensor - 0.5  # Medium: weak signal
        )
    )
    
    # Apply contrastive weighting
    if loss_per_sample.dim() > 0 and contrastive_weight.dim() > 0:
        contrastive_loss = (loss_per_sample * contrastive_weight).mean()
    else:
        contrastive_loss = loss_per_sample * contrastive_weight
    
    # Value loss (TD error for advantage estimation)
    target_value = reward if not torch.is_tensor(reward) else reward
    if not torch.is_tensor(target_value):
        target_value = torch.tensor(target_value, device=value.device).reshape_as(value)
    value_loss = nn.MSELoss()(value, target_value.detach())
    
    # Entropy regularization to encourage diversity
    probs = torch.softmax(prediction / 0.1, dim=1)
    prediction_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
    
    # Diversity penalty based on prompt usage frequency
    diversity_penalty = 0.0
    if prompt_idx is not None and hasattr(model, 'get_diversity_penalty'):
        if isinstance(prompt_idx, (list, np.ndarray)):
            diversity_penalty = sum(model.get_diversity_penalty(idx) for idx in prompt_idx) / len(prompt_idx)
        else:
            diversity_penalty = model.get_diversity_penalty(prompt_idx)
        diversity_penalty = torch.tensor(diversity_penalty, device=prediction.device)
    
    # Final loss: contrastive loss + value loss - entropy bonus + diversity penalty
    final_loss = contrastive_loss + 0.5 * value_loss - entropy_coef * prediction_entropy + diversity_penalty_weight * diversity_penalty
    
    final_loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Update reward statistics for normalization
    if hasattr(model, 'update_reward_stats'):
        model.update_reward_stats(reward)
    
    # Update prompt usage for diversity tracking
    if prompt_idx is not None and hasattr(model, 'update_prompt_usage'):
        if isinstance(prompt_idx, (list, np.ndarray)):
            for idx in prompt_idx:
                model.update_prompt_usage(idx)
        else:
            model.update_prompt_usage(prompt_idx)
    
    # Return detailed metrics
    return {
        'loss': final_loss.item(),
        'contrastive_loss': contrastive_loss.item() if torch.is_tensor(contrastive_loss) else contrastive_loss,
        'value_loss': value_loss.item(),
        'entropy': prediction_entropy.item(),
        'diversity_penalty': diversity_penalty.item() if torch.is_tensor(diversity_penalty) else diversity_penalty,
        'mean_reward': reward.mean().item() if torch.is_tensor(reward) else reward,
        'contrastive_weight': contrastive_weight.mean().item() if torch.is_tensor(contrastive_weight) else contrastive_weight
    }

def main():
    parser = argparse.ArgumentParser(description="Train Agent Model (Contextual Bandit Step).")
    parser.add_argument("--input_state", type=str, required=True, help="Path to input 1024-dim vector (.npy).")
    parser.add_argument("--retrieved_action", type=str, required=True, help="Path to retrieved 256-dim vector (.npy).")
    parser.add_argument("--reward", type=float, required=True, help="Reward value [0, 1].")
    parser.add_argument("--prompt_idx", type=int, default=None, help="Index of selected prompt for diversity tracking.")
    parser.add_argument("--model_path", type=str, default=os.path.join(CURRENT_DIR, "agent_model.pth"), help="Path to save/load model.")
    parser.add_argument("--buffer_path", type=str, default=os.path.join(CURRENT_DIR, "replay_buffer.pkl"), help="Path to save/load replay buffer.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (EXPERT FIX: reduced from 5e-4 to 1e-4 for stability).")
    parser.add_argument("--use_replay", action="store_true", help="Use replay buffer for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for replay buffer training (increased for better gradients).")
    parser.add_argument("--buffer_capacity", type=int, default=5000, help="Replay buffer capacity (increased from 1000).")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs per experience (increased for convergence).")
    parser.add_argument("--entropy_coef", type=float, default=0.1, help="Entropy regularization coefficient (increased after bug fix).")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon-greedy exploration probability.")
    parser.add_argument("--diversity_penalty_weight", type=float, default=0.02, help="Weight for diversity penalty (reduced to avoid suppression).")
    parser.add_argument("--use_lr_scheduler", action="store_true", help="Use cosine annealing LR scheduler.")
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Early stopping patience (0=disabled).")
    
    args = parser.parse_args()

    # 1. Load Current Experience
    try:
        state_np = np.load(args.input_state)
        action_np = np.load(args.retrieved_action)
        
        # Flatten if necessary
        if len(state_np.shape) > 1:
            state_np = state_np.reshape(-1)
        if len(action_np.shape) > 1:
            action_np = action_np.reshape(-1)
            
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 2. Initialize Model
    device = torch.device("cpu")
    model = SonarAgent().to(device)
    
    # Load existing weights if available
    if os.path.exists(args.model_path):
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print("Loaded model weights successfully")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print("Starting from scratch")
            print(f"Warning: Could not load model, starting fresh. Error: {e}")
    else:
        print("No existing model found, starting with random initialization.")

    # 3. Initialize Replay Buffer
    replay_buffer = None
    if args.use_replay:
        replay_buffer = ReplayBuffer(
            capacity=args.buffer_capacity,
            state_dim=1024,
            action_dim=256
        )
        
        # Load existing buffer if available
        if os.path.exists(args.buffer_path):
            replay_buffer.load(args.buffer_path)
        
        # Add current experience to buffer
        replay_buffer.add(state_np, action_np, args.reward, metadata={'prompt_idx': args.prompt_idx})
        print(f"Replay buffer size: {len(replay_buffer)}")

    # 4. Optimizer with optional LR scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = None
    if args.use_lr_scheduler and args.use_replay:
        # Cosine annealing: gradually reduce LR
        total_steps = args.num_epochs * max(1, len(replay_buffer) // args.batch_size) if replay_buffer else args.num_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
        print(f"Using CosineAnnealingLR scheduler (T_max={total_steps})")
    
    # 5. Early stopping setup
    best_loss = float('inf')
    patience_counter = 0
    
    # 6. Training
    if args.use_replay and replay_buffer and len(replay_buffer) >= args.batch_size:
        # Train on mini-batches from replay buffer
        print(f"Training with replay buffer (batch_size={args.batch_size})...")
        
        total_loss = 0.0
        total_metrics = {'value_loss': 0.0, 'entropy': 0.0, 'diversity_penalty': 0.0}
        num_batches = max(1, len(replay_buffer) // args.batch_size)
        
        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            
            for batch_idx in range(num_batches):
                # Sample batch with diversity sampling
                states, actions, rewards, metadata_list = replay_buffer.sample(args.batch_size, diversity_sampling=True)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                
                # Extract prompt indices from metadata
                prompt_indices = [m.get('prompt_idx') for m in metadata_list if m and 'prompt_idx' in m]
                
                try:
                    # Training step with all improvements
                    model.train()
                    optimizer.zero_grad()
                    
                    # Forward pass with value head
                    predictions, values = model(states, add_noise=True, noise_std=0.15, epsilon=args.epsilon, return_value=True)
                    
                    # Check for NaN/Inf
                    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                        print("WARNING: NaN/Inf detected in predictions, skipping batch")
                        continue
                    
                    # MSE Loss
                    mse_loss = nn.MSELoss(reduction='none')(predictions, actions)
                    loss_per_sample = mse_loss.mean(dim=1)
                    
                    # Clip extreme rewards for stability (0.01 to 0.99 range)
                    rewards_clipped = torch.clamp(rewards, min=0.01, max=0.99)
                    
                    # Normalize rewards
                    normalized_rewards = torch.stack([model.normalize_reward(r) for r in rewards_clipped])
                    
                    # Additional stability check
                    if torch.isnan(normalized_rewards).any() or torch.isinf(normalized_rewards).any():
                        print("WARNING: NaN/Inf in normalized rewards, using raw rewards")
                        normalized_rewards = rewards_clipped
                    
                    # Reward-weighted loss
                    weighted_loss = (loss_per_sample * normalized_rewards).mean()
                    
                    # Value loss
                    value_loss = nn.MSELoss()(values.squeeze(), rewards_clipped)
                    
                    # Entropy regularization (FIXED: using softmax-based entropy measure)
                    # Treat predictions as logits, compute Shannon entropy over softmax distribution
                    # High entropy = diverse/uncertain predictions, Low entropy = collapsed/deterministic
                    probs = torch.softmax(predictions / 0.1, dim=1)  # Temperature=0.1 for sensitivity
                    prediction_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                    
                    # Diversity penalty
                    diversity_penalty = 0.0
                    if prompt_indices:
                        diversity_penalty = sum(model.get_diversity_penalty(idx) for idx in prompt_indices if idx is not None) / len(prompt_indices)
                        diversity_penalty = torch.tensor(diversity_penalty, device=predictions.device)
                    
                    # Final loss
                    final_loss = weighted_loss + 0.5 * value_loss - args.entropy_coef * prediction_entropy + args.diversity_penalty_weight * diversity_penalty
                    
                    # Check final loss validity
                    if torch.isnan(final_loss) or torch.isinf(final_loss):
                        print("WARNING: NaN/Inf in final loss, skipping batch")
                        continue
                    
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                except RuntimeError as e:
                    print(f"ERROR during training step: {e}")
                    print(f"Rewards: {rewards.cpu().numpy()}")
                    print(f"Skipping batch and continuing...")
                    continue
                
                if scheduler:
                    scheduler.step()
                
                # Update statistics
                for r in rewards:
                    model.update_reward_stats(r.item())
                for idx in prompt_indices:
                    if idx is not None:
                        model.update_prompt_usage(idx)
                
                epoch_loss += final_loss.item()
                total_loss += final_loss.item()
                total_metrics['value_loss'] += value_loss.item()
                total_metrics['entropy'] += prediction_entropy.item()
                total_metrics['diversity_penalty'] += diversity_penalty.item() if torch.is_tensor(diversity_penalty) else diversity_penalty
            
            avg_epoch_loss = epoch_loss / num_batches
            
            # Early stopping check
            if args.early_stopping_patience > 0:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}/{args.num_epochs}")
                        break
        
        avg_loss = total_loss / (num_batches * args.num_epochs)
        avg_metrics = {k: v / (num_batches * args.num_epochs) for k, v in total_metrics.items()}
        
        print(f"Training Complete. Avg Loss: {avg_loss:.6f}, Value Loss: {avg_metrics['value_loss']:.6f}, "
              f"Entropy: {avg_metrics['entropy']:.4f}, Diversity Penalty: {avg_metrics['diversity_penalty']:.4f}, "
              f"Current Reward: {args.reward:.4f}")
        
    else:
        # Train on single experience (original behavior)
        print(f"Training on single experience (Reward: {args.reward:.4f})...")
        
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
        action = torch.from_numpy(action_np).float().unsqueeze(0).to(device)
        
        total_metrics = {'loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'diversity_penalty': 0.0}
        for epoch in range(args.num_epochs):
            metrics = train_step(model, optimizer, state, action, args.reward, 
                               prompt_idx=args.prompt_idx,
                               entropy_coef=args.entropy_coef, 
                               epsilon=args.epsilon,
                               diversity_penalty_weight=args.diversity_penalty_weight)
            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v
        
        avg_metrics = {k: v / args.num_epochs for k, v in total_metrics.items()}
        print(f"Training Step Complete. Loss: {avg_metrics['loss']:.6f}, "
              f"Value Loss: {avg_metrics['value_loss']:.6f}, "
              f"Entropy: {avg_metrics['entropy']:.4f}, "
              f"Diversity Penalty: {avg_metrics['diversity_penalty']:.4f}, "
              f"Reward: {args.reward:.4f}")
    
    # 7. Save Model with checkpoint info
    # Save full model state dict (includes tensors)
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")
    
    # 8. Save Replay Buffer
    if args.use_replay and replay_buffer is not None:
        replay_buffer.save(args.buffer_path)
    
    # 9. Checkpoint Management (every 100 sentences)
    checkpoint_dir = os.path.join(AGENT_DIR, "checkpoint")
    sentence_counter_file = os.path.join(checkpoint_dir, "sentence_counter.txt")
    checkpoint_interval = 100
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Read current sentence count
    if os.path.exists(sentence_counter_file):
        with open(sentence_counter_file, 'r') as f:
            sentence_count = int(f.read().strip())
    else:
        sentence_count = 0
    
    # Increment sentence count
    sentence_count += 1
    
    # Save updated count
    with open(sentence_counter_file, 'w') as f:
        f.write(str(sentence_count))
    
    # Save checkpoint every 100 sentences
    if sentence_count % checkpoint_interval == 0:
        print(f"[CHECKPOINT] Saving checkpoint at sentence {sentence_count}...")
        
        # Save model checkpoint
        checkpoint_model_path = os.path.join(checkpoint_dir, f"agent_model_sentence_{sentence_count}.pth")
        shutil.copy2(args.model_path, checkpoint_model_path)
        print(f"[CHECKPOINT] Model saved to: {checkpoint_model_path}")
        
        # Save replay buffer checkpoint if it exists
        if args.use_replay and replay_buffer is not None and os.path.exists(args.buffer_path):
            checkpoint_buffer_path = os.path.join(checkpoint_dir, f"replay_buffer_sentence_{sentence_count}.pkl")
            shutil.copy2(args.buffer_path, checkpoint_buffer_path)
            print(f"[CHECKPOINT] Replay buffer saved to: {checkpoint_buffer_path}")

if __name__ == "__main__":
    main()
