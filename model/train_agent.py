import torch
import torch.nn as nn
import torch.nn.functional as F
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


def compute_adaptive_temperature(rewards, base_temp=0.1, min_temp=0.05, max_temp=0.5):
    """
    Compute adaptive temperature based on reward distribution.
    
    - When rewards are clustered (low variance), use higher temperature for softer gradients
    - When rewards are spread (high variance), use lower temperature for sharper discrimination
    
    Args:
        rewards: Tensor of rewards in the batch
        base_temp: Base temperature value
        min_temp: Minimum temperature (most aggressive)
        max_temp: Maximum temperature (softest)
        
    Returns:
        float: Adaptive temperature value
    """
    reward_std = rewards.std().item()
    reward_range = (rewards.max() - rewards.min()).item()
    
    # If rewards are very similar, increase temperature to avoid gradient saturation
    if reward_range < 0.2 or reward_std < 0.1:
        temp = max_temp
    elif reward_range > 0.5 and reward_std > 0.2:
        temp = min_temp
    else:
        # Linear interpolation based on variance
        temp = base_temp + (max_temp - base_temp) * (1 - min(reward_std / 0.3, 1.0))
    
    return max(min_temp, min(temp, max_temp))


def compute_info_nce_loss_improved(predictions, actions, rewards, base_temperature=0.1, use_reward_weighting=True):
    """
    InfoNCE contrastive loss with adaptive temperature and reward weighting.
    
    Args:
        predictions: Agent predictions (batch_size, dim)
        actions: Retrieved actions (batch_size, dim)  
        rewards: Reward values (batch_size,)
        base_temperature: Base temperature for softmax
        use_reward_weighting: If True, use soft reward weighting instead of hard split
        
    Returns:
        InfoNCE loss scalar
    """
    batch_size = predictions.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device)
    
    # Normalize all vectors for cosine similarity
    predictions = F.normalize(predictions, dim=1)
    actions = F.normalize(actions, dim=1)
    
    # Compute adaptive temperature
    temperature = compute_adaptive_temperature(rewards, base_temp=base_temperature)
    
    # Compute all pairwise similarities: (batch, batch)
    similarities = torch.mm(predictions, actions.T) / temperature
    
    if use_reward_weighting:
        # Soft reward weighting: higher reward = more "positive"
        # Normalize rewards to [0, 1] range within batch
        rewards_norm = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
        
        # Create soft labels: each prediction should be closest to high-reward actions
        # Weight matrix: how much each action should attract each prediction
        # High reward actions attract more, low reward actions repel
        reward_weights = rewards_norm.unsqueeze(0).expand(batch_size, -1)  # (batch, batch)
        
        # Diagonal elements (self-similarity) should have highest weight if reward is good
        # Create target distribution: softmax over reward-weighted similarities
        target_logits = similarities * reward_weights
        
        # Compute log_softmax over actions for each prediction
        log_probs = F.log_softmax(similarities, dim=1)
        
        # Weight the log probabilities by reward (encourage matching high-reward actions)
        # Each prediction should maximize similarity to high-reward actions
        weighted_log_probs = log_probs * reward_weights
        
        # Loss: negative weighted log probability (encourage high similarity to high-reward actions)
        loss = -weighted_log_probs.sum(dim=1).mean()
        
    else:
        # Fallback: standard InfoNCE with diagonal as positive
        # Each prediction's positive is its own action
        labels = torch.arange(batch_size, device=predictions.device)
        loss = F.cross_entropy(similarities, labels)
    
    return loss


def compute_info_nce_loss(predictions, positive_actions, negative_actions, temperature=0.1):
    """
    Compute InfoNCE contrastive loss for proper contrastive learning.
    
    This is the correct way to do contrastive learning:
    - Pull predictions toward positive (high reward) actions
    - Push predictions away from negative (low reward) actions
    
    Args:
        predictions: Agent predictions (batch_size, dim)
        positive_actions: Actions with good rewards (n_pos, dim)
        negative_actions: Actions with bad rewards (n_neg, dim)
        temperature: Softmax temperature for sharpness
        
    Returns:
        InfoNCE loss scalar
    """
    # Normalize all vectors for cosine similarity
    predictions = F.normalize(predictions, dim=1)
    positive_actions = F.normalize(positive_actions, dim=1)
    negative_actions = F.normalize(negative_actions, dim=1)
    
    # Compute similarities
    pos_sim = torch.mm(predictions, positive_actions.T) / temperature  # (batch, n_pos)
    neg_sim = torch.mm(predictions, negative_actions.T) / temperature  # (batch, n_neg)
    
    # For each prediction, compute log-softmax over positives vs negatives
    # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    pos_mean = pos_sim.mean(dim=1, keepdim=True)  # Average positive similarity
    all_sim = torch.cat([pos_mean, neg_sim], dim=1)  # (batch, 1+n_neg)
    
    # Labels: first column (index 0) is positive
    labels = torch.zeros(predictions.shape[0], dtype=torch.long, device=predictions.device)
    
    loss = F.cross_entropy(all_sim, labels)
    return loss


def train_step(model, optimizer, state, action, reward, prompt_idx=None, 
               entropy_coef=0.01, epsilon=0.0, diversity_penalty_weight=0.1,
               baseline_reward=0.5, negative_actions=None, all_rewards=None):
    """
    Perform a single training step using policy gradient with InfoNCE regularization.
    
    Args:
        model: The SonarAgent model.
        optimizer: The optimizer.
        state: Input tensor (batch_size, 1024).
        action: Retrieved prompt vector (batch_size, 256).
        reward: Scalar or tensor reward [0, 1].
        prompt_idx: Index of the selected prompt (for diversity tracking).
        entropy_coef: Coefficient for entropy regularization.
        epsilon: Epsilon-greedy exploration probability.
        diversity_penalty_weight: Weight for diversity penalty.
        baseline_reward: Running average reward for advantage estimation.
        negative_actions: Optional tensor of negative examples (low reward actions).
        all_rewards: Optional tensor of all rewards in batch for normalization.
        
    Returns:
        dict: Dictionary with loss components and metrics.
    """
    model.train() 
    optimizer.zero_grad()
    
    # Forward pass - get prediction and value estimate
    prediction, value = model(state, add_noise=True, noise_std=0.1, epsilon=epsilon, return_value=True)
    
    # Convert reward to tensor
    if isinstance(reward, (int, float)):
        reward_tensor = torch.tensor([reward], device=prediction.device, dtype=torch.float32)
    else:
        reward_tensor = reward.clone().detach().float() if torch.is_tensor(reward) else torch.tensor(reward, device=prediction.device, dtype=torch.float32)
    
    # Policy gradient loss (REINFORCE-style)
    # ============================================================
    # Compute advantage = reward - baseline (with normalization)
    advantage = reward_tensor - baseline_reward
    
    # Normalize advantage if we have multiple samples
    if all_rewards is not None and len(all_rewards) > 1:
        all_rewards_t = torch.tensor(all_rewards, device=prediction.device, dtype=torch.float32)
        adv_std = all_rewards_t.std() + 1e-8
        advantage = advantage / adv_std
    
    # Cosine similarity between prediction and retrieved action
    # This measures how well the prediction matches the chosen action
    cos_sim = F.cosine_similarity(prediction, action, dim=1)
    
    # Policy gradient: maximize similarity for high advantage, minimize for low
    # log(similarity) * advantage -> gradient pushes toward good actions
    log_prob = torch.log(cos_sim.clamp(min=1e-8))
    pg_loss = -(log_prob * advantage).mean()
    
    # Contrastive loss (InfoNCE when negatives available)
    # ============================================================
    contrastive_loss = torch.tensor(0.0, device=prediction.device)
    if negative_actions is not None and len(negative_actions) > 0:
        # True contrastive learning: pull toward good, push from bad
        positive_mask = reward_tensor > baseline_reward
        if positive_mask.any():
            positive_actions = action[positive_mask]
            contrastive_loss = compute_info_nce_loss(
                prediction, positive_actions, negative_actions, temperature=0.1
            )
    
    # Value loss (for advantage estimation)
    # ============================================================
    value_target = reward_tensor.detach().view_as(value)
    value_loss = F.mse_loss(value, value_target)
    
    # Prevent value head collapse: add small regularization to keep it active
    # Only compute variance if we have multiple samples
    if value.numel() > 1:
        value_reg = -0.01 * value.var()
    else:
        value_reg = torch.tensor(0.0, device=prediction.device)
    
    # Entropy regularization (prevent policy collapse)
    # ============================================================
    # Use prediction variance as a proxy for entropy
    pred_std = prediction.std(dim=1).mean()
    # Also compute output diversity across the batch
    if prediction.shape[0] > 1:
        batch_diversity = F.pdist(prediction).mean()
    else:
        batch_diversity = pred_std
    entropy_bonus = pred_std + 0.5 * batch_diversity
    
    # Diversity penalty
    # ============================================================
    diversity_penalty = torch.tensor(0.0, device=prediction.device)
    if prompt_idx is not None and hasattr(model, 'get_diversity_penalty'):
        if isinstance(prompt_idx, (list, np.ndarray)):
            pen = sum(model.get_diversity_penalty(idx) for idx in prompt_idx if idx is not None)
            diversity_penalty = torch.tensor(pen / max(1, len(prompt_idx)), device=prediction.device)
        else:
            diversity_penalty = torch.tensor(model.get_diversity_penalty(prompt_idx), device=prediction.device)
    
    # Final loss combination
    # ============================================================
    # Balance the loss components properly
    final_loss = (
        1.0 * pg_loss +                          # Main policy gradient
        0.3 * contrastive_loss +                 # Contrastive (when available)
        0.5 * value_loss +                       # Value estimation
        value_reg +                              # Prevent value collapse
        -entropy_coef * entropy_bonus +          # Encourage exploration
        diversity_penalty_weight * diversity_penalty  # Prompt diversity
    )
    
    # Check for NaN/Inf before backward
    if torch.isnan(final_loss) or torch.isinf(final_loss):
        print(f"WARNING: Invalid loss detected! pg_loss={pg_loss.item():.4f}, "
              f"contrastive={contrastive_loss.item():.4f}, value={value_loss.item():.4f}")
        return {
            'loss': 0.0, 'policy_loss': 0.0, 'contrastive_loss': 0.0,
            'value_loss': 0.0, 'entropy': 0.0, 'diversity_penalty': 0.0,
            'mean_reward': reward_tensor.mean().item(), 'cos_similarity': cos_sim.mean().item()
        }
    
    final_loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Update reward statistics
    if hasattr(model, 'update_reward_stats'):
        for r in (reward_tensor if reward_tensor.dim() > 0 else [reward_tensor]):
            model.update_reward_stats(r.item())
    
    # Update prompt usage for diversity
    if prompt_idx is not None and hasattr(model, 'update_prompt_usage'):
        indices = prompt_idx if isinstance(prompt_idx, (list, np.ndarray)) else [prompt_idx]
        for idx in indices:
            if idx is not None:
                model.update_prompt_usage(idx)
    
    return {
        'loss': final_loss.item(),
        'policy_loss': pg_loss.item(),
        'contrastive_loss': contrastive_loss.item() if torch.is_tensor(contrastive_loss) else 0.0,
        'value_loss': value_loss.item(),
        'entropy': entropy_bonus.item(),
        'diversity_penalty': diversity_penalty.item() if torch.is_tensor(diversity_penalty) else 0.0,
        'mean_reward': reward_tensor.mean().item(),
        'cos_similarity': cos_sim.mean().item(),
        'advantage': advantage.mean().item() if advantage.dim() > 0 else advantage.item()
    }

def main():
    parser = argparse.ArgumentParser(description="Train Agent Model (Contextual Bandit Step).")
    parser.add_argument("--input_state", type=str, required=True, help="Path to input 1024-dim vector (.npy).")
    parser.add_argument("--retrieved_action", type=str, required=True, help="Path to retrieved 256-dim vector (.npy).")
    parser.add_argument("--reward", type=float, required=True, help="Reward value [0, 1].")
    parser.add_argument("--baseline_reward", type=float, default=0.5, help="Baseline reward for advantage estimation.")
    parser.add_argument("--prompt_idx", type=int, default=None, help="Index of selected prompt for diversity tracking.")
    parser.add_argument("--model_path", type=str, default=os.path.join(CURRENT_DIR, "agent_model.pth"), help="Path to save/load model.")
    parser.add_argument("--buffer_path", type=str, default=os.path.join(CURRENT_DIR, "replay_buffer.pkl"), help="Path to save/load replay buffer.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--use_replay", action="store_true", help="Use replay buffer for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for replay buffer training.")
    parser.add_argument("--buffer_capacity", type=int, default=5000, help="Replay buffer capacity.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs per experience.")
    parser.add_argument("--entropy_coef", type=float, default=0.1, help="Entropy regularization coefficient.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon-greedy exploration probability.")
    parser.add_argument("--diversity_penalty_weight", type=float, default=0.02, help="Weight for diversity penalty.")
    parser.add_argument("--manifold_alignment_weight", type=float, default=0.2, help="Weight for manifold alignment loss.")
    parser.add_argument("--centroid_proximity_weight", type=float, default=0.1, help="Weight for centroid proximity regularization.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients.")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of warmup epochs for LR scheduler.")
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

    # 4. Optimizer with improved LR scheduler (warmup + cosine decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = None
    warmup_scheduler = None
    if args.use_lr_scheduler and args.use_replay:
        num_batches = max(1, len(replay_buffer) // args.batch_size) if replay_buffer else 1
        total_steps = args.num_epochs * num_batches
        warmup_steps = args.warmup_epochs * num_batches
        
        # Create warmup + cosine annealing scheduler
        # Warmup: linearly increase LR from lr/10 to lr
        # Then cosine annealing from lr to lr/10
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return 0.1 + 0.9 * (step / max(1, warmup_steps))
            else:
                # Cosine annealing after warmup
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.1 + 0.9 * (0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"Using Warmup ({warmup_steps} steps) + Cosine Annealing LR scheduler (total_steps={total_steps})")
    
    # 5. Early stopping setup
    best_loss = float('inf')
    patience_counter = 0
    
    # 6. Training
    if args.use_replay and replay_buffer and len(replay_buffer) >= args.batch_size:
        # Train on mini-batches from replay buffer with improved policy gradient
        print(f"Training with replay buffer (batch_size={args.batch_size})...")
        
        total_loss = 0.0
        total_metrics = {
            'policy_loss': 0.0, 'contrastive_loss': 0.0, 'value_loss': 0.0, 
            'entropy': 0.0, 'diversity_penalty': 0.0, 'cos_similarity': 0.0,
            'manifold_loss': 0.0, 'proximity_loss': 0.0
        }
        num_batches = max(1, len(replay_buffer) // args.batch_size)
        
        # Get all rewards for baseline computation
        all_rewards = [replay_buffer.rewards[i] for i in range(len(replay_buffer))]
        running_baseline = np.mean(all_rewards) if all_rewards else 0.5
        
        # Gradient accumulation counter
        accumulation_steps = args.gradient_accumulation_steps
        accumulated_loss = 0.0
        global_step = 0
        
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
                    # Policy gradient + contrastive + manifold alignment training
                    model.train()
                    
                    # Only zero gradients at accumulation boundaries
                    if global_step % accumulation_steps == 0:
                        optimizer.zero_grad()
                    
                    # Forward pass
                    predictions, values = model(states, add_noise=True, noise_std=0.1, epsilon=args.epsilon, return_value=True)
                    
                    # Check for NaN/Inf
                    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                        print("WARNING: NaN/Inf detected in predictions, skipping batch")
                        continue
                    
                    # Clip extreme rewards
                    rewards_clipped = torch.clamp(rewards, min=0.01, max=0.99)
                    
                    # Policy gradient loss
                    # Compute advantage with running baseline
                    advantages = rewards_clipped - running_baseline
                    
                    # Normalize advantages within batch
                    adv_std = advantages.std() + 1e-8
                    advantages_norm = advantages / adv_std
                    
                    # Cosine similarity between predictions and actions
                    cos_sim = F.cosine_similarity(predictions, actions, dim=1)
                    
                    # Policy gradient: -log(sim) * advantage
                    log_prob = torch.log(cos_sim.clamp(min=1e-8))
                    pg_loss = -(log_prob * advantages_norm).mean()
                    
                    # Contrastive learning (with adaptive temperature)
                    # Use the new improved InfoNCE with reward weighting
                    contrastive_loss = compute_info_nce_loss_improved(
                        predictions, actions, rewards_clipped, 
                        base_temperature=0.1, use_reward_weighting=True
                    )
                    
                    # Manifold alignment loss
                    # ============================================================
                    manifold_loss = torch.tensor(0.0, device=device)
                    if hasattr(model, 'compute_manifold_alignment_loss'):
                        manifold_loss = model.compute_manifold_alignment_loss(predictions)
                    
                    # Centroid proximity regularization
                    # ============================================================
                    proximity_loss = torch.tensor(0.0, device=device)
                    if hasattr(model, 'compute_centroid_proximity_regularization'):
                        proximity_loss = model.compute_centroid_proximity_regularization(predictions, soft_constraint=True)
                    
                    # Value loss with regularization
                    value_loss = F.mse_loss(values.squeeze(), rewards_clipped)
                    value_reg = -0.01 * values.var()  # Prevent value head collapse
                    
                    # Entropy bonus (prevent policy collapse)
                    pred_std = predictions.std(dim=1).mean()
                    batch_diversity = F.pdist(predictions).mean() if predictions.shape[0] > 1 else pred_std
                    entropy_bonus = pred_std + 0.5 * batch_diversity
                    
                    # Diversity penalty
                    diversity_penalty = torch.tensor(0.0, device=device)
                    if prompt_indices:
                        pen = sum(model.get_diversity_penalty(idx) for idx in prompt_indices if idx is not None)
                        diversity_penalty = torch.tensor(pen / max(1, len(prompt_indices)), device=device)
                    
                    # Combined loss
                    # ============================================================
                    final_loss = (
                        1.0 * pg_loss +
                        0.3 * contrastive_loss +
                        args.manifold_alignment_weight * manifold_loss +
                        args.centroid_proximity_weight * proximity_loss +
                        0.5 * value_loss +
                        value_reg +
                        -args.entropy_coef * entropy_bonus +
                        args.diversity_penalty_weight * diversity_penalty
                    )
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = final_loss / accumulation_steps
                    
                    if torch.isnan(final_loss) or torch.isinf(final_loss):
                        print("WARNING: NaN/Inf in final loss, skipping batch")
                        continue
                    
                    # Backward pass with scaled loss for gradient accumulation
                    scaled_loss.backward()
                    accumulated_loss += final_loss.item()
                    global_step += 1
                    
                    # Only update weights at accumulation boundaries
                    if global_step % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        if scheduler:
                            scheduler.step()
                    
                except RuntimeError as e:
                    print(f"ERROR during training step: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Update running baseline (exponential moving average)
                running_baseline = 0.99 * running_baseline + 0.01 * rewards_clipped.mean().item()
                
                # Update statistics
                for r in rewards:
                    model.update_reward_stats(r.item())
                for idx in prompt_indices:
                    if idx is not None:
                        model.update_prompt_usage(idx)
                
                epoch_loss += final_loss.item()
                total_loss += final_loss.item()
                total_metrics['policy_loss'] += pg_loss.item()
                total_metrics['contrastive_loss'] += contrastive_loss.item() if torch.is_tensor(contrastive_loss) else 0.0
                total_metrics['value_loss'] += value_loss.item()
                total_metrics['entropy'] += entropy_bonus.item()
                total_metrics['diversity_penalty'] += diversity_penalty.item() if torch.is_tensor(diversity_penalty) else 0.0
                total_metrics['cos_similarity'] += cos_sim.mean().item()
                total_metrics['manifold_loss'] += manifold_loss.item() if torch.is_tensor(manifold_loss) else 0.0
                total_metrics['proximity_loss'] += proximity_loss.item() if torch.is_tensor(proximity_loss) else 0.0
            
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
        
        # Handle remaining accumulated gradients
        if global_step % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_batches = num_batches * args.num_epochs
        avg_loss = total_loss / total_batches
        avg_metrics = {k: v / total_batches for k, v in total_metrics.items()}
        
        print(f"Training Complete. Avg Loss: {avg_loss:.6f}, Policy Loss: {avg_metrics['policy_loss']:.6f}, "
              f"Contrastive: {avg_metrics['contrastive_loss']:.4f}, Value Loss: {avg_metrics['value_loss']:.6f}, "
              f"Entropy: {avg_metrics['entropy']:.4f}, CosSim: {avg_metrics['cos_similarity']:.4f}, "
              f"Current Reward: {args.reward:.4f}")
        
    else:
        # Train on single experience (original behavior) - with improved policy gradient
        print(f"Training on single experience (Reward: {args.reward:.4f})...")
        
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
        action = torch.from_numpy(action_np).float().unsqueeze(0).to(device)
        
        # Compute advantage for logging
        advantage = args.reward - args.baseline_reward
        print(f"Advantage (reward - baseline): {advantage:.4f}")
        
        total_metrics = {
            'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 
            'entropy': 0.0, 'diversity_penalty': 0.0, 'cos_similarity': 0.0
        }
        for epoch in range(args.num_epochs):
            metrics = train_step(model, optimizer, state, action, args.reward, 
                               prompt_idx=args.prompt_idx,
                               entropy_coef=args.entropy_coef, 
                               epsilon=args.epsilon,
                               diversity_penalty_weight=args.diversity_penalty_weight,
                               baseline_reward=args.baseline_reward)
            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v
        
        avg_metrics = {k: v / args.num_epochs for k, v in total_metrics.items()}
        cos_sim = avg_metrics.get('cos_similarity', 0.0)
        print(f"Training Step Complete. Loss: {avg_metrics['loss']:.6f}, "
              f"Policy Loss: {avg_metrics.get('policy_loss', 0.0):.6f}, "
              f"Value Loss: {avg_metrics['value_loss']:.6f}, "
              f"Entropy: {avg_metrics['entropy']:.4f}, "
              f"CosSim: {cos_sim:.4f}, "
              f"Reward: {args.reward:.4f}, Advantage: {advantage:.4f}")
    
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
