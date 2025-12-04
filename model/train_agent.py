import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(CURRENT_DIR)
if AGENT_DIR not in sys.path:
    sys.path.append(AGENT_DIR)

from model.agent_model import SonarAgent

def train_step(model, optimizer, state, action, reward):
    model.eval()
    optimizer.zero_grad()

    prediction = model(state)
    mse_loss = nn.MSELoss(reduction='none')(prediction, action)
    loss_per_sample = mse_loss.mean(dim=1)
    weighted_loss = loss_per_sample * reward
    final_loss = weighted_loss.mean()

    final_loss.backward()
    optimizer.step()

    return final_loss.item()

def main():
    parser = argparse.ArgumentParser(description="Train Agent Model (Contextual Bandit Step).")
    parser.add_argument("--input_state", type=str, required=True, help="Path to input 1024-dim vector (.npy).")
    parser.add_argument("--retrieved_action", type=str, required=True, help="Path to retrieved 100-dim vector (.npy).")
    parser.add_argument("--reward", type=float, required=True, help="Reward value [0, 1].")
    parser.add_argument("--model_path", type=str, default=os.path.join(CURRENT_DIR, "agent_model.pth"), help="Path to save/load model.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")

    args = parser.parse_args()

    try:
        state_np = np.load(args.input_state)
        action_np = np.load(args.retrieved_action)

        state = torch.from_numpy(state_np).float()
        action = torch.from_numpy(action_np).float()

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    device = torch.device("cpu")

    model = SonarAgent().to(device)

    if os.path.exists(args.model_path):
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        except Exception as e:
            print(f"Warning: Could not load model, starting fresh. Error: {e}")
    else:
        print("No existing model found, starting fresh.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    state = state.to(device)
    action = action.to(device)

    epochs = 1
    final_loss = 0.0
    for epoch in range(epochs):
        loss = train_step(model, optimizer, state, action, args.reward)
        final_loss = loss

    print(f"Training Step Complete. Loss: {final_loss:.6f}, Reward: {args.reward:.4f}")

    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main()
