import argparse
import sys

def calculate_reward(cer):
    cer = float(cer)
    reward = max(0.0, 1.0 - cer)
    return reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Reward from CER.")
    parser.add_argument("cer", type=float, help="Character Error Rate (CER) value.")
    parser.add_argument("--output_reward", type=str, help="Path to save the reward value.")

    args = parser.parse_args()

    try:
        reward = calculate_reward(args.cer)
        print(f"Reward: {reward:.4f}")

        if args.output_reward:
            with open(args.output_reward, 'w') as f:
                f.write(f"{reward:.4f}")

    except Exception as e:
        print(f"Error calculating reward: {e}")
        sys.exit(1)
