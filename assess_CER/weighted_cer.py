import argparse
import sys

def calculate_reward(cer):

    cer = float(cer)
    
    # Calculate reward
    # If CER is 0.0 (perfect), Reward is 1.0
    # If CER is 0.5, Reward is 0.5
    # If CER >= 1.0, Reward is 0.0
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
