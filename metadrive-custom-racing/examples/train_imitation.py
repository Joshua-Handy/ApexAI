"""
Train an agent using imitation learning (behavioral cloning).

This script trains a new agent to imitate demonstrations collected from
an expert agent. Uses supervised learning to predict expert actions.

Usage:
    python train_imitation.py --demonstrations demonstrations_agent_0v2.pkl --agent F16
    python train_imitation.py --demonstrations demonstrations_agent_0v2.pkl --agent F16 --epochs 50
"""
import os
import sys
import argparse
import pickle
from pathlib import Path

# Add src to path
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class DemonstrationDataset(Dataset):
    """Dataset of (observation, action) pairs for behavioral cloning."""

    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def load_demonstrations(demo_path):
    """Load demonstration data from pickle file."""

    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demonstrations not found: {demo_path}")

    print(f"Loading demonstrations from: {demo_path}")
    with open(demo_path, 'rb') as f:
        demonstrations = pickle.load(f)

    print(f"[OK] Loaded demonstrations:")
    print(f"   Episodes: {demonstrations['metadata']['num_episodes']}")
    print(f"   Total steps: {demonstrations['metadata']['total_steps']}")
    print(f"   Successful episodes: {demonstrations['metadata']['successful_episodes']}")
    print(f"   Observation shape: {demonstrations['metadata']['obs_shape']}")
    print(f"   Action shape: {demonstrations['metadata']['action_shape']}")

    return demonstrations


def create_policy_network(obs_shape, action_shape):
    """Create a policy network matching SB3's PPO architecture."""

    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from gymnasium import spaces

    # Create dummy env spaces for SB3 policy
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=action_shape, dtype=np.float32)

    # Create PPO policy (we'll only use the actor part)
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    policy = ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        **policy_kwargs
    )

    return policy


def train_behavioral_cloning(
    policy,
    demonstrations,
    epochs=50,
    batch_size=256,
    learning_rate=3e-4,
    validation_split=0.1,
    verbose=True
):
    """Train policy using behavioral cloning (supervised learning)."""

    observations = demonstrations['observations']
    actions = demonstrations['actions']

    # Split into train/validation
    split_idx = int(len(observations) * (1 - validation_split))
    train_obs, val_obs = observations[:split_idx], observations[split_idx:]
    train_actions, val_actions = actions[:split_idx], actions[split_idx:]

    # Create datasets
    train_dataset = DemonstrationDataset(train_obs, train_actions)
    val_dataset = DemonstrationDataset(val_obs, val_actions)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup optimizer - only train the actor (policy) part
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Loss function: Mean Squared Error for continuous actions
    criterion = nn.MSELoss()

    print(f"\n{'='*70}")
    print(f"TRAINING BEHAVIORAL CLONING")
    print(f"{'='*70}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        # Training
        policy.train()
        train_loss = 0.0
        train_batches = 0

        for batch_obs, batch_actions in train_loader:
            optimizer.zero_grad()

            # Forward pass through policy
            # Use policy's action_net (the actor part)
            features = policy.extract_features(batch_obs)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            predicted_actions = policy.action_net(latent_pi)

            # Compute loss
            loss = criterion(predicted_actions, batch_actions)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches

        # Validation
        policy.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_obs, batch_actions in val_loader:
                features = policy.extract_features(batch_obs)
                latent_pi = policy.mlp_extractor.forward_actor(features)
                predicted_actions = policy.action_net(latent_pi)

                loss = criterion(predicted_actions, batch_actions)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        # Print progress
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} @ epoch {best_epoch + 1}")

    print(f"\n{'='*70}")
    print(f"[OK] TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")
    print(f"{'='*70}\n")

    return policy


def save_as_ppo_model(policy, output_path, obs_shape, action_shape):
    """Save the trained policy as a PPO model for compatibility."""

    # Create a dummy environment for PPO initialization
    import gymnasium as gym
    from gymnasium import spaces

    class DummyEnv(gym.Env):
        def __init__(self, obs_shape, action_shape):
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=action_shape, dtype=np.float32)

        def reset(self, **kwargs):
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

    dummy_env = DummyEnv(obs_shape, action_shape)

    # Create a PPO model with the trained policy
    model = PPO(
        policy=ActorCriticPolicy,
        env=dummy_env,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Fixed: dict not list
        ),
        learning_rate=3e-4,
    )

    # Replace the policy with our trained one
    model.policy = policy

    # Save
    model.save(output_path)
    print(f"[OK] Model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train agent using imitation learning')
    parser.add_argument('--demonstrations', type=str, required=True, help='Path to demonstrations pickle file')
    parser.add_argument('--agent', type=str, required=True, help='Name for new agent (e.g., F16)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    # Setup paths
    demo_path = os.path.join(os.path.dirname(__file__), args.demonstrations)
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), '..', 'results', f'results_agent_{args.agent}')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'{args.agent}_custom_speedway.zip')

    # Load demonstrations
    demonstrations = load_demonstrations(demo_path)

    # Create policy network
    obs_shape = demonstrations['metadata']['obs_shape']
    action_shape = demonstrations['metadata']['action_shape']

    print(f"\nCreating policy network...")
    policy = create_policy_network(obs_shape, action_shape)

    # Train using behavioral cloning
    trained_policy = train_behavioral_cloning(
        policy,
        demonstrations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Save as PPO-compatible model
    print(f"Saving model...")
    save_as_ppo_model(trained_policy, output_path, obs_shape, action_shape)

    print(f"\n[OK] Imitation learning complete!")
    print(f"   Model saved to: {output_path}")
    print(f"\n[NOTE] Next steps:")
    print(f"   1. Test the agent:")
    print(f"      python race_side_by_side_curve.py --agents {args.agent} agent_0v2 --episodes 3 --results-dir ../results")
    print(f"   2. (Optional) Fine-tune with PPO:")
    print(f"      python train_multi_custom_speedway_curve.py --agent-names {args.agent} --timesteps 100000 --pretrained {output_path}")


if __name__ == '__main__':
    main()
