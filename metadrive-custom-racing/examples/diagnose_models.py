#!/usr/bin/env python3
"""
Diagnose what's wrong with the trained models
"""
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from stable_baselines3 import PPO

def diagnose_models():
    """Load models and test their outputs with dummy observations"""
    
    models_dir = "multi_agent_racing_results/final_models"
    
    print("🔍 Diagnosing trained models...")
    
    # Load one model
    try:
        model_path = os.path.join(models_dir, "agent0_final.zip")
        print(f"📂 Loading model: {model_path}")
        model = PPO.load(model_path)
        print("✅ Model loaded successfully!")
        
        # Create dummy observation with CORRECT size (107 as the error message showed)
        dummy_obs = np.random.random(107)  # Correct observation size
        
        print(f"🧪 Testing model with correct observation shape: {dummy_obs.shape}")
        
        # Test model prediction
        for i in range(5):
            action, _ = model.predict(dummy_obs, deterministic=True)
            print(f"   Test {i+1}: action = {action} (throttle={action[0]:.3f}, steering={action[1]:.3f})")
        
        # Test with different observations
        print("\n🧪 Testing with different observation patterns:")
        
        # All zeros
        zero_obs = np.zeros(107)
        action, _ = model.predict(zero_obs, deterministic=True)
        print(f"   Zero obs: action = {action} (throttle={action[0]:.3f}, steering={action[1]:.3f})")
        
        # All ones
        ones_obs = np.ones(107)
        action, _ = model.predict(ones_obs, deterministic=True)
        print(f"   Ones obs: action = {action} (throttle={action[0]:.3f}, steering={action[1]:.3f})")
        
        # Check model architecture
        print(f"\n🏗️  Model info:")
        print(f"   Policy: {model.policy}")
        print(f"   Action space: {model.action_space}")
        print(f"   Observation space: {model.observation_space}")
        
    except Exception as e:
        print(f"❌ Error loading/testing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_models()