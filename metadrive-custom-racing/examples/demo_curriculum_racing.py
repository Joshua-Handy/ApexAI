"""
MetaDrive Multi-Agent Racing Demo

This script demonstrates how to run the curriculum-based multi-agent racing system.
Run this to see how 4 agents learn to race competitively from basic safety to full racing.

Usage:
    python demo_curriculum_racing.py               # Start training
    python demo_curriculum_racing.py --test        # Test existing models
    python demo_curriculum_racing.py --wandb       # Train with logging
"""
import os
import sys
import argparse

def main():
    """Demo the curriculum-based multi-agent racing system."""
    parser = argparse.ArgumentParser(description='MetaDrive Multi-Agent Racing Demo')
    parser.add_argument('--test', action='store_true',
                       help='Test existing models instead of training')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Training timesteps (default: 500K for demo)')
    parser.add_argument('--agents', type=int, default=4,
                       help='Number of racing agents (default: 4)')
    
    args = parser.parse_args()
    
    print("🏁 MetaDrive Multi-Agent Racing with Curriculum Learning")
    print("=" * 60)
    print()
    print("🎯 This demo showcases:")
    print("   ✅ 4 agents learning to race competitively")
    print("   ✅ Curriculum learning: Safety → Control → Speed → Racing")
    print("   ✅ Per-agent done logic (no respawn loops)")
    print("   ✅ Racing grid formation like real motorsports")
    print("   ✅ Progressive reward shaping")
    print("   ✅ Shared policy for coordinated behavior")
    print()
    
    if args.test:
        print("🧪 TESTING MODE: Loading and testing existing models...")
        print()
        print("Commands to test models:")
        print("   python test_curriculum_racing.py --models-dir multi_agent_racing_results")
        print("   python test_curriculum_racing.py --phase racing --races-per-phase 5")
        print("   python test_curriculum_racing.py --phase all --no-render")
        print()
        
        # Try to run the test script
        try:
            import subprocess
            print("🏁 Running test script...")
            result = subprocess.run([
                sys.executable, "test_curriculum_racing.py", 
                "--models-dir", "multi_agent_racing_results",
                "--phase", "racing",
                "--races-per-phase", "3"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"❌ Test failed: {result.stderr}")
                print("💡 Make sure you have trained models in 'multi_agent_racing_results/final_models/'")
                
        except FileNotFoundError:
            print("⚠️  Test script not found. Run training first!")
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            print("💡 Make sure you have trained models in 'multi_agent_racing_results/final_models/'")
    
    else:
        print("🚀 TRAINING MODE: Starting curriculum-based training...")
        print()
        print("🎓 Training will progress through 4 phases:")
        print("   1. SAFETY: Agents learn track boundaries (low speed)")
        print("   2. CONTROL: Agents improve steering precision (medium speed)")
        print("   3. SPEED: Agents learn efficient racing lines (high speed)")
        print("   4. RACING: Full competition with overtaking")
        print()
        print("⏱️  Estimated training time:")
        print(f"   - {args.timesteps:,} timesteps ≈ {args.timesteps // 50000} minutes")
        print("   - Automatic phase progression based on performance")
        print("   - Models saved every 50K steps + final models")
        print()
        
        # Import and run training
        try:
            from curriculum_multi_agent_racing import train_multi_agent_racing
            
            print("🏁 Starting training...")
            train_multi_agent_racing(
                num_agents=args.agents,
                total_timesteps=args.timesteps,
                results_dir='multi_agent_racing_results',
                use_wandb=args.wandb
            )
            
            print()
            print("✅ Training completed!")
            print("🎯 Next steps:")
            print("   1. Test your models: python demo_curriculum_racing.py --test")
            print("   2. View specific phases: python test_curriculum_racing.py --phase racing")
            print("   3. Run championship: python test_curriculum_racing.py --races-per-phase 10")
            print()
            
        except ImportError as e:
            print(f"❌ Training failed - missing dependencies: {e}")
            print("💡 Make sure you have:")
            print("   - stable-baselines3 installed")
            print("   - MetaDrive environment set up")
            print("   - Custom environment files in src/environments/")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Common issues:")
            print("   - Check that MetaDrive is properly installed")
            print("   - Ensure CUDA is available if using GPU")
            print("   - Verify workspace directory structure")


if __name__ == '__main__':
    main()