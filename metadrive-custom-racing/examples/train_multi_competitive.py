"""
Multi-agent competitive training launcher.

This script launches multiple single-agent training processes in parallel to create
competitive racing agents. Each agent trains independently but can later race against each other.
"""
import argparse
import os
import sys
import time
import subprocess
import threading
import shutil
from typing import List
import signal

# Global list to track running processes
running_processes: List[subprocess.Popen] = []

def signal_handler(sig, frame):
    """Handle Ctrl+C by terminating all child processes."""
    print("\n🛑 Stopping all training processes...")
    for proc in running_processes:
        if proc.poll() is None:  # Process is still running
            proc.terminate()
    
    # Wait a bit for graceful termination
    time.sleep(2)
    
    # Force kill if still running
    for proc in running_processes:
        if proc.poll() is None:
            proc.kill()
    
    print("✅ All processes stopped.")
    sys.exit(0)

def run_training_agent(agent_id: int, args: argparse.Namespace) -> int:
    """Run training for a single agent with personality-specific configuration."""
    # Create unique results directory for this agent
    agent_results_dir = f"results_agent_{agent_id}"
    os.makedirs(agent_results_dir, exist_ok=True)
    
    # Define selected hybrid personality mapping (4 chosen personalities)
    personalities = [
        "cautious_speedster",     # High speed, but careful driving
        "balanced_racer",         # Medium speed, balanced approach
        "aggressive_speedster",   # High speed, risky driving  
        "conservative_cruiser",   # Low speed, very safe driving
    ]
    personality = personalities[agent_id] if agent_id < len(personalities) else "balanced_racer"
    
    # Construct the training command with personality
    cmd = [
        sys.executable, 
        "examples/train_sb3.py",
        "--track", args.track,
        "--timesteps", str(args.timesteps),
        "--learning-rate", str(args.learning_rate),
        "--checkpoint-freq", str(args.checkpoint_freq),
        "--results-dir", agent_results_dir,  # Use agent-specific results dir
        "--personality", personality,  # Add personality parameter
    ]
    
    # Add optional flags
    if args.vecnorm:
        cmd.append("--vecnorm")
    
    # Always disable evaluation during competitive training to avoid MetaDrive issues
    # We'll evaluate using the tournament system instead
    cmd.append("--no-eval")
    
    # Add wandb if enabled
    if args.wandb:
        # Create descriptive names for each agent
        personalities = ["aggressive", "conservative", "speed_demon", "balanced"]
        personality = personalities[agent_id] if agent_id < len(personalities) else f"agent_{agent_id}"
        
        cmd.extend([
            "--wandb",
            "--wandb-project", args.wandb_project,
            "--wandb-name", f"{args.wandb_name}_{personality}",
        ])
    
    print(f"🚀 Starting Agent {agent_id} training...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        # Start the process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        running_processes.append(proc)
        
        # Stream output with agent prefix
        for line in proc.stdout:
            print(f"[Agent {agent_id}] {line.strip()}")
        
        # Wait for completion
        return_code = proc.wait()
        
        if return_code == 0:
            # Move the trained model to the main results directory with proper naming
            source_model = os.path.join(agent_results_dir, f"ppo_{args.track}.zip")
            source_stats = os.path.join(agent_results_dir, f"ppo_{args.track}_vecnormalize.pkl")
            
            # Create main results directory if it doesn't exist
            main_results_dir = "results"
            os.makedirs(main_results_dir, exist_ok=True)
            
            # Target filenames
            target_model = os.path.join(main_results_dir, f"competitive_agent_{agent_id}_{args.track}.zip")
            target_stats = os.path.join(main_results_dir, f"competitive_agent_{agent_id}_{args.track}_vecnormalize.pkl")
            
            # Move files if they exist
            if os.path.exists(source_model):
                try:
                    # Remove target file if it exists
                    if os.path.exists(target_model):
                        os.remove(target_model)
                    os.rename(source_model, target_model)
                    print(f"[Agent {agent_id}] ✅ Model saved to {target_model}")
                except OSError as e:
                    print(f"[Agent {agent_id}] ⚠️  Could not move model: {e}")
            
            if os.path.exists(source_stats):
                try:
                    # Remove target file if it exists
                    if os.path.exists(target_stats):
                        os.remove(target_stats)
                    os.rename(source_stats, target_stats)
                    print(f"[Agent {agent_id}] ✅ VecNormalize stats saved to {target_stats}")
                except OSError as e:
                    print(f"[Agent {agent_id}] ⚠️  Could not move VecNormalize stats: {e}")
            
            # Clean up agent-specific directory
            try:
                import shutil
                shutil.rmtree(agent_results_dir)
            except:
                pass  # Don't fail if cleanup fails
            
            print(f"✅ Agent {agent_id} training completed successfully!")
        else:
            print(f"❌ Agent {agent_id} training failed with code {return_code}")
        
        return return_code
        
    except Exception as e:
        print(f"❌ Agent {agent_id} failed to start: {e}")
        return 1

def main():
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='Train multiple competitive agents simultaneously')
    parser.add_argument('--track', type=str, default='right_oval', 
                       help='Track name from assets/track_configs/')
    parser.add_argument('--num-agents', type=int, default=4,
                       help='Number of competitive agents to train')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps per agent')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate for PPO')
    parser.add_argument('--vecnorm', action='store_true',
                       help='Use VecNormalize for observation normalization')
    parser.add_argument('--checkpoint-freq', type=int, default=25000,
                       help='Save checkpoint every N timesteps')
    
    # Weights & Biases arguments
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='metadrive-competitive',
                       help='W&B project name')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='W&B run name base (agent ID will be appended)')
    
    # Execution options
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Run agents in parallel (default: True)')
    parser.add_argument('--sequential', action='store_true',
                       help='Run agents sequentially instead of parallel')
    
    args = parser.parse_args()
    
    # Override parallel if sequential is specified
    if args.sequential:
        args.parallel = False
    
    # Set default wandb name if not provided
    if args.wandb and args.wandb_name is None:
        timestamp = int(time.time())
        args.wandb_name = f"racing_{args.track}_{timestamp}"
    
    print("🏁 Multi-Agent Competitive Training Launcher")
    print("=" * 60)
    print(f"Track: {args.track}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Timesteps per agent: {args.timesteps:,}")
    print(f"Execution mode: {'Parallel' if args.parallel else 'Sequential'}")
    print(f"Wandb enabled: {args.wandb}")
    if args.wandb:
        print(f"Wandb project: {args.wandb_project}")
        print(f"Base run name: {args.wandb_name}")
    print("=" * 60)
    
    # Confirm before starting
    if args.timesteps > 100000:
        response = input(f"\n⚠️  This will train {args.num_agents} agents for {args.timesteps:,} timesteps each. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    start_time = time.time()
    
    if args.parallel:
        print(f"\n🚀 Starting {args.num_agents} agents in parallel...")
        
        # Start all agents in parallel using threads
        threads = []
        results = [None] * args.num_agents
        
        def agent_wrapper(agent_id):
            results[agent_id] = run_training_agent(agent_id, args)
        
        # Launch all threads
        for i in range(args.num_agents):
            thread = threading.Thread(target=agent_wrapper, args=(i,))
            thread.start()
            threads.append(thread)
            time.sleep(2)  # Small delay between starts
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Check results
        successful_agents = sum(1 for result in results if result == 0)
        failed_agents = args.num_agents - successful_agents
        
    else:
        print(f"\n🚀 Starting {args.num_agents} agents sequentially...")
        
        successful_agents = 0
        failed_agents = 0
        
        # Run agents one by one
        for i in range(args.num_agents):
            print(f"\n--- Training Agent {i} ({i+1}/{args.num_agents}) ---")
            result = run_training_agent(i, args)
            if result == 0:
                successful_agents += 1
            else:
                failed_agents += 1
    
    # Final summary
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 60)
    print("🏁 COMPETITIVE TRAINING SUMMARY")
    print("=" * 60)
    print(f"✅ Successful agents: {successful_agents}")
    print(f"❌ Failed agents: {failed_agents}")
    print(f"⏱️  Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"📁 Models saved with prefix: competitive_agent_[0-{args.num_agents-1}]_{args.track}")
    
    if successful_agents > 0:
        print(f"\n🎮 To evaluate and race agents against each other:")
        print(f"   python examples/race_tournament.py --track {args.track}")
        print(f"   python examples/play_competitive_agents.py --track {args.track}")
        print(f"\n🏎️  To watch individual agents:")
        print(f"   python examples/play_model.py --model results_agent_0/ppo_{args.track}.zip")
        print(f"   python examples/play_model.py --model results_agent_1/ppo_{args.track}.zip")
        print(f"   # etc...")
    
    if args.wandb and successful_agents > 0:
        print(f"\n📊 View training results at:")
        print(f"   https://wandb.ai/{args.wandb_project}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()