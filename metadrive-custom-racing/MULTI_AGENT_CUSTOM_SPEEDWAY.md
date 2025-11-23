# Multi-Agent Custom Speedway Racing

Run multiple agents (like Alek and Saegan) racing together on your complex custom speedway track!

## 🎯 Quick Start

### Option 1: Train Multiple Agents (Sequential)

Train agents one after another:

```powershell
# Train Alek and Saegan sequentially
python .\examples\train_multi_custom_speedway.py --num-agents 2 --agent-names Alek Saegan --timesteps 500000 --vecnorm

# Train 3 agents with default names
python .\examples\train_multi_custom_speedway.py --num-agents 3 --timesteps 300000 --vecnorm
```

### Option 2: Train Multiple Agents (Parallel - Faster!)

Train both agents at the same time (uses more RAM/CPU):

```powershell
# Train Alek and Saegan in parallel
python .\examples\train_multi_custom_speedway.py --num-agents 2 --agent-names Alek Saegan --timesteps 500000 --vecnorm --parallel

# Train 4 agents in parallel
python .\examples\train_multi_custom_speedway.py --num-agents 4 --timesteps 200000 --vecnorm --parallel
```

### Option 3: Watch Them Race Together!

After training, watch your agents compete on the same track:

```powershell
# Race Alek and Saegan together
python .\examples\race_custom_speedway.py --agents Alek Saegan --episodes 5

# Auto-detect and race all trained agents
python .\examples\race_custom_speedway.py --auto-detect --episodes 3

# Race specific model files
python .\examples\race_custom_speedway.py --model-paths results\results_agent_Alek\Alek_custom_speedway.zip results\results_agent_Saegan\Saegan_custom_speedway.zip
```

## 📁 File Structure

After training, your files will be organized like this:

```
results/
├── results_agent_Alek/
│   ├── Alek_custom_speedway.zip          # Alek's trained model
│   ├── vecnorm_Alek_custom_speedway.pkl  # Alek's normalization stats
│   ├── tensorboard/                      # Training logs
│   └── checkpoint_*.zip                  # Training checkpoints
│
└── results_agent_Saegan/
    ├── Saegan_custom_speedway.zip         # Saegan's trained model
    ├── vecnorm_Saegan_custom_speedway.pkl # Saegan's normalization stats
    ├── tensorboard/                       # Training logs
    └── checkpoint_*.zip                   # Training checkpoints
```

## 🏎️ Training Options

### Basic Training

```powershell
# Quick test (10k timesteps per agent)
python .\examples\train_multi_custom_speedway.py --num-agents 2 --agent-names Alek Saegan --timesteps 10000

# Full training (500k timesteps per agent)
python .\examples\train_multi_custom_speedway.py --num-agents 2 --agent-names Alek Saegan --timesteps 500000 --vecnorm
```

### Advanced Training

```powershell
python .\examples\train_multi_custom_speedway.py \
    --num-agents 2 \
    --agent-names Alek Saegan \
    --timesteps 1000000 \
    --learning-rate 3e-4 \
    --vecnorm \
    --checkpoint-freq 50000 \
    --parallel \
    --wandb \
    --wandb-project "custom-speedway-racing"
```

### Training Parameters

- `--num-agents` - Number of agents to train (default: 2)
- `--agent-names` - Custom names for agents (e.g., `--agent-names Alek Saegan`)
- `--timesteps` - Training steps per agent (default: 500,000)
- `--parallel` - Train agents simultaneously (faster but uses more resources)
- `--vecnorm` - Enable observation normalization (recommended)
- `--wandb` - Enable Weights & Biases experiment tracking
- `--learning-rate` - PPO learning rate (default: 3e-4)
- `--batch-size` - Batch size for training (default: 256)
- `--no-eval` - Disable periodic evaluation (faster training)

## 🏁 Racing Options

### Watch Multiple Agents Race

```powershell
# Race 2 specific agents
python .\examples\race_custom_speedway.py --agents Alek Saegan --episodes 5

# Race all available agents
python .\examples\race_custom_speedway.py --auto-detect

# Quick 1-episode race
python .\examples\race_custom_speedway.py --agents Alek Saegan --episodes 1
```

### Racing Parameters

- `--agents` - Names of agents to race (e.g., `--agents Alek Saegan`)
- `--model-paths` - Direct paths to model files
- `--auto-detect` - Automatically find and race all trained agents
- `--episodes` - Number of racing episodes (default: 3)
- `--no-render` - Run without visualization (faster)

## 🎮 Features

### Custom Speedway Track

The track includes:

- ✅ Hairpin turn (180° tight corner)
- ✅ S-section (alternating curves)
- ✅ Complex chicanes
- ✅ Long straights for overtaking
- ✅ Multiple left and right turns
- ✅ Challenging braking zones

### Multi-Agent Features

- ✅ Multiple agents race simultaneously
- ✅ Staggered starting positions to prevent collisions
- ✅ Individual agent statistics tracking
- ✅ Crash detection and penalty system
- ✅ Lane violation detection
- ✅ Real-time performance monitoring

## 📊 What Gets Tracked

During racing, the system tracks:

- Total reward per agent
- Number of crashes
- Steps completed
- Average speed
- Lane violations
- Position on track

## 🔧 Technical Details

### Environment Configuration

The multi-agent environment automatically configures:

- **Lane Setup**: 4 lanes, 40.0 width each
- **Starting Positions**: Staggered (-30 units per agent)
- **Lateral Offset**: Alternating left/right (+/-4 units)
- **Horizon**: 10,000 steps per episode
- **Crash Handling**: Penalty but episode continues

### Model Architecture

All agents use PPO with:

- Network: [256, 256, 128] neurons
- Learning rate: 3e-4
- Batch size: 256
- Steps per update: 4,096
- Entropy coefficient: 0.01
- Value function coefficient: 0.5

## 💡 Tips

### For Better Training

1. Use `--vecnorm` for stable learning
2. Train for at least 500k timesteps
3. Use `--parallel` if you have enough RAM (2+ agents)
4. Enable `--wandb` to track experiments
5. Save checkpoints frequently (`--checkpoint-freq 50000`)

### For Better Racing

1. Train multiple agents with different seeds
2. Test with `--episodes 5` for statistical significance
3. Use `--auto-detect` to race all agents at once
4. Check individual agent folders for detailed logs

## 🐛 Troubleshooting

### "Module not found" error

Make sure you're in the project root and run:

```powershell
$env:PYTHONPATH = "c:\Users\balde\OneDrive\Documents\SeniorProject\metadrive-custom-racing\src"
```

### Agents crash immediately

- Train for more timesteps (500k+)
- Enable VecNormalize (`--vecnorm`)
- Check that vecnorm stats are loading correctly

### Training is too slow

- Use `--no-eval` to skip periodic evaluation
- Reduce `--checkpoint-freq` to save less often
- Use sequential mode instead of parallel if RAM limited

### Multi-agent rendering issues

- Ensure MetaDrive is properly installed
- Try reducing number of agents
- Use `--no-render` for faster testing

## 📝 Examples

### Example 1: Train and Race Alek vs Saegan

```powershell
# Step 1: Train both agents (sequential)
python .\examples\train_multi_custom_speedway.py --num-agents 2 --agent-names Alek Saegan --timesteps 500000 --vecnorm

# Step 2: Watch them race
python .\examples\race_custom_speedway.py --agents Alek Saegan --episodes 5
```

### Example 2: Train 4 Agents in Parallel

```powershell
# Train 4 agents with custom names
python .\examples\train_multi_custom_speedway.py --num-agents 4 --agent-names Alpha Beta Gamma Delta --timesteps 300000 --vecnorm --parallel --wandb

# Race all 4
python .\examples\race_custom_speedway.py --auto-detect
```

### Example 3: Quick Test

```powershell
# Quick 10k timestep test
python .\examples\train_multi_custom_speedway.py --num-agents 2 --agent-names TestA TestB --timesteps 10000 --no-eval

# Quick race
python .\examples\race_custom_speedway.py --agents TestA TestB --episodes 1
```

## 🎉 Summary

You now have:

1. ✅ Multi-agent version of your custom speedway track
2. ✅ Training script for multiple agents (sequential or parallel)
3. ✅ Racing script to watch agents compete together
4. ✅ Individual agent result directories
5. ✅ Full statistics and tracking

Happy racing! 🏎️💨
