# MetaDrive Custom Racing

This project provides tools for training competitive racing agents using MetaDrive with custom tracks.

## Prerequisites

- Python 3.8+
- MetaDrive installed
- Stable-Baselines3
- Weights & Biases (optional)

## Quick Start

### 🎯 **ONE COMMAND FOR ALL AGENTS** (Main Feature!)

Train multiple competitive agents with a single command:

```powershell
# Train 4 competitive agents in parallel
python .\examples\train_multi_competitive.py --track right_oval --num-agents 4 --timesteps 500000 --vecnorm --wandb

# Quick test with 2 agents (10,000 timesteps each)
python .\examples\train_multi_competitive.py --track right_oval --num-agents 2 --timesteps 10000 --vecnorm --no-eval

# Train 6 agents sequentially (if you have limited resources)
python .\examples\train_multi_competitive.py --track right_oval --num-agents 6 --timesteps 300000 --sequential --vecnorm
```

**What this does:**
- ✅ Trains multiple agents in parallel (or sequential if specified)
- ✅ Each agent gets unique naming: `competitive_agent_0_right_oval.zip`, `competitive_agent_1_right_oval.zip`, etc.
- ✅ Automatic VecNormalize stats saving for each agent
- ✅ Individual Wandb tracking for each agent (if enabled)
- ✅ Graceful shutdown with Ctrl+C
- ✅ Proper file management and cleanup

### Racing Tournament

After training, see which agent performs best:

```powershell
# Race all competitive agents against each other
python .\examples\race_tournament.py --track right_oval --episodes 10

# Quick 3-episode tournament
python .\examples\race_tournament.py --track right_oval --episodes 3
```

### Watch Individual Agents

```powershell
# Watch a specific trained agent
python .\examples\play_model.py --model results\competitive_agent_0_right_oval.zip --track right_oval --episodes 1

# Watch another agent
python .\examples\play_model.py --model results\competitive_agent_1_right_oval.zip --track right_oval --episodes 1
```

### Single Agent Training (Alternative)

Train a single agent on the right oval track:

```powershell
python .\examples\train_sb3.py --track right_oval --timesteps 500000 --vecnorm --wandb
```

## Available Tracks

- `right_oval` - Custom right-turn-only oval track (recommended)
- `simple_oval` - Alternative oval configuration
- `custom_speedway` - Original demo track

## Training Options

### Basic Training

```powershell
python .\examples\train_multi_competitive.py --num-agents 4
```

### Advanced Training

```powershell
python .\examples\train_multi_competitive.py \
    --track right_oval \
    --num-agents 6 \
    --timesteps 1000000 \
    --learning-rate 3e-4 \
    --vecnorm \
    --checkpoint-freq 50000 \
    --wandb \
    --wandb-project "my-racing-project" \
    --wandb-tags "experiment1" "oval_track"
```

### Resource Management

```powershell
# Use sequential training to reduce memory usage
python .\examples\train_multi_competitive.py --num-agents 8 --sequential

# Disable evaluation to speed up training
python .\examples\train_multi_competitive.py --num-agents 4 --no-eval
```

## Key Features

- **One Command Training**: Single command trains multiple competitive agents
- **Parallel/Sequential**: Choose based on your system resources
- **Weights & Biases**: Full experiment tracking with individual agent runs
- **Tournament System**: Automated racing competitions between trained agents
- **Resume Training**: Continue from saved checkpoints
- **Custom Tracks**: Use right-turn-only ovals optimized for racing

## File Structure

```
examples/
├── train_multi_competitive.py  # 🎯 Main multi-agent trainer
├── race_tournament.py          # 🏁 Tournament between agents
├── train_sb3.py               # Single agent training
├── play_model.py              # Watch trained agents
└── use_custom_track.py        # Basic track demo

results/
└── competitive_agent_X_track.zip  # Trained agent models

assets/track_configs/
├── right_oval.json            # Recommended racing track
├── simple_oval.json           # Alternative oval
└── custom_speedway.json       # Original demo track
```

## Notes

- Agents train independently but can race against each other
- Models are saved with unique names for each agent
- VecNormalize stats are saved separately for proper evaluation
- Training can be stopped/resumed at any time
- Use Ctrl+C to gracefully stop all training processes
