# MetaDrive Custom Racing (minimal)

This small project shows how to run MetaDrive with a custom track and a single car (no traffic).

Prerequisites:

- Python 3.8+
- MetaDrive installed (you already said it's installed)

Quick start:

1. Run the demo:

```powershell
python .\examples\use_custom_track.py
```

Files created:

- `assets/track_configs/custom_speedway.json` — custom track definition
- `src/utils/track_loader.py` — loader for track configs
- `examples/use_custom_track.py` — example that runs a single car on the custom track

Notes:

- This project doesn't modify the original `metadrive/` folder.
- To create your own track, add another JSON file to `assets/track_configs/` and call it by filename (without .json) from the example script.
