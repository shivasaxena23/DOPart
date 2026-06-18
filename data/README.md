# Data Layout

Use this directory for stable experiment inputs.

Current profile data still lives in the legacy `DNNs/` directory and is loaded
through `dopart.profiles.system_values`. When you are ready to fully migrate,
move profile text files into `data/profiles/` and update `dopart.profiles`.

Current subdirectories:

- `raw/`: raw measurements used by legacy/default experiments.
- `raw/legacy/`: older ResNet/profile measurements kept for reproducibility.
- `profiles/`: intended future home for cleaned per-stage profile text files.
- `profiling_scripts/`: scripts used to generate or inspect profile data.
