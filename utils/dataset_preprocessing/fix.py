import os
from pathlib import Path

# Path to your custom dataset directory
DATASET_DIR = Path("./IRoS_Table_1")

# Target FPS for the sequence (Kinect v1 default is 30 FPS)
FPS = 30.0
DT = 1.0 / FPS

assoc_path = DATASET_DIR / "association.txt"

if not assoc_path.exists():
    raise FileNotFoundError(f"Could not find {assoc_path}")

# Read and parse existing association.txt entries
entries = []
with open(assoc_path, "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 4:
                # Format: rgb_ts, rgb_rel_path, depth_ts, depth_rel_path
                entries.append((parts[0], parts[1], parts[2], parts[3]))

print(f"Found {len(entries)} paired frames in association.txt.")

new_assoc_lines = []
new_rgb_lines = []
new_depth_lines = []

# Process each frame sequentially
for idx, (old_rgb_ts, old_rgb_rel, old_depth_ts, old_depth_rel) in enumerate(entries):
    # Generate new clean timestamp (4 decimal places: 0.0000, 0.0333, 0.0667...)
    new_ts_str = f"{idx * DT:.4f}"
    
    # Define old file paths
    old_rgb_file = DATASET_DIR / old_rgb_rel
    old_depth_file = DATASET_DIR / old_depth_rel
    
    # Define new file paths
    new_rgb_rel = f"rgb/{new_ts_str}.png"
    new_depth_rel = f"depth/{new_ts_str}.png"
    new_rgb_file = DATASET_DIR / new_rgb_rel
    new_depth_file = DATASET_DIR / new_depth_rel
    
    # Rename RGB file
    if old_rgb_file.exists():
        old_rgb_file.rename(new_rgb_file)
    else:
        print(f"Warning: Missing RGB file {old_rgb_file}")

    # Rename Depth file
    if old_depth_file.exists():
        old_depth_file.rename(new_depth_file)
    else:
        print(f"Warning: Missing Depth file {old_depth_file}")

    # Build new metadata lines
    new_assoc_lines.append(f"{new_ts_str} {new_rgb_rel} {new_ts_str} {new_depth_rel}\n")
    new_rgb_lines.append(f"{new_ts_str} {new_rgb_rel}\n")
    new_depth_lines.append(f"{new_ts_str} {new_depth_rel}\n")

# Overwrite metadata files with clean timestamps
with open(DATASET_DIR / "association.txt", "w") as f:
    f.writelines(new_assoc_lines)

with open(DATASET_DIR / "rgb.txt", "w") as f:
    f.writelines(new_rgb_lines)

with open(DATASET_DIR / "depth.txt", "w") as f:
    f.writelines(new_depth_lines)

print("Dataset successfully sanitized and renamed!")
