import os
from pathlib import Path

DATASET_DIR = Path("./IRoS_Table_1")

# Read timestamps from rgb.txt
rgb_txt = DATASET_DIR / "rgb.txt"
gt_lines = ["# ground truth trajectory\n", "# file: 'custom_kinect'\n", "# timestamp tx ty tz qx qy qz qw\n"]

with open(rgb_txt, "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            ts = line.split()[0]
            # Dummy identity pose: tx=0, ty=0, tz=0, qx=0, qy=0, qz=0, qw=1
            gt_lines.append(f"{ts} 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")

with open(DATASET_DIR / "groundtruth.txt", "w") as f:
    f.writelines(gt_lines)

print("Created dummy groundtruth.txt successfully!")
