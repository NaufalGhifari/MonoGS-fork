from pathlib import Path
from PIL import Image

rgb_dir = Path("./IRoS_Table_1/rgb")

for img_path in rgb_dir.glob("*.png"):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img.convert("RGB").save(img_path)

print("Converted all images in rgb/ to 3-channel RGB!")
