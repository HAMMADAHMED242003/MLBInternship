from PIL import Image, ImageEnhance
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
BASE_IMAGE = "n02110185_120.jpg"

base_path = os.path.join(IMG_DIR, BASE_IMAGE)
img = Image.open(base_path).convert("RGB")

name, ext = os.path.splitext(BASE_IMAGE)

# 1. Resized version (downscaled then kept at new size - different dimensions)
resized = img.resize((200, 150))
resized.save(os.path.join(IMG_DIR, f"{name}_mod_resized{ext}"), quality=90)

# 2. Cropped version (crop out a border, i.e. a tighter crop of the same scene)
w, h = img.size
crop_box = (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))
cropped = img.crop(crop_box)
cropped.save(os.path.join(IMG_DIR, f"{name}_mod_cropped{ext}"), quality=90)

# 3. Brightness-changed version
enhancer = ImageEnhance.Brightness(img)
brighter = enhancer.enhance(1.6)  # 60% brighter
brighter.save(os.path.join(IMG_DIR, f"{name}_mod_bright{ext}"), quality=90)

print("Created 3 modified versions of", BASE_IMAGE)
for f in [f"{name}_mod_resized{ext}", f"{name}_mod_cropped{ext}", f"{name}_mod_bright{ext}"]:
    print(" -", f, Image.open(os.path.join(IMG_DIR, f)).size)