"""
Generates a synthetic dataset of ~28 'photos' since no real image folder
was provided. Images are procedurally drawn (gradients + shapes) grouped
into visual "themes" so that:
  - Several images are exact pixel duplicates (copy)
  - Several images are near-duplicates within the same theme (small jitter)
  - Several images are visually distinct (different themes)
This gives the duplicate/similarity finder something meaningful to detect.
"""
import os
import random
from PIL import Image, ImageDraw, ImageFilter

random.seed(42)

OUT_DIR = "/home/claude/images"
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 400, 300


def gradient_bg(c1, c2):
    img = Image.new("RGB", (W, H), c1)
    top = Image.new("RGB", (W, H), c2)
    mask = Image.new("L", (W, H))
    mask_data = []
    for y in range(H):
        mask_data.extend([int(255 * (y / H))] * W)
    mask.putdata(mask_data)
    img = Image.composite(top, img, mask)
    return img


def draw_scene(theme, variant_seed, jitter=0):
    """Draw a themed scene. `jitter` adds small random variation for near-dup groups."""
    rnd = random.Random(variant_seed)

    def j(v, spread=10):
        return v + rnd.randint(-spread, spread) if jitter else v

    if theme == "beach":
        img = gradient_bg((255, 224, 130), (100, 181, 246))
        d = ImageDraw.Draw(img)
        d.rectangle([0, j(200, 6), W, H], fill=(240, 220, 160))  # sand
        d.ellipse([j(300), j(30), j(340), j(70)], fill=(255, 235, 59))  # sun
        for i in range(3):
            d.polygon([(j(50 + i * 80), 200), (j(90 + i * 80), 150), (j(130 + i * 80), 200)],
                      fill=(3, 155, 229))
    elif theme == "mountain":
        img = gradient_bg((187, 222, 251), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.polygon([(j(0), H), (j(120), j(90)), (j(240), H)], fill=(120, 144, 156))
        d.polygon([(j(150), H), (j(280), j(60)), (j(400), H)], fill=(96, 125, 139))
        d.polygon([(j(260), H), (j(280), j(60)), (j(300), H)], fill=(255, 255, 255))
    elif theme == "forest":
        img = gradient_bg((200, 230, 201), (129, 199, 132))
        d = ImageDraw.Draw(img)
        for i in range(6):
            x = j(30 + i * 60, 8)
            d.rectangle([x, 180, x + 10, 260], fill=(93, 64, 55))
            d.ellipse([x - 25, j(110, 8), x + 35, 200], fill=(46, 125, 50))
    elif theme == "city":
        img = gradient_bg((255, 224, 178), (100, 100, 130))
        d = ImageDraw.Draw(img)
        for i in range(7):
            x = j(10 + i * 55, 6)
            h = j(90 + (i % 4) * 30, 10)
            d.rectangle([x, H - h, x + 40, H], fill=(55 + i * 10, 55, 80))
    elif theme == "abstract":
        img = Image.new("RGB", (W, H), (30, 30, 40))
        d = ImageDraw.Draw(img)
        for i in range(8):
            x0, y0 = j(rnd.randint(0, W)), j(rnd.randint(0, H))
            r = j(rnd.randint(20, 60), 8)
            color = (rnd.randint(100, 255), rnd.randint(50, 200), rnd.randint(100, 255))
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=color)
    elif theme == "desert":
        img = gradient_bg((255, 204, 128), (255, 236, 179))
        d = ImageDraw.Draw(img)
        for i in range(4):
            cx = j(60 + i * 90, 10)
            d.polygon([(cx - 50, H), (cx, j(120, 10)), (cx + 50, H)], fill=(224, 168, 96))
    else:
        img = Image.new("RGB", (W, H), (128, 128, 128))

    return img


# Theme plan: each theme gets a "base" image, some exact duplicates, some near-duplicates
themes = ["beach", "mountain", "forest", "city", "abstract", "desert"]

manifest = []
count = 0

for theme in themes:
    base = draw_scene(theme, variant_seed=hash(theme) % 1000, jitter=0)
    base_name = f"{theme}_01.jpg"
    base.save(os.path.join(OUT_DIR, base_name), quality=90)
    manifest.append((base_name, theme, "original"))
    count += 1

    # exact duplicate (simulate someone saving the same photo twice)
    dup_name = f"{theme}_01_copy.jpg"
    base.save(os.path.join(OUT_DIR, dup_name), quality=90)
    manifest.append((dup_name, theme, "exact_duplicate_of_" + base_name))
    count += 1

    # 2 near-duplicates (same theme, slightly different composition/lighting)
    for k in range(2):
        near = draw_scene(theme, variant_seed=hash(theme + str(k)) % 1000, jitter=1)
        # slight brightness/blur variation to mimic a "similar but not identical" photo
        if k == 1:
            near = near.filter(ImageFilter.GaussianBlur(0.5))
        near_name = f"{theme}_0{k+2}.jpg"
        near.save(os.path.join(OUT_DIR, near_name), quality=90)
        manifest.append((near_name, theme, "near_duplicate_of_" + base_name))
        count += 1

print(f"Generated {count} images across {len(themes)} themes in {OUT_DIR}")

with open(os.path.join(OUT_DIR, "_manifest.csv"), "w") as f:
    f.write("filename,theme,relationship\n")
    for row in manifest:
        f.write(",".join(row) + "\n")
