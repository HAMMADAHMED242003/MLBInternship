"""
Similar & Duplicate Image Finder
=================================
1. Extracts CNN feature embeddings (MobileNetV2, ImageNet-pretrained) for every
   image in a folder.
2. Finds the top-5 most similar images to a chosen query image via cosine similarity.
3. Uses perceptual hashing (imagehash, pHash) to separately flag exact / near-duplicate
   images based on Hamming distance between hashes.
4. Saves a visual results grid (PNG) and a CSV + JSON report of everything found.
5. Validates the mandatory challenge: 3 modified versions (resized / cropped /
   brightness-changed) of one image are still correctly matched as near-duplicates
   by both methods.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model
from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

QUERY_IMAGE = "n02110185_120.jpg"    # image we search "similar to"
PHASH_DUP_THRESHOLD = 10              # Hamming distance <= this => flagged as duplicate/near-dup
TOP_K = 10

# ---------------------------------------------------------------------------
# 1. Load image file list
# ---------------------------------------------------------------------------
valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
image_files = sorted(
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith(valid_ext)
)
print(f"Found {len(image_files)} images in {IMG_DIR}")

# ---------------------------------------------------------------------------
# 2. Build MobileNetV2 feature extractor (drop the classification head)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MobileNetV2 (width multiplier 1.0), ImageNet-1K pretrained weights
mobilenet = ptcv_get_model("mobilenetv2_w1", pretrained=True)
mobilenet.output = nn.Identity()   # drop classification head -> 1280-d pooled feature vector
mobilenet.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def get_embedding(path):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = mobilenet(x)
    return feat.squeeze().cpu().numpy()


def get_phash(path):
    img = Image.open(path).convert("RGB")
    return imagehash.phash(img)


# ---------------------------------------------------------------------------
# 3. Extract embeddings + perceptual hashes for every image
# ---------------------------------------------------------------------------
embeddings = {}
phashes = {}

for fname in image_files:
    path = os.path.join(IMG_DIR, fname)
    embeddings[fname] = get_embedding(path)
    phashes[fname] = get_phash(path)

print("Extracted embeddings + perceptual hashes for all images.")

emb_matrix = np.stack([embeddings[f] for f in image_files])
sim_matrix = cosine_similarity(emb_matrix)
sim_df = pd.DataFrame(sim_matrix, index=image_files, columns=image_files)

# ---------------------------------------------------------------------------
# 4. Top-K most similar images to the query (CNN embedding, cosine similarity)
# ---------------------------------------------------------------------------
query_idx = image_files.index(QUERY_IMAGE)
sims_to_query = sim_df.loc[QUERY_IMAGE].drop(QUERY_IMAGE).sort_values(ascending=False)
top_matches = sims_to_query.head(TOP_K)

print(f"\nTop {TOP_K} images most similar to '{QUERY_IMAGE}' (CNN cosine similarity):")
for fname, score in top_matches.items():
    print(f"  {fname:35s} similarity={score:.4f}")

# ---------------------------------------------------------------------------
# 5. Perceptual-hash duplicate / near-duplicate detection (all pairs)
# ---------------------------------------------------------------------------
dup_pairs = []
files_list = image_files
for i in range(len(files_list)):
    for j in range(i + 1, len(files_list)):
        f1, f2 = files_list[i], files_list[j]
        dist = phashes[f1] - phashes[f2]  # Hamming distance
        if dist <= PHASH_DUP_THRESHOLD:
            dup_pairs.append({
                "image_1": f1,
                "image_2": f2,
                "phash_hamming_distance": int(dist),
                "cnn_cosine_similarity": float(sim_df.loc[f1, f2]),
                "exact_duplicate": bool(dist == 0),
            })

dup_pairs.sort(key=lambda r: r["phash_hamming_distance"])
print(f"\nDetected {len(dup_pairs)} duplicate/near-duplicate pairs via perceptual hashing "
      f"(Hamming distance <= {PHASH_DUP_THRESHOLD}):")
for p in dup_pairs:
    tag = "EXACT DUP" if p["exact_duplicate"] else "near-dup"
    print(f"  [{tag:9s}] {p['image_1']:32s} <-> {p['image_2']:32s} "
          f"dist={p['phash_hamming_distance']:2d}  cos_sim={p['cnn_cosine_similarity']:.4f}")

# ---------------------------------------------------------------------------
# 6. Mandatory challenge check: the 3 modified versions of beach_01.jpg
# ---------------------------------------------------------------------------
modified_versions = ["n02110185_120_mod_resized.jpg", "n02110185_120_mod_cropped.jpg", "n02110185_120_mod_bright.jpg"]
challenge_report = []
for mod in modified_versions:
    if mod not in image_files:
        continue
    cos_sim = float(sim_df.loc[QUERY_IMAGE, mod])
    dist = phashes[QUERY_IMAGE] - phashes[mod]
    rank_in_topk = mod in top_matches.index
    challenge_report.append({
        "modified_image": mod,
        "cnn_cosine_similarity_to_original": round(cos_sim, 4),
        "phash_hamming_distance_to_original": int(dist),
        "flagged_as_near_duplicate_by_phash": bool(dist <= PHASH_DUP_THRESHOLD),
        "appears_in_top5_cnn_matches": bool(rank_in_topk),
    })

print(f"\nMandatory challenge — modified versions of '{QUERY_IMAGE}':")
for r in challenge_report:
    print(f"  {r['modified_image']:30s} cos_sim={r['cnn_cosine_similarity_to_original']:.4f} "
          f"phash_dist={r['phash_hamming_distance_to_original']:2d} "
          f"near_dup={r['flagged_as_near_duplicate_by_phash']} "
          f"in_top5={r['appears_in_top5_cnn_matches']}")

# ---------------------------------------------------------------------------
# 7. Results grid image: query + top-5 CNN matches
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 11, figsize=(30, 4))

axes[0].imshow(Image.open(os.path.join(IMG_DIR, QUERY_IMAGE)))
axes[0].set_title(f"QUERY\n{QUERY_IMAGE}", fontsize=9, color="darkred")
axes[0].axis("off")

for i, (fname, score) in enumerate(top_matches.items(), start=1):
    axes[i].imshow(Image.open(os.path.join(IMG_DIR, fname)))
    axes[i].set_title(f"#{i}: {fname}\nsim={score:.3f}", fontsize=8)
    axes[i].axis("off")

plt.suptitle("Similar & Duplicate Image Finder — Top 10 Matches", fontsize=14, fontweight="bold")
plt.tight_layout()
grid_path = os.path.join(OUT_DIR, "results_grid.png")
plt.savefig(grid_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"\nSaved results grid to {grid_path}")

# ---------------------------------------------------------------------------
# 8. Save CSV + JSON reports
# ---------------------------------------------------------------------------
# CSV: top-K similarity results
topk_csv = pd.DataFrame({
    "rank": range(1, len(top_matches) + 1),
    "query_image": QUERY_IMAGE,
    "matched_image": top_matches.index,
    "cosine_similarity": top_matches.values,
})
topk_csv_path = os.path.join(OUT_DIR, "top5_similarity_report.csv")
topk_csv.to_csv(topk_csv_path, index=False)

# CSV: duplicate pairs
dup_csv_path = os.path.join(OUT_DIR, "duplicate_pairs_report.csv")
pd.DataFrame(dup_pairs).to_csv(dup_csv_path, index=False)

# Full JSON report (everything)
json_report = {
    "num_images_scanned": len(image_files),
    "image_folder": IMG_DIR,
    "query_image": QUERY_IMAGE,
    "phash_duplicate_threshold_hamming": PHASH_DUP_THRESHOLD,
    "top5_cnn_similarity_matches": [
        {"rank": i + 1, "image": fname, "cosine_similarity": round(float(score), 4)}
        for i, (fname, score) in enumerate(top_matches.items())
    ],
    "duplicate_and_near_duplicate_pairs_phash": dup_pairs,
    "mandatory_challenge_modified_versions": challenge_report,
}
json_path = os.path.join(OUT_DIR, "full_report.json")
with open(json_path, "w") as f:
    json.dump(json_report, f, indent=2)

# also save full pairwise similarity matrix for reference
sim_matrix_path = os.path.join(OUT_DIR, "full_similarity_matrix.csv")
sim_df.round(4).to_csv(sim_matrix_path)

print(f"Saved reports:\n  {topk_csv_path}\n  {dup_csv_path}\n  {json_path}\n  {sim_matrix_path}")
