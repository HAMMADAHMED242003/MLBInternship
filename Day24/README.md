# Similar & Duplicate Image Finder

## What's here
- `similarity_finder.py` — main pipeline (run this)
- `generate_dataset.py` — builds the 24-image synthetic test set (6 themes x 4 images:
  1 original + 1 exact copy + 2 near-duplicates), since no real photo folder was provided
- `make_modified_versions.py` — mandatory-challenge script: creates a resized, a cropped,
  and a brightness-changed version of `beach_01.jpg`
- `images/` — the 27-image dataset used (24 generated + 3 challenge versions)
- `output/results_grid.png` — visual grid: query image, its top-5 CNN matches, and the
  challenge/duplicate examples
- `output/top5_similarity_report.csv` — top-5 cosine-similarity matches for the query image
- `output/duplicate_pairs_report.csv` — every pair flagged by perceptual hashing (pHash,
  Hamming distance ≤ 8), tagged exact vs. near-duplicate
- `output/full_report.json` — everything combined (top-5 matches, duplicate pairs, and the
  mandatory-challenge validation) in one JSON file
- `output/full_similarity_matrix.csv` — full pairwise cosine similarity matrix (27x27)

## How it works
1. **Embeddings**: MobileNetV2 (ImageNet-pretrained, via `pytorchcv` since PyTorch's own
   weight host is blocked in this sandbox — same ImageNet-1K MobileNetV2 weights) with the
   classification head removed → 1280-dim feature vector per image.
2. **Similarity search**: cosine similarity between the query image's embedding and every
   other image's embedding; top 5 reported.
3. **Duplicate detection**: `imagehash.phash` (perceptual hash) computed per image;
   any pair with Hamming distance ≤ 8 is flagged (distance 0 = exact duplicate).
4. **Mandatory challenge**: `beach_01.jpg` was resized, center-cropped, and brightened by
   60%. Results:
   - resized → cos_sim 0.954, pHash distance 0 → caught by **both** methods
   - cropped → cos_sim 0.936, pHash distance 18 → caught by CNN embeddings, **missed** by
     pHash (crops shift the hash grid) — shows why the two methods are complementary
   - brightened → cos_sim 0.863, pHash distance 6 → caught by **both** methods
   All three modified versions rank in the real top matches for the original image.

## Run it yourself
```
pip install torch torchvision pytorchcv imagehash scikit-learn pandas matplotlib pillow
python3 generate_dataset.py        # only needed if images/ folder is empty
python3 make_modified_versions.py  # only needed if the mod images are missing
python3 similarity_finder.py
```
To point it at your own photos, just replace the contents of `images/` with your own
20-30 JPG/PNG files and change `QUERY_IMAGE` at the top of `similarity_finder.py`.
