# Similar & Duplicate Image Finder

A tool that finds visually similar images using CNN feature embeddings, and
separately flags exact/near-duplicate images using perceptual hashing.

## Dataset

30 real photos from the [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset),
placed in `images/`:

- **12 photos of the same breed** (Siberian Husky, `n02110185_*`) — used as the
  near-duplicate / similarity test group
- **~12 photos of different breeds** — used as visually distinct negative examples
- **3 manually duplicated files** (`*_copy.jpg`) — exact byte-identical duplicates,
  to test the duplicate detector
- **3 modified versions of one query image** (`n02110185_120.jpg`), created for
  the mandatory challenge — see below

## Method

1. **CNN Embeddings** — every image is passed through **MobileNetV2**
   (ImageNet-1K pretrained weights, classification head removed) to get a
   1280-dimensional feature vector per image.
2. **Similarity search** — cosine similarity is computed between the query
   image's embedding and every other image's embedding; the **top 10** most
   similar images are reported (`TOP_K = 10`).
3. **Duplicate detection (separate method)** — `imagehash.phash` (perceptual
   hashing) is computed per image. Any pair with Hamming distance <= 10 is
   flagged as duplicate/near-duplicate.
   - `exact_duplicate = True` means pHash Hamming distance is exactly 0
     (visually indistinguishable to the hash), **not** necessarily a
     byte-identical file. In this dataset, both the manually copied files
     and the resized version of the query image land at distance 0 - the
     hash can't tell a true file copy apart from a resize, since resizing
     doesn't change the image's low-frequency structure. True file-level
     copies are `*_copy.jpg` files; the resized version is a different file
     that simply hashes identically.
4. **Outputs** - a results grid (PNG, query + top-10 matches), a CSV of the
   top-10 matches, a CSV of all duplicate pairs, the full pairwise
   similarity matrix, and a combined JSON report.

## Mandatory Challenge - Results

Query image: `n02110185_120.jpg`. Three modified versions were created:
resized (200x150), cropped (10% border removed), and brightened (+60%).

| Modification | CNN cosine similarity | pHash Hamming distance | Caught by CNN top-10? | Caught by pHash (dist <= 10)? |
|---|---|---|---|---|
| Resized | 0.946 | 0 | Yes | Yes |
| Brightened (+60%) | 0.960 | 10 | Yes | Yes |
| Cropped (10% border) | 0.929 | 20 | Yes | No |

**All three modified versions correctly rank in the real top-10 CNN matches**
for the original image - the mandatory challenge is satisfied.

The one exception is the crop: pHash misses it even at a relaxed threshold,
because cropping shifts pixel content relative to pHash's fixed grid - a
known structural limitation of perceptual hashing, not a bug in this
implementation. CNN embeddings, by contrast, catch the crop because they
encode *semantic content* rather than exact pixel layout. This is the core
justification for using both methods together rather than either alone.

## Additional Finding: same-breed is not the same as near-duplicate

The other Husky photos (different individual dogs, same breed) score
noticeably lower (~0.70 cosine similarity) than the transformed versions of
the query image (~0.93-0.96). This is expected: MobileNetV2 is a
general-purpose ImageNet-trained model, not a breed-specific classifier, so
it correctly distinguishes "the same photo, transformed" from "a different
dog of a visually similar breed." This confirms the tool detects true
near-duplicates rather than just loosely-related images.

## Files

- `similarity_finder.py` - main pipeline (run this)
- `make_modified_versions.py` - creates the 3 mandatory-challenge versions of
  a chosen base image
- `images/` - the 25-photo real dataset used for the final run
- `output/results_grid.png` - query image + top-10 CNN matches, one row
- `output/top5_similarity_report.csv` - top-10 cosine similarity matches for
  the query image (filename kept for consistency; contains 10 rows)
- `output/duplicate_pairs_report.csv` - every pair flagged by perceptual
  hashing (Hamming distance <= 10), with `exact_duplicate` marking distance-0
  pairs as described above
- `output/full_report.json` - everything combined in one JSON file
- `output/full_similarity_matrix.csv` - full pairwise cosine similarity
  matrix

## How to Run

```
pip install torch torchvision pytorchcv imagehash scikit-learn pandas matplotlib pillow
python3 similarity_finder.py
```

To test with a different query image or a different base image for the
challenge, edit `QUERY_IMAGE` in `similarity_finder.py` and `BASE_IMAGE` in
`make_modified_versions.py`.

## Tech Stack

- Python 3
- PyTorch / torchvision (image preprocessing)
- `pytorchcv` (MobileNetV2, ImageNet-1K pretrained weights)
- `imagehash` (perceptual hashing)
- `scikit-learn` (cosine similarity)
- `pandas` (CSV/JSON reports)
- `matplotlib` (results grid visualization)