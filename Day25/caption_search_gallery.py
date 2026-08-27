"""
Caption & Search Photo Gallery
==============================
Pipeline:
  1. Load every image in ./images
  2. Generate a caption per image with BLIP (blip-image-captioning-base)
  3. Compute a CLIP embedding per image (clip-vit-base-patch32)
  4. Accept natural-language text queries, embed them with CLIP, and
     return the top-5 most similar images (cosine similarity)
  5. Save a full report (captions + embeddings metadata + query results)
     to both CSV and JSON

CPU-friendly by design: base/patch32 checkpoints only, small image set.

Usage:
    python caption_search_gallery.py                 # run default demo queries
    python caption_search_gallery.py --query "a red car"
"""

import argparse
import base64
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)

IMAGES_DIR = "images"
OUTPUT_DIR = "output"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Queries used for the mandatory "abstract / indirect" challenge, mixed in
# with a couple of direct queries for comparison.
DEMO_QUERIES = [
    "a red car",                 # direct
    "someone cooking",           # abstract
    "something to eat",          # abstract
    "a peaceful scene",          # abstract
    "an animal in the wild",     # abstract
]


def load_images(images_dir):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    paths = sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith(exts)
    )
    if not paths:
        raise SystemExit(f"No images found in '{images_dir}'.")
    return paths


def generate_captions(image_paths, device):
    print(f"Loading BLIP ({BLIP_MODEL_NAME}) ...")
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(device)
    model.eval()

    captions = {}
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions[path] = caption
        print(f"  {os.path.basename(path):35s} -> {caption}")

    del model
    return captions


def _as_tensor(feats):
    """transformers >=5.0 wraps get_image_features/get_text_features output in
    BaseModelOutputWithPooling; older versions return a raw tensor directly.
    Handle both so this works across versions."""
    if hasattr(feats, "pooler_output"):
        return feats.pooler_output
    if hasattr(feats, "last_hidden_state") and not torch.is_tensor(feats):
        return feats.last_hidden_state
    return feats


def compute_clip_embeddings(image_paths, device):
    print(f"Loading CLIP ({CLIP_MODEL_NAME}) ...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()

    embeddings = {}
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = _as_tensor(model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings[path] = feats.squeeze(0).cpu().numpy()

    return model, processor, embeddings


def embed_text_query(query, model, processor, device):
    # CLIP was trained on "a photo of X"-style captions, so wrapping bare
    # queries in this template gives noticeably tighter, more accurate
    # similarity scores than embedding the raw phrase.
    prompted = f"a photo of {query}"
    inputs = processor(text=[prompted], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = _as_tensor(model.get_text_features(**inputs))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy()


def search(query, image_paths, image_embeddings, clip_model, clip_processor,
           captions, device, top_k=5):
    text_vec = embed_text_query(query, clip_model, clip_processor, device)
    sims = []
    for path in image_paths:
        img_vec = image_embeddings[path]
        score = float(np.dot(text_vec, img_vec))  # both L2-normalized -> cosine sim
        sims.append((path, score))
    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:top_k]
    results = [
        {
            "rank": i + 1,
            "image": os.path.basename(path),
            "similarity": round(score, 4),
            "caption": captions[path],
        }
        for i, (path, score) in enumerate(top)
    ]
    return results


def _image_to_data_uri(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def generate_html_report(all_query_results, path_by_name, out_path):
    """Self-contained HTML report: for every query, shows each of the top-5
    images (embedded inline as base64, so the file has no external
    dependencies), its similarity score, and its BLIP caption."""
    blocks = []
    for query, results in all_query_results.items():
        cards = []
        for r in results:
            img_path = path_by_name[r["image"]]
            data_uri = _image_to_data_uri(img_path)
            cards.append(f"""
            <div class="card">
              <img src="{data_uri}" alt="{r['image']}">
              <div class="meta">
                <div class="rank">#{r['rank']}</div>
                <div class="score">similarity: {r['similarity']:.4f}</div>
                <div class="fname">{r['image']}</div>
                <div class="caption">"{r['caption']}"</div>
              </div>
            </div>""")
        blocks.append(f"""
        <section>
          <h2>Query: &ldquo;{query}&rdquo;</h2>
          <div class="row">{''.join(cards)}</div>
        </section>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Caption &amp; Search Photo Gallery — Report</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; margin: 2rem; background: #fafafa; }}
  h1 {{ margin-bottom: .2rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 2px solid #333; padding-bottom: .3rem; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1rem; }}
  .card {{ width: 220px; background: #fff; border: 1px solid #ddd; border-radius: 8px;
           overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .card img {{ width: 100%; height: 160px; object-fit: cover; display: block; }}
  .meta {{ padding: .6rem .7rem; font-size: .85rem; }}
  .rank {{ font-weight: bold; color: #444; }}
  .score {{ color: #0a7d34; font-weight: 600; }}
  .fname {{ color: #888; font-size: .75rem; margin: .2rem 0; word-break: break-all; }}
  .caption {{ font-style: italic; color: #333; }}
</style></head>
<body>
<h1>Caption &amp; Search Photo Gallery — Report</h1>
<p>BLIP: image captioning &nbsp;|&nbsp; CLIP (ViT-B/32): text-to-image similarity search &nbsp;|&nbsp; top-5 per query</p>
{''.join(blocks)}
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append",
                         help="Natural language query. Repeatable. Defaults to a built-in demo set.")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    queries = args.query if args.query else DEMO_QUERIES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = load_images(IMAGES_DIR)
    print(f"Found {len(image_paths)} images.\n")

    t0 = time.time()
    captions = generate_captions(image_paths, device)
    print(f"\nCaptioning done in {time.time() - t0:.1f}s\n")

    t0 = time.time()
    clip_model, clip_processor, image_embeddings = compute_clip_embeddings(image_paths, device)
    print(f"CLIP embedding done in {time.time() - t0:.1f}s\n")

    # --- Per-image report (captions + embedding availability) ---
    per_image_rows = [
        {"image": os.path.basename(p), "caption": captions[p], "embedding_dim": image_embeddings[p].shape[0]}
        for p in image_paths
    ]
    pd.DataFrame(per_image_rows).to_csv(os.path.join(OUTPUT_DIR, "captions.csv"), index=False)

    # --- Run every query, collect results ---
    all_query_results = {}
    print("=" * 60)
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        results = search(query, image_paths, image_embeddings, clip_model,
                          clip_processor, captions, device, top_k=args.top_k)
        all_query_results[query] = results
        for r in results:
            print(f"  #{r['rank']} {r['image']:35s} sim={r['similarity']:.4f}  caption: {r['caption']}")

    # --- Save combined report: JSON (full) + CSV (flattened) ---
    with open(os.path.join(OUTPUT_DIR, "search_report.json"), "w") as f:
        json.dump(
            {
                "images_dir": IMAGES_DIR,
                "num_images": len(image_paths),
                "blip_model": BLIP_MODEL_NAME,
                "clip_model": CLIP_MODEL_NAME,
                "captions": {os.path.basename(k): v for k, v in captions.items()},
                "queries": all_query_results,
            },
            f,
            indent=2,
        )

    flat_rows = []
    for query, results in all_query_results.items():
        for r in results:
            flat_rows.append({"query": query, **r})
    pd.DataFrame(flat_rows).to_csv(os.path.join(OUTPUT_DIR, "search_report.csv"), index=False)

    # --- Visual HTML report: embeds each result's actual image thumbnail ---
    path_by_name = {os.path.basename(p): p for p in image_paths}
    generate_html_report(all_query_results, path_by_name,
                          os.path.join(OUTPUT_DIR, "search_report.html"))

    print("\nSaved: output/captions.csv, output/search_report.json, "
          "output/search_report.csv, output/search_report.html")


if __name__ == "__main__":
    main()