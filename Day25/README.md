  [View live report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/HAMMADAHMED242003/MLBInternship/refs/heads/main/Day25/output/search_report.html)
# Caption & Search Photo Gallery

A CPU-friendly pipeline: BLIP captions every image, CLIP embeds every image,
and you can search the gallery with a natural-language query.

## What's included
- `images/` — sample images (pulled from a public ImageNet sample-image
  set), covering vehicles, food, nature/scenery, and animals so both direct
  and abstract queries have something reasonable to match against.
- `caption_search_gallery.py` — the full pipeline (captioning, embeddings,
  search, CSV/JSON report).
- `requirements.txt` — CPU-only dependencies.

Models used (small/base variants, as requested, for fast CPU inference):
- `Salesforce/blip-image-captioning-base`
- `openai/clip-vit-base-patch32`

## ⚠️ One thing to know before you run it
This project was assembled in a sandboxed environment whose network access
is restricted to package registries (PyPI, GitHub, npm, etc.) — it does
**not** have access to `huggingface.co`, which is where the actual BLIP and
CLIP model *weights* are hosted. So I could build, sanity-check, and hand
you the complete, correct pipeline and a real 23-image dataset, but I
couldn't execute the captioning/embedding/search steps myself to hand you
filled-in results — the model downloads fail in this sandbox.

The code is complete and standard (straight `transformers` usage), so it
will run as-is anywhere with normal internet access.

## How to run it
```bash
pip install -r requirements.txt
python caption_search_gallery.py
```
First run downloads the two model checkpoints (~1GB combined) from
Hugging Face — after that they're cached locally. On a typical CPU this
takes a few minutes for 23 images.

This runs 5 built-in demo queries (see below) and writes:
- `output/captions.csv` — every image + its BLIP caption
- `output/search_report.json` — full report (captions + all query results)
- `output/search_report.csv` — flattened table: query, rank, image, similarity, caption

To run your own query:
```bash
python caption_search_gallery.py --query "a red car" --query "people relaxing outdoors"
```

## The mandatory abstract-query challenge
Built-in queries (1 direct, 4 abstract/indirect) — chosen because the image
set (sports cars, pizza/cheeseburger/banana, lakeside/seashore/alp/volcano,
golden retriever/tabby cat/lion/zebra/elephant) gives each one a plausible
target without naming it directly:

| Query | Type | Intended match |
|---|---|---|
| "a red car" | direct | sports car / convertible / fire engine |
| "someone cooking" | abstract | frying pan / wok / espresso maker |
| "something to eat" | abstract | pizza / cheeseburger / banana / ice cream / guacamole |
| "a peaceful scene" | abstract | lakeside / seashore / alp |
| "an animal in the wild" | abstract | lion / zebra / elephant |

