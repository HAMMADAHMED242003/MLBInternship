# Caption & Search Photo Gallery

BLIP captions every image, CLIP finds the top-5 best matches for a text query.

**Live report:** https://htmlpreview.github.io/?https://raw.githubusercontent.com/HAMMADAHMED242003/MLBInternship/refs/heads/main/Day25/output/search_report.html
*(GitHub can't render raw HTML, so this link uses htmlpreview.github.io. You can also just open `output/search_report.html` locally.)*

## Models
- `blip-image-captioning-base` — captions
- `clip-vit-base-patch32` — search

## Run
```bash
pip install -r requirements.txt
python caption_search_gallery.py
```

Outputs in `output/`: `captions.csv`, `search_report.csv/json`, `search_report.html`

## Results

**Direct query** — "a red car": worked perfectly, all top-5 matches were red cars.

**Abstract queries:**

| Query | Result |
|---|---|
| someone cooking |  Correct — matched actual cooking photos |
| something to eat |  Correct — matched food photos |
| a peaceful scene |  Correct — matched calm nature photos |
| an animal in the wild |  Correct — matched real wildlife photos |

