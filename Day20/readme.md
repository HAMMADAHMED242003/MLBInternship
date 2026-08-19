# Bottle Instance Segmentation — Scratch vs Pretrained YOLOv8n-seg

## Task
Build a small custom instance segmentation dataset (~50 images), annotate manually with polygons, then train two YOLOv8n-seg models — one from scratch, one from pretrained COCO weights — and compare them.

## Dataset
- 50 images, single class: **bottle**
- Own raw images, no pre-existing labels
- Manually annotated polygon masks using **makesense.ai**
- Exported as COCO JSON → converted to YOLO segmentation format
- Split: 35 train / 8 val / 7 unseen test

## Training setup
Both models trained with identical CPU-friendly settings:
- Model: YOLOv8n-seg
- Image size: 320
- Batch size: 4
- Epochs: 30
- Device: CPU

**Model 1** — trained from scratch (random weights)
**Model 2** — fine-tuned from COCO pretrained weights

## Results

| Metric | Model 1 (Scratch) | Model 2 (Pretrained) |
|---|---|---|
| Mask mAP50-95 | 0.0887 | 0.9736 |
| Mask mAP50 | 0.1309 | 0.9950 |
| Train time (sec) | 423.0 | 293.7 |
| Inference time/img (sec) | 0.1283 | 0.0680 |

## Conclusion
Pretrained model wins on everything — accuracy, training time, and inference speed.

Makes sense: with only 35 training images, a model starting from random weights doesn't have enough data to learn basic shapes/edges on its own. The pretrained model already knows this stuff from COCO, so it just has to adapt to bottles — way less to learn, so it converges faster and generalizes better.

Basically: if your dataset is small, don't train from scratch — always start from pretrained weights.

## Files
- `build_dataset.py` — auto-splits images + labels into train/val/unseen_test and builds the dataset folder
- `train_compare.py` — trains both models and prints the comparison table
- `dataset/` — final YOLO-format dataset (images, labels, data.yaml)