"""
Train and compare two YOLOv8n-seg models on the same custom dataset:
  Model 1: trained from scratch (random weights)
  Model 2: fine-tuned from COCO pretrained weights

Both use identical CPU-friendly settings.

Usage:
    python train_compare.py
"""

import time
import glob
import os
from ultralytics import YOLO

DATA_YAML = "dataset/data.yaml"
EPOCHS = 30          
IMGSZ = 320
BATCH = 4
DEVICE = "cpu"

UNSEEN_DIR = "dataset/unseen_test"


def train_model(weights_or_yaml, name):
    print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
    model = YOLO(weights_or_yaml)

    t0 = time.time()
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="runs_compare",
        name=name,
        exist_ok=True,
        verbose=False,
    )
    train_time = time.time() - t0

    print(f"\n{name} training time: {train_time:.1f} sec")

    # Validate
    metrics = model.val()
    map50_95 = metrics.seg.map      # mask mAP50-95
    map50 = metrics.seg.map50       # mask mAP50

    # Inference on unseen test images
    test_imgs = glob.glob(os.path.join(UNSEEN_DIR, "*.*"))
    test_imgs = [f for f in test_imgs if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    t0 = time.time()
    _ = model.predict(test_imgs, verbose=False)
    inf_time = (time.time() - t0) / max(len(test_imgs), 1)

    return {
        "name": name,
        "train_time_sec": train_time,
        "map50_95": map50_95,
        "map50": map50,
        "inference_time_per_img_sec": inf_time,
        "num_test_images": len(test_imgs),
    }


def main():
    results = []

    # Model 1: from scratch (architecture only, random weights)
    results.append(train_model("yolov8n-seg.yaml", "model1_scratch"))

    # Model 2: fine-tuned from COCO pretrained weights
    results.append(train_model("yolov8n-seg.pt", "model2_pretrained"))

    print(f"\n\n{'='*70}")
    print(f"{'Metric':<28}{'Model 1 (Scratch)':<22}{'Model 2 (Pretrained)':<22}")
    print(f"{'='*70}")
    r1, r2 = results
    print(f"{'Mask mAP50-95':<28}{r1['map50_95']:<22.4f}{r2['map50_95']:<22.4f}")
    print(f"{'Mask mAP50':<28}{r1['map50']:<22.4f}{r2['map50']:<22.4f}")
    print(f"{'Train time (sec)':<28}{r1['train_time_sec']:<22.1f}{r2['train_time_sec']:<22.1f}")
    print(f"{'Inference time/img (sec)':<28}{r1['inference_time_per_img_sec']:<22.4f}{r2['inference_time_per_img_sec']:<22.4f}")
    print(f"{'='*70}")

    winner = "Model 2 (Pretrained)" if r2["map50_95"] > r1["map50_95"] else "Model 1 (Scratch)"
    print(f"\nBetter mask mAP: {winner}")
    print("\nDone. Weights saved under runs_compare/model1_scratch and runs_compare/model2_pretrained")


if __name__ == "__main__":
    main()