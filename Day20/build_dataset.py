"""
Auto-builds the YOLO segmentation dataset folder structure from:
  - a folder of all your images
  - a folder of YOLO label .txt files (unzipped from yolo_labels.zip)

Splits into: 35 train / 8 val / 7 unseen_test (no labels copied for unseen_test)

Usage:
    python build_dataset.py <images_folder> <labels_folder> <output_dataset_folder> [class_name]

Example:
    python build_dataset.py ./all_images ./yolo_labels ./dataset bottle
"""

import os
import sys
import shutil
import random

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build(images_dir, labels_dir, out_dir, class_name="bottle",
          n_train=35, n_val=8, n_test=7, seed=42):

    # Collect images that actually have a matching label file
    all_images = [f for f in os.listdir(images_dir)
                  if os.path.splitext(f)[1].lower() in IMG_EXTS]

    matched = []
    missing_labels = []
    for img in all_images:
        base = os.path.splitext(img)[0]
        label_path = os.path.join(labels_dir, base + ".txt")
        if os.path.exists(label_path):
            matched.append(img)
        else:
            missing_labels.append(img)

    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images have no matching label file, skipping them:")
        for m in missing_labels[:10]:
            print("   -", m)
        if len(missing_labels) > 10:
            print(f"   ... and {len(missing_labels)-10} more")

    total_needed = n_train + n_val + n_test
    if len(matched) < total_needed:
        print(f"\nERROR: only {len(matched)} labeled images found, need {total_needed}.")
        print("Fix your images/labels folders and try again.")
        sys.exit(1)

    random.seed(seed)
    random.shuffle(matched)

    train_imgs = matched[:n_train]
    val_imgs = matched[n_train:n_train + n_val]
    test_imgs = matched[n_train + n_val:n_train + n_val + n_test]

    # Build folders
    paths = {
        "images/train": os.path.join(out_dir, "images", "train"),
        "images/val": os.path.join(out_dir, "images", "val"),
        "labels/train": os.path.join(out_dir, "labels", "train"),
        "labels/val": os.path.join(out_dir, "labels", "val"),
        "unseen_test": os.path.join(out_dir, "unseen_test"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    def copy_pair(img_name, img_split, label_split=None):
        src_img = os.path.join(images_dir, img_name)
        shutil.copy(src_img, os.path.join(paths[img_split], img_name))
        if label_split:
            base = os.path.splitext(img_name)[0]
            src_label = os.path.join(labels_dir, base + ".txt")
            shutil.copy(src_label, os.path.join(paths[label_split], base + ".txt"))

    for img in train_imgs:
        copy_pair(img, "images/train", "labels/train")
    for img in val_imgs:
        copy_pair(img, "images/val", "labels/val")
    for img in test_imgs:
        # unseen test: only image, no label copied
        shutil.copy(os.path.join(images_dir, img), os.path.join(paths["unseen_test"], img))

    # write data.yaml
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: images/train\nval: images/val\nnc: 1\nnames: ['{class_name}']\n")

    print(f"\nDone!")
    print(f"  Train: {len(train_imgs)} images -> {paths['images/train']}")
    print(f"  Val:   {len(val_imgs)} images -> {paths['images/val']}")
    print(f"  Unseen test: {len(test_imgs)} images -> {paths['unseen_test']} (no labels)")
    print(f"  data.yaml written to: {yaml_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python build_dataset.py <images_folder> <labels_folder> <output_dataset_folder> [class_name]")
        sys.exit(1)
    images_dir = sys.argv[1]
    labels_dir = sys.argv[2]
    out_dir = sys.argv[3]
    class_name = sys.argv[4] if len(sys.argv) > 4 else "bottle"
    build(images_dir, labels_dir, out_dir, class_name)