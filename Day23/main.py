import cv2
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from difflib import SequenceMatcher
from ultralytics import YOLO
from paddleocr import PaddleOCR


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = "Input"
OUTPUT_FOLDER = "Output"
CROPS_FOLDER = os.path.join(OUTPUT_FOLDER, "crops")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CROPS_FOLDER, exist_ok=True)

# How large (relative to the frame) a vehicle's box must be to be
# considered "close". Bigger box = closer to camera. Tune this if
# your camera angle/distance is different (0.05-0.20 is typical).
MIN_CLOSE_AREA_RATIO = 0.08


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO models...")

vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("license_plate_detector.pt")

print("Loading PaddleOCR...")

plate_ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False
)

print("Models loaded successfully.")


# ============================================================
# VEHICLE CLASSES
# Only detecting cars now, per request (was car/motorcycle/bus/truck)
# ============================================================

VEHICLE_CLASSES = {
    2: "car",
}


# ============================================================
# VEHICLE DETECTION
# Now also filters out vehicles that are far away (small box relative
# to the frame). Only "close" cars are kept.
# ============================================================

def detect_vehicles(frame):

    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w

    results = vehicle_model(frame, conf=0.15, verbose=False)[0]

    boxes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, VEHICLE_CLASSES[cls_id], conf))

    def iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    boxes.sort(key=lambda b: b[5], reverse=True)

    filtered = []
    for b in boxes:
        if not any(iou(b, kept) > 0.5 for kept in filtered):
            filtered.append(b)

    # Keep only vehicles that are "close" -- i.e. their box takes up
    # a large enough fraction of the frame area. Distant cars (small
    # boxes) are dropped here.
    close_vehicles = []
    for (x1, y1, x2, y2, cls, conf) in filtered:
        box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        area_ratio = box_area / frame_area if frame_area > 0 else 0
        if area_ratio >= MIN_CLOSE_AREA_RATIO:
            close_vehicles.append((x1, y1, x2, y2, cls))

    return close_vehicles


# ============================================================
# LICENSE PLATE DETECTION
# Now returns BOTH the cropped plate image AND its box coordinates
# (relative to the vehicle crop it was found in), so the caller can
# convert to full-frame coordinates and draw a box on the plate itself.
# ============================================================

def detect_plate(vehicle_crop):

    results = plate_model(vehicle_crop, verbose=False)[0]

    if len(results.boxes) == 0:
        return None, None

    box = max(results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    h, w = vehicle_crop.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None, None

    plate_crop = vehicle_crop[y1:y2, x1:x2]

    return plate_crop, (x1, y1, x2, y2)


# ============================================================
# GLARE REMOVAL
# ============================================================

def remove_glare(plate_img):

    if plate_img is None or plate_img.size == 0:
        return plate_img

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    _, glare_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    glare_ratio = np.count_nonzero(glare_mask) / glare_mask.size

    if glare_ratio > 0.15:
        return plate_img

    glare_mask = cv2.dilate(glare_mask, np.ones((3, 3), np.uint8))
    result = cv2.inpaint(plate_img, glare_mask, 5, cv2.INPAINT_TELEA)

    return result


# ============================================================
# PLATE PREPROCESSING
# ============================================================

def preprocess_plate(plate_img):

    if plate_img is None or plate_img.size == 0:
        return []

    h, w = plate_img.shape[:2]

    # Denoise BEFORE upscaling. The old order (upscale first, denoise
    # never) blows up compression/sensor noise right along with the
    # text, which is what produced grainy, hard-to-read crops on
    # small/blurry plates. Denoising at native resolution is also
    # much cheaper and more effective.
    gray_small = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    denoised_small = cv2.fastNlMeansDenoising(
        gray_small, h=10, templateWindowSize=7, searchWindowSize=21
    )

    scale = 3
    if max(h, w) < 60:
        scale = 8
    elif max(h, w) < 120:
        scale = 5

    # Lanczos holds character-stroke edges together better than cubic
    # when upscaling small crops by a large factor.
    interp = cv2.INTER_LANCZOS4 if scale >= 5 else cv2.INTER_CUBIC
    gray = cv2.resize(denoised_small, None, fx=scale, fy=scale, interpolation=interp)

    # Unsharp mask to recover crisp edges that upscaling/denoising softened
    gaussian_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian_blur, -0.5, 0)

    # Lower clip limit than before -- on a denoised image, aggressive
    # CLAHE just re-amplifies whatever grain is left.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)

    v1 = cv2.bilateralFilter(enhanced, 9, 30, 30)

    v2 = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )

    blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)
    _, v3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, v4 = cv2.threshold(v1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [v1, v2, v3, v4]


# ============================================================
# OCR TEXT EXTRACTION
# Groups all lines found in a single OCR pass by their vertical
# position, sorts them top-to-bottom (and left-to-right within a
# similar row), and joins them into one combined string per source
# image, since real plates read top-to-bottom as ONE string.
# ============================================================

def _line_center_y(box):
    ys = [pt[1] for pt in box]
    return sum(ys) / len(ys)


def _line_center_x(box):
    xs = [pt[0] for pt in box]
    return sum(xs) / len(xs)


# ------------------------------------------------------------
# CONSENSUS VOTING ACROSS OCR VARIANTS
#
# Picking a single "highest confidence" candidate (old behaviour)
# rewards short, easy, but INCOMPLETE reads over longer correct
# ones, because average-confidence-per-line doesn't care about
# completeness. Instead: cluster all candidates from all 5 sources
# (raw + 4 processed variants) by text similarity, then within the
# strongest-supported cluster, do a confidence-weighted
# character-level vote using the most complete candidate as the
# alignment reference. This lets variants that only saw part of the
# plate still contribute correct characters, and rewards agreement
# across multiple independent OCR passes over a single lucky guess.
# No hardcoded plate length/format -- fully dynamic.
# ------------------------------------------------------------

def _similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _cluster_candidates(candidates, threshold=0.5):
    clusters = []
    for text, conf in candidates:
        best_cluster = None
        best_sim = 0.0
        for cluster in clusters:
            rep_text = cluster[0][0]
            sim = _similarity(text, rep_text)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster
        if best_cluster is not None and best_sim >= threshold:
            best_cluster.append((text, conf))
        else:
            clusters.append([(text, conf)])
    return clusters


def _character_vote(cluster):
    # Use the longest (most complete) candidate as the alignment
    # reference; ties broken by confidence.
    ref_text, _ = max(cluster, key=lambda t: (len(t[0]), t[1]))

    votes = [dict() for _ in range(len(ref_text))]

    for text, conf in cluster:
        matcher = SequenceMatcher(None, ref_text, text)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for offset in range(i2 - i1):
                    ch = ref_text[i1 + offset]
                    votes[i1 + offset][ch] = votes[i1 + offset].get(ch, 0.0) + conf
            elif tag == "replace":
                len_i = i2 - i1
                len_j = j2 - j1
                for offset in range(len_i):
                    src_offset = min(offset, len_j - 1) if len_j > 0 else None
                    if src_offset is not None:
                        ch = text[j1 + src_offset]
                        votes[i1 + offset][ch] = votes[i1 + offset].get(ch, 0.0) + conf
            # 'insert' (extra chars not in ref) and 'delete' (ref chars
            # this candidate is missing) are skipped -- they don't
            # vote on a ref position.

    final_chars = []
    for idx, ch_votes in enumerate(votes):
        if ch_votes:
            best_ch = max(ch_votes.items(), key=lambda kv: kv[1])[0]
            final_chars.append(best_ch)
        else:
            final_chars.append(ref_text[idx])

    return "".join(final_chars)


def _resolve_best_candidate(all_candidates):
    if not all_candidates:
        return "Unreadable", 0.0

    clusters = _cluster_candidates(all_candidates, threshold=0.5)

    # Score each cluster by total confidence-weighted support, with a
    # bonus for being independently seen by more sources -- agreement
    # across multiple OCR passes is stronger evidence than one
    # confident-but-lone read.
    def cluster_score(cluster):
        total_conf = sum(conf for _, conf in cluster)
        agreement_bonus = 1.0 + 0.15 * (len(cluster) - 1)
        return total_conf * agreement_bonus

    best_cluster = max(clusters, key=cluster_score)

    voted_text = _character_vote(best_cluster)
    avg_conf = sum(conf for _, conf in best_cluster) / len(best_cluster)

    return voted_text, avg_conf


def extract_text(processed_variants, raw_plate=None):

    all_candidates = []

    sources = []
    if raw_plate is not None:
        sources.append(("raw", raw_plate))
    for i, img in enumerate(processed_variants):
        sources.append((f"processed_{i + 1}", img))

    for source_name, img in sources:

        if img is None or img.size == 0:
            continue

        try:
            results = plate_ocr.predict(img)
        except Exception as e:
            print(f"    OCR failed on {source_name}: {e}")
            continue

        for result in results:

            try:
                rec_texts = result["rec_texts"]
                rec_scores = result["rec_scores"]
                rec_boxes = result.get("rec_polys") or result.get("dt_polys") or []
            except Exception:
                try:
                    rec_texts = result.get("rec_texts", [])
                    rec_scores = result.get("rec_scores", [])
                    rec_boxes = result.get("rec_polys") or result.get("dt_polys") or []
                except Exception as e:
                    print(f"    Could not parse OCR result: {e}")
                    continue

            if not rec_texts:
                continue

            # Pair up text/score/box; fall back to reading order if no boxes returned
            lines = []
            for idx, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                text_clean = "".join(c for c in str(text).upper() if c.isalnum())
                if not text_clean:
                    continue
                try:
                    score = float(score)
                except Exception:
                    continue
                box = rec_boxes[idx] if idx < len(rec_boxes) else None
                cy = _line_center_y(box) if box is not None else idx
                cx = _line_center_x(box) if box is not None else 0
                lines.append((text_clean, score, cy, cx))

            if not lines:
                continue

            # Sort top-to-bottom, then left-to-right (handles 2-line plates like LEF / 3503)
            lines.sort(key=lambda t: (t[2], t[3]))

            combined_text = "".join(t[0] for t in lines)
            combined_conf = sum(t[1] for t in lines) / len(lines)

            print(f"    [{source_name}] merged OCR: {combined_text} (conf: {combined_conf:.2f})")

            if len(combined_text) >= 3 and combined_conf >= 0.30:
                all_candidates.append((combined_text, combined_conf))

    best_text, best_conf = _resolve_best_candidate(all_candidates)

    if best_text != "Unreadable":
        print(f"    -> consensus result: {best_text} (avg conf: {best_conf:.2f})")

    return best_text, best_conf


# ============================================================
# OVERLAY FONT SIZING
# Font scale/thickness now derive from the actual frame size instead
# of a fixed 1.1 scale, so text stays readable without spilling off
# small images or looking tiny on large ones.
# Values are tuned around a 1280px-wide reference frame.
# ============================================================

def get_overlay_font_params(frame_w, frame_h):

    ref_dim = 1280.0
    scale_factor = min(frame_w, frame_h) / ref_dim if ref_dim > 0 else 1.0

    font_scale = 1.1 * scale_factor
    font_scale = max(0.4, min(font_scale, 1.6))

    thickness = max(1, round(3 * scale_factor))

    line_height = int(45 * scale_factor)
    line_height = max(20, line_height)

    return font_scale, thickness, line_height


# ============================================================
# MAIN PROCESSING
# ============================================================

if __name__ == "__main__":

    all_records = []

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CROPS_FOLDER, exist_ok=True)

    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: Input folder '{INPUT_FOLDER}' not found.")
        exit()

    for filename in os.listdir(INPUT_FOLDER):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(INPUT_FOLDER, filename)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Could not read {filename}")
            continue

        print(f"\nProcessing {filename}...")

        frame_h, frame_w = frame.shape[:2]
        font_scale, font_thickness, line_height = get_overlay_font_params(frame_w, frame_h)

        vehicles = detect_vehicles(frame)
        print(f"Found {len(vehicles)} close car(s) in {filename}")

        overlay_line = 0  # tracks stacked top-left label position per image

        for i, (x1, y1, x2, y2, vclass) in enumerate(vehicles):

            # Vehicle box: thin, unobtrusive (context only)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            vehicle_crop = frame[y1:y2, x1:x2]

            plate_crop, plate_box = detect_plate(vehicle_crop)

            crop_path = None

            if plate_crop is not None and plate_crop.size > 0:

                plate_h, plate_w = plate_crop.shape[:2]
                print(f"  Plate detected: {plate_w}x{plate_h}")

                # Convert plate box from vehicle-relative to full-frame coordinates
                px1, py1, px2, py2 = plate_box
                abs_px1, abs_py1 = x1 + px1, y1 + py1
                abs_px2, abs_py2 = x1 + px2, y1 + py2

                # Highlight ONLY the plate — thick, bright box
                cv2.rectangle(
                    frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 0, 255), 3
                )

                plate_crop = remove_glare(plate_crop)

                crop_path = os.path.join(CROPS_FOLDER, f"{filename}_{i}.jpg")
                cv2.imwrite(crop_path, plate_crop)

                processed_variants = preprocess_plate(plate_crop)

                if processed_variants:
                    processed_path = os.path.join(
                        CROPS_FOLDER, f"{filename}_{i}_processed.jpg"
                    )
                    cv2.imwrite(processed_path, processed_variants[0])

                text, conf = extract_text(processed_variants, raw_plate=plate_crop)

                label = f"{text} ({conf:.2f})" if text != "Unreadable" else "Unreadable"

            else:
                print("  No license plate detected")
                text, conf, crop_path, label = "Unreadable", 0.0, None, "Unreadable"

            # ============================================================
            # TOP-LEFT OVERLAY — font scaled to frame size, stacked per vehicle
            # Labeled "Vehicle 1", "Vehicle 2", etc. (no car/truck class name)
            # ============================================================
            overlay_text = f"Vehicle {i + 1}: {label}"
            text_y = int(frame_h * 0.05) + 20 + overlay_line * line_height
            overlay_line += 1

            # Background box behind text for readability
            (tw, th), _ = cv2.getTextSize(
                overlay_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Clamp so the label box never runs off the right/bottom edge
            box_x2 = min(10 + tw + 10, frame_w - 1)
            box_y2 = min(text_y + 10, frame_h - 1)

            cv2.rectangle(
                frame, (10, text_y - th - 10), (box_x2, box_y2),
                (0, 0, 0), -1
            )
            cv2.putText(
                frame, overlay_text, (6, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness
            )

            print(f"  {vclass}: {text} (conf: {conf:.2f})")

            all_records.append({
                "image": filename,
                "vehicle_class": vclass,
                "plate_crop_path": crop_path,
                "plate_text": text,
                "confidence": round(float(conf), 3),
                "timestamp": datetime.now().isoformat()
            })

        output_image_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_image_path, frame)

    df = pd.DataFrame(all_records)

    csv_path = os.path.join(OUTPUT_FOLDER, "results.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(OUTPUT_FOLDER, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)

    print("\nAll results saved to Output/results.csv and Output/results.json")
    print(df)