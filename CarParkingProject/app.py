import time

import cv2
import streamlit as st
import tempfile
import os
from ultralytics import YOLO


# Page Setup
# =========================

st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS for a cleaner, more "product" feel ----
st.markdown(
    """
    <style>
        .main > div {
            padding-top: 1.5rem;
        }
        #MainMenu, footer {visibility: hidden;}

        .app-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.25rem;
        }
        .app-header h1 {
            font-size: 2rem;
            margin: 0;
        }
        .app-subtitle {
            color: #9aa0a6;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }

        div[data-testid="stMetric"] {
            background: #1e1f24;
            border: 1px solid #2c2d33;
            border-radius: 12px;
            padding: 1rem 1rem 0.6rem 1rem;
        }
        div[data-testid="stMetric"] label {
            font-size: 0.8rem;
            color: #9aa0a6 !important;
        }

        .video-frame {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid #2c2d33;
        }

        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .status-idle { background: #2c2d33; color: #9aa0a6; }
        .status-running { background: rgba(46, 204, 113, 0.15); color: #2ecc71; }
        .status-done { background: rgba(52, 152, 219, 0.15); color: #3498db; }
    </style>
    """,
    unsafe_allow_html=True,
)



# Load Model
# =========================

MODEL_PATH = "best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()



# Header
# =========================

st.markdown(
    """
    <div class="app-header">
        <h1>🚗 Smart Parking AI</h1>
    </div>
    <div class="app-subtitle">Real-time parking occupancy detection powered by YOLO</div>
    """,
    unsafe_allow_html=True,
)

# Sidebar — Configuration
# =========================

with st.sidebar:
    st.header("⚙️ Configuration")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for a detection to be counted.",
    )

    st.divider()

    show_boxes = st.checkbox("Show bounding boxes", value=True)
    label_style = st.radio(
        "Slot labels",
        options=["Color only (no text)"],
        index=0,
        help="Use 'Color only' for dense lots where text would overlap between narrow slots.",
    )
    playback_speed = st.select_slider(
        "Playback speed",
        options=["0.5x", "1x", "1.5x", "2x", "Max"],
        value="1x",
    )

    st.divider()
    st.caption("Upload a parking-lot video below to begin analysis.")


# Upload
# =========================

uploaded_video = st.file_uploader(
    "Upload Parking Video",
    type=["mp4", "avi", "mov"],
    label_visibility="collapsed",
)

status_placeholder = st.empty()

if not uploaded_video:
    status_placeholder.markdown(
        '<span class="status-pill status-idle">● Waiting for video</span>',
        unsafe_allow_html=True,
    )
    st.info("👆 Upload a parking video (MP4, AVI, or MOV) to start detection.")



# Video Processing
# =========================

if uploaded_video:

    status_placeholder.markdown(
        '<span class="status-pill status-running">● Processing</span>',
        unsafe_allow_html=True,
    )

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(uploaded_video.read())
    video_path = temp.name

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    speed_map = {"0.5x": 2.0, "1x": 1.0, "1.5x": 0.66, "2x": 0.5, "Max": 0.0}
    frame_delay = (1 / fps) * speed_map[playback_speed] if fps else 0

    # --- Layout: video on the left, live stats on the right ---
    col_video, col_stats = st.columns([2.2, 1], gap="large")

    with col_video:
        st.markdown('<div class="video-frame">', unsafe_allow_html=True)
        frame_box = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
        progress_bar = st.progress(0)

    with col_stats:
        st.subheader("Live Dashboard")
        m1, m2 = st.columns(2)
        occupied_metric = m1.empty()
        free_metric = m2.empty()
        total_metric = st.empty()
        st.markdown("**Utilization**")
        utilization_bar = st.empty()

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # YOLO Prediction
        # =====================
        results = model(frame, conf=confidence, verbose=False)

        free_count = 0
        occupied_count = 0

 
        # Detection Drawing
        # =====================
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if class_name.lower() == "free":
                color = (0, 220, 0)
                label_full = "Free"
                label_short = "F"
                free_count += 1
            else:
                color = (0, 0, 230)
                label_full = "Occupied"
                label_short = "O"
                occupied_count += 1

            if show_boxes:
                box_w = max(x2 - x1, 1)
                box_h = max(y2 - y1, 1)

                # Thin border only — color already conveys status,
                # so we don't need heavy boxes.
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

                # Pick the largest label that still fits inside the box
                # width, falling back to a single-letter tag, and to no
                # text at all if the slot is too small even for that.
                candidates = [] if label_style.startswith("Color") else [
                    label_full,
                    label_short,
                ]
                chosen_label = None
                chosen_scale = None

                for candidate in candidates:
                    for scale in (0.4, 0.32, 0.25):
                        text_size, _ = cv2.getTextSize(
                            candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
                        )
                        if text_size[0] + 6 <= box_w and text_size[1] + 6 <= max(
                            box_h * 0.35, 10
                        ):
                            chosen_label, chosen_scale = candidate, scale
                            break
                    if chosen_label:
                        break

                if chosen_label:
                    text_size, _ = cv2.getTextSize(
                        chosen_label, cv2.FONT_HERSHEY_SIMPLEX, chosen_scale, 1
                    )
                    tw, th = text_size

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (min(x1 + tw + 6, x2), y1 + th + 6),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        chosen_label,
                        (x1 + 3, y1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        chosen_scale,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    # Too narrow for any text — just a small colored
                    # corner dot so the slot still reads at a glance.
                    dot_r = max(min(box_w, box_h) // 6, 2)
                    cv2.circle(
                        frame,
                        (x1 + dot_r + 2, y1 + dot_r + 2),
                        dot_r,
                        color,
                        -1,
                    )

        total_slots = free_count + occupied_count
        occupancy = (occupied_count / total_slots * 100) if total_slots else 0


        # Streamlit Display
        # =====================
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_box.image(frame_rgb, channels="RGB", use_container_width=True)

        occupied_metric.metric("🔴 Occupied", occupied_count)
        free_metric.metric("🟢 Available", free_count)
        total_metric.metric("🅿️ Total Slots", total_slots)
        utilization_bar.progress(
            min(int(occupancy), 100), text=f"{occupancy:.1f}% utilized"
        )

        progress_bar.progress(min(frame_index / total_frames, 1.0))

        if frame_delay > 0:
            time.sleep(frame_delay)

    cap.release()
    os.remove(video_path)

    status_placeholder.markdown(
        '<span class="status-pill status-done">● Completed</span>',
        unsafe_allow_html=True,
    )
    st.success("Processing completed successfully")