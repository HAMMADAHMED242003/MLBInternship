# Day-18: Video Processing with OpenCV

## Objective

The objective of this project is to learn how to process videos using OpenCV by reading videos frame by frame, applying image processing techniques, and saving the processed output.

## Topics Covered

- Reading video files
- Capturing frames
- FPS
- Video properties
- Saving processed videos
- Real-time webcam processing

## Processing Techniques Applied

1. Grayscale Conversion
2. Gaussian Blur
3. Canny Edge Detection

## What is FPS?

FPS (Frames Per Second) indicates how many frames are displayed or processed each second. A higher FPS results in smoother video playback.

## How OpenCV Reads Videos

OpenCV uses `cv2.VideoCapture()` to open a video file. Each call to `read()` retrieves one frame until the video ends.

## Results

Three different videos were processed successfully. Each processed video highlights object boundaries using edge detection while reducing noise with Gaussian Blur.

## Challenges Faced

- Kaggle does not support live webcam access.
- `cv2.imshow()` is not available in Kaggle notebooks, so Matplotlib was used to visualize sample frames.
- Choosing appropriate Canny thresholds required experimentation for different lighting conditions.