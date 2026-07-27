# Smart Parking AI 

A Computer Vision based Smart Parking System that detects parking spaces and identifies whether they are occupied or available.

This project combines **OpenCV** and **YOLO** to analyze parking lot images/videos and display parking occupancy results.


## Project Overview

The goal of this project is to build a system that can automatically monitor a parking area and provide information about:

- Available parking spaces
- Occupied parking spaces
- Total parking occupancy

The project uses traditional computer vision techniques for image processing and YOLO for object detection.


## Dataset Used

**Roboflow Parking Lot Dataset**

The dataset contains parking lot images with annotations used for training the YOLO model.

The dataset was used for:
- Training the object detection model
- Testing parking occupancy detection



## Workflow

The overall pipeline of the project:

Input Image / Video
        |
        ↓
Image Preprocessing
        |
        ↓
Edge Detection
        |
        ↓
Morphological Operations
        |
        ↓
Contour Detection
        |
        ↓
Parking Slot Analysis
        |
        ↓
YOLO Vehicle Detection
        |
        ↓
Occupied / Vacant Decision
        |
        ↓
Visualization
```

---

## Technologies Used

- Python
- OpenCV
- YOLO (Ultralytics)
- Streamlit
- NumPy

---

## Features

- Detect vehicles in parking areas
- Identify occupied and empty parking spaces
- Display parking status using colors

Color indication:

Green → Available  
Red → Occupied

- Show parking statistics:
  - Total slots
  - Occupied slots
  - Available slots



## Installation

Clone the repository:

Install required libraries:

pip install -r requirements.txt

## Run Application

Start the Streamlit app:

python -m streamlit run app.py


Upload a parking video/image and the system will display the detection results.


## Results

The system successfully detects parking occupancy and provides visual feedback.

Example output:

```
Total Slots: 126
Occupied: 29
Available: 97
```

The output is displayed with:
- Green boxes for available spaces
- Red boxes for occupied spaces


## Challenges Faced

Some challenges during development:

- Detecting accurate parking slot boundaries
- Handling shadows and different lighting conditions
- Removing unwanted contours
- Making detection results easy to visualize

---

## Future Improvements

Possible improvements:

- Real-time CCTV camera integration
- Better parking slot detection using segmentation models
- Vehicle tracking
- Mobile application support
- Cloud-based parking monitoring

