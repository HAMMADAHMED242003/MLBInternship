# Shape Detection using OpenCV

## Overview

In this task, I created a simple shape detection program using OpenCV. The program reads an image, converts it to grayscale, applies thresholding, detects contours, and identifies different geometric shapes. I tested the program on 10 different images and saved the original image, contour detection result, and final shape detection result with labels.

## What are Contours?

Contours are the outlines or boundaries of objects in an image. They help OpenCV identify where an object starts and ends, making it easier to measure and recognize different shapes.

## How Contour Detection Works

The program first converts the image to grayscale and applies thresholding to separate the objects from the background. After that, OpenCV finds the contours of each object. Using these contours, the program calculates the area, perimeter, and number of corners to identify whether the object is a triangle, square, rectangle, or circle.

## Shapes Detected

The program can detect:

* Triangle
* Square
* Rectangle
* Circle

For every test image, the program saves:

* Original Image
* Contour Detection Result
* Final Shape Detection Result with Shape Labels

## Challenges Faced

The biggest challenge was detecting light-colored shapes. Since contour detection depends on thresholding, some light-colored shapes were not detected properly, especially when the background was also light. I experimented with different threshold values and contour settings to improve the detection, but lighting and color differences still affected the results in some cases.

## Conclusion

This task helped me understand how contour detection works in OpenCV and how it can be used for shape recognition. I also learned that image preprocessing, especially thresholding, plays an important role in getting accurate contour and shape detection results.
