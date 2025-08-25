# Real-time Object Tracking Comparison

A Python project to detect, track, and compare the performance of multiple state-of-the-art object tracking algorithms (**ByteTrack**, **BoT-SORT**) using the **YOLOv8** object detector and **OpenCV**.

## Core Features
* **YOLOv8 Detection**: Fast and accurate object detection.
* **Multi-Tracker Support**: Implements three different modern trackers for comparison.

## Tech Stack
* Python
* OpenCV
* Ultralytics YOLOv8
* NumPy

## Quick Start
1.  **Clone & Setup Environment:**
    ```bash
    git clone [https://github.com/dysrea/cv_projects.git](https://github.com/dysrea/cv_projects.git)
    cd cv_projects
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

2.  **Install Dependencies:**
    ```bash
    pip install ultralytics opencv-contrib-python numpy deep-sort-realtime
    ```

3.  **Run the Tracker:**
    Place your video in the folder and run the main script.
    ```bash
    python tracker.py
    ```
