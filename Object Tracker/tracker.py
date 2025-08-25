from ultralytics import YOLO
import os

VIDEO_SOURCE = "test_video.mp4"
MODEL_PATH = "yolov8n.pt"
TRACKERS_TO_TEST = ["bytetrack.yaml", "botsort.yaml"]

# Load model
model = YOLO(MODEL_PATH)

# Loop through trackers
for tracker_config in TRACKERS_TO_TEST:
    tracker_name = os.path.splitext(tracker_config)[0]
    print(f"\n--- Running tracker: {tracker_name} ---")

    model.track(
        source=VIDEO_SOURCE,
        tracker=tracker_config,
        save=True,
        name=f"{tracker_name}_run"
    )

print("\n--- All tracking tests are complete! ---")

