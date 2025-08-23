import cv2
import numpy as np

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Define the range for the color blue in HSV
    # You can change these values to track different colors
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # 3. Create a mask to isolate only the blue pixels
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 4. Find contours (outlines) in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Find the largest contour and draw a bounding box
    if len(contours) > 0:
        # Find the contour with the maximum area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box coordinates (x, y, width, height)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw the rectangle on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original frame with the bounding box
    cv2.imshow("Color Tracker", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()