import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for blue in HSV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Create mask; isolate blue pixels
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Find the largest contour, draw box
    if len(contours) > 0:
        # Find contour with max area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get box coordinates (x, y, width, height)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw rectangle on original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display original frame with bounding box
    cv2.imshow("Color Tracker", frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
