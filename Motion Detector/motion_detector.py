import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Capture the first frame to use as the static background reference
ret, background_frame = cap.read()
if not ret:
    print("Failed to grab the first frame. Please check your webcam.")
    exit()

# Convert the background frame to grayscale and apply a blur to reduce noise
background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

print("Background set. Monitoring for motion...")

while True:
    # Read a new frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Compute the absolute difference between the background and the current frame
    # This difference image will highlight areas where pixels have changed (i.e., motion)
    diff_frame = cv2.absdiff(background_gray, gray)

    # Apply a threshold to the difference image.
    # Pixels with a difference greater than 30 will be set to 255 (white), others to black.
    thresh = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in small holes and make motion blobs more solid
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find the contours (outlines) of the white motion blobs
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the found contours
    for contour in contours:
        # If a contour is too small, ignore it (this filters out noise)
        if cv2.contourArea(contour) < 1000:
            continue
        
        # Otherwise, calculate the bounding box and draw it on the original color frame
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the final frame with the motion detections
    cv2.imshow("Motion Detector", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()