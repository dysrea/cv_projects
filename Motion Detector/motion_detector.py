import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Capture first frame to use as static background reference
ret, background_frame = cap.read()
if not ret:
    print("Failed to grab the first frame. Please check your webcam.")
    exit()

# Convert background frame to grayscale and apply blur to reduce noise
background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

print("Background set. Monitoring for motion...")

while True:
    # Read new frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Compute absolute difference between background and current frame
    # This difference image will highlight areas where pixels have changed (motion)
    diff_frame = cv2.absdiff(background_gray, gray)

    # Apply threshold to the difference image.
    # Pixels with difference greater than 30 will be set to 255 (white), others to black.
    thresh = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate thresholded image to fill in small holes and make motion blobs more solid
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours (outlines) of the white motion blobs
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the found contours
    for contour in contours:
        # If contour is too small, ignore (filters out noise)
        if cv2.contourArea(contour) < 1000:
            continue
        
        # Otherwise, calculate bounding box and draw it on original color frame
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display final frame with motion detections
    cv2.imshow("Motion Detector", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()