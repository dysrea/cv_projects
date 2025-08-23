import cv2
import numpy as np

# 1. Read the two images
img_left = cv2.imread('image_left.jpg')
img_right = cv2.imread('image_right.jpg')

if img_left is None or img_right is None:
    print("Error: Could not read one or both images.")
    exit()

# 2. Find keypoints and descriptors using ORB
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img_left, None)
keypoints2, descriptors2 = orb.detectAndCompute(img_right, None)

# 3. Match features
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
num_good_matches = int(len(matches) * 0.15)
good_matches = matches[:num_good_matches]

# 4. Find the Homography matrix
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

# --- THE FIX IS IN THIS SECTION ---

# 5. Get the dimensions of the images
h1, w1, _ = img_left.shape
h2, w2, _ = img_right.shape

# 6. Get the corner points of the right image
corners_right = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape(-1, 1, 2)

# 7. Transform the corners using the homography matrix
transformed_corners = cv2.perspectiveTransform(corners_right, matrix)

# 8. Calculate the required size of the output canvas
all_corners = np.concatenate((np.float32([[0,0], [0,h1-1], [w1-1,h1-1], [w1-1,0]]).reshape(-1,1,2), transformed_corners), axis=0)
x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# Create a translation matrix to move the stitched image into the visible area
translation_dist = [-x_min, -y_min]
matrix_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

# 9. Warp the right image onto a canvas that is the correct final size
final_width = x_max - x_min
final_height = y_max - y_min
warped_img = cv2.warpPerspective(img_right, matrix_translation.dot(matrix), (final_width, final_height))

# 10. Place the left image onto the canvas
warped_img[translation_dist[1]:h1+translation_dist[1], translation_dist[0]:w1+translation_dist[0]] = img_left

# Display the final, correctly sized panorama
cv2.imshow("Panorama", warped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()