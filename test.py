import cv2
import numpy as np

# Read the image
image = cv2.imread("\\Users\yunis\PycharmProjects\pythonProject1\images\image_1.jpg")

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for red color
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Define the lower and upper bounds for red color
lower_orange = np.array([0, 100, 100])
upper_orange = np.array([30, 255, 255])


# Create masks for red color (two ranges)
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
mask3 = cv2.inRange(hsv_image,lower_orange,upper_orange)


# Combine the masks to get the final red and orange  mask
mask = mask1 + mask2 + mask3

# Perform morphological operations to reduce noise in the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Draw the detected contours on the original image
for contour in contours:
    cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
    # Get the bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)

    # Draw the bounding box on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Write text inside the bounding box

    text = "Object"
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

import cv2

def save_image(image_path, output_path):
    # Read the image
    contours = cv2.imread(image_path)

    # Save the image to the specified output path
    cv2.imwrite(output_path, image)

# Example usage:
input_image_path = "\\Users\yunis\PycharmProjects\pythonProject1\images\image_1.jpg"
output_image_path = "\\Users\yunis\PycharmProjects\pythonProject1\images\image_1o.jpg"
save_image(input_image_path, output_image_path)


# Display the result

cv2.imshow("Object Detection using color", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

