import cv2
import numpy as np

# Define a function for Canny edge detection
def canny(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Reduce noise in the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using the Canny algorithm
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Define a function to create a mask
def region_of_interest(image):
    # Get the height and width of the image
    height = image.shape[0]
    width = image.shape[1]
    # Define the polygon for the mask
    polygons = np.array([[(0, height), (width, height), (width, 320), (0, 320)]])
    # Create an empty mask
    mask = np.zeros_like(image)
    # Fill the polygon with white
    cv2.fillPoly(mask, polygons, 255)
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Define a function to draw lines on the image
def display_lines(image, lines):
    # Create an empty image with the same size as the original
    line_image = np.zeros_like(image)
    # If there are no lines, return the original image
    if lines is None:
        return image
    # Define the color and thickness of the lines
    line_color = (0, 255, 0)
    line_thickness = 3
    # Iterate over the lines and draw them on the image
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_thickness)
    # Merge the original image and the line image
    merged_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return merged_image

# Load the image
image = cv2.imread("test_image.jpg")
# Create a copy of the image
lane_image = np.copy(image)

# Get the edges of the image
canny_image = canny(lane_image)

# Define the vertices of the region of interest
height = image.shape[0]
width = image.shape[1]
vertices = np.array([[(0, height), (width, height), (width, 320), (0, 320)]])

# Create a mask for the image
cropped_image = region_of_interest(canny_image)

# Detect lines in the image using the Hough transform
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# Draw lines on the image
line_image = display_lines(lane_image, lines)

# Show the image
cv2.imshow("result", line_image)
cv2.waitKey(0)

# Release the window and de-allocate memory associated with it
cv2.destroyAllWindows()```

Please note that the code assumes that there is an image file named "test_image.jpg" in the same directory as the script. You may need to modify the file path if your image is stored elsewhere.
