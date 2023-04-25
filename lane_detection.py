import cv2
import numpy as np

# Define a function for Canny edge detection
def canny(image: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    """
    Applies the Canny edge detection algorithm to an image.

    Parameters:
    image (numpy.ndarray): The input image.
    low_threshold (int): The lower threshold for the Canny algorithm.
    high_threshold (int): The higher threshold for the Canny algorithm.

    Returns:
    numpy.ndarray: The output image with the edges detected.
    """
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Reduce noise in the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using the Canny algorithm
    canny = cv2.Canny(blur, low_threshold, high_threshold)
    return canny

# Define a function to create a mask
def region_of_interest(image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Applies a mask to an image to isolate a region of interest.

    Parameters:
    image (numpy.ndarray): The input image.
    vertices (numpy.ndarray): The vertices of the polygon to be used as a mask.

    Returns:
    numpy.ndarray: The output image with the mask applied.
    """
    
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
def display_lines(image: np.ndarray, lines: np.ndarray, line_color: Tuple[int, int, int], line_thickness: int) -> np.ndarray:
  """
    Draws lines on an image.

    Parameters:
    image (numpy.ndarray): The input image.
    lines (numpy.ndarray): An array containing the lines to be drawn.
    line_color (Tuple[int, int, int]): The color of the lines.
    line_thickness (int): The thickness of the lines.

    Returns:
    numpy.ndarray: The output image with the lines drawn.
    """

       # Create an empty image with the same shape as the input image
    line_image = np.zeros_like(image)
    # Draw lines on the empty image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_thickness)
    # Merge the input image and the line image
    line_image = cv2.addWeighted(image, 0.8, line_image, 1, 0.0)
    return line_image
    
# Define a function to detect lanes in an image
def detect_lanes(image: np.ndarray) -> np.ndarray:
    """
    Detects lanes in an image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The output image with the detected lanes drawn.
    """
    
    # Define the parameters for the Canny edge detection algorithm
    low_threshold = 50
    high_threshold = 150
    # Define the vertices of the region of interest
    height, width = image.shape[:2]
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    # Detect edges using the Canny algorithm
    edges = canny(image, low_threshold, high_threshold)
    # Apply a mask to the image to isolate the region of interest
    masked_edges = region_of_interest(edges, vertices)
    # Define the parameters for the Hough transform
    rho = 2
    theta = np.pi/180
    threshold = 100
    min_line_length = 40
    max_line_gap = 5
    # Detect lines using the Hough transform
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    # Display the detected lines on the original image
    line_image = display_lines(image, lines, line_color=(255, 0, 0), line_thickness=2)
    return line_image

if __name__ == '__main__':
    # Get the path to the input image from the command line argument
    if len(sys.argv) < 2:
        print('Usage: python lane_detection.py path/to/image')
        sys.exit()
    image_path = sys.argv[1]
    # Load the image
    image = cv2.imread(image_path)
    # Detect lanes in the image
    lane_image = detect_lanes(image)
    # Display the output image
    cv2.imshow('Lane Detection', lane_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
# Load the image
image = cv2.imread("test_image.jpg")
# Create a copy of the image
lane_image = np.copy(image)

# Get the edges of the image
canny_image = canny(lane_image)

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
