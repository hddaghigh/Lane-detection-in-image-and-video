import cv2
import numpy as np

def canny(image, low_threshold, high_threshold):
    """
    Applies the Canny edge detection algorithm to an image.

    Parameters:
    image (numpy.ndarray): The input image.
    low_threshold (int): The lower threshold for the Canny algorithm.
    high_threshold (int): The higher threshold for the Canny algorithm.

    Returns:
    numpy.ndarray: The output image with the edges detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    return edges

def region_of_interest(image, vertices):
    """
    Applies a mask to an image to isolate a region of interest.

    Parameters:
    image (numpy.ndarray): The input image.
    vertices (numpy.ndarray): The vertices of the polygon to be used as a mask.

    Returns:
    numpy.ndarray: The output image with the mask applied.
    """
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines, line_color=(0, 0, 255), line_thickness=2):
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
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_thickness)
    return cv2.addWeighted(image, 0.8, line_image, 1, 0)

def make_coordinates(image, line_parameters):
    """
    Calculates the coordinates of the endpoints of a line.

    Parameters:
    image (numpy.ndarray): The input image.
    line_parameters (numpy.ndarray): An array containing the parameters of the line.

    Returns:
    numpy.ndarray: An array containing the coordinates of the endpoints of the line.
    """
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    """
    Calculates the average slope and intercept of the detected lane lines.

    Parameters:
    image (numpy.ndarray): The input image.
    lines (numpy.ndarray): An array containing the detected lines.

    Returns:
    Tuple[numpy.ndarray, numpy.ndarray]: An array containing the parameters of the left lane line
    and an array containing the parameters of the right lane line.
    """
    left_fit = []
    right_fit = []
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    else:
        left_line = None

    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
    else:
        right_line = None

    return np.array([left_line, right_line])

def detect_lanes(image):
    """
    Detects the left and right lane lines in an image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The output image with the lane lines drawn.
    """
    height, width, _ = image.shape
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    edges = canny(image, 50, 150)
    cropped_edges = region_of_interest(edges, vertices)
    lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(image, lines)
    line_image = display_lines(image, averaged_lines)
    return line_image

def detect_lanes_video(input_path, output_path):
    """
    Detects the left and right lane lines in a video file and saves the result to a new file.

    Parameters:
    input_path (str): The path to the input video file.
    output_path (str): The path to the output video file.

    Returns:
    None
    """
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            processed_frame = detect_lanes(frame)
            out.write(processed_frame)
            cv2.imshow('Lane Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

To detect the lane lines in a video file, you can use the detect_lanes_video function, passing in the path to the input video file and the path to the output video file. For example:

input_path = 'test_videos/solidWhiteRight.mp4'
output_path = 'output_videos/solidWhiteRight_output.mp4'
detect_lanes_video(input_path, output_path)

This will save a new video file with the detected lane lines drawn on each frame. You can adjust the parameters in the detect_lanes function as needed to improve the accuracy of the lane detection.