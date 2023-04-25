import cv2
import numpy as np
from lane_detection import canny, region_of_interest, display_lines

def test_canny():
    # Test canny() function
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 20:80] = 255
    canny_image = canny(image, 50, 150)
    assert canny_image[25, 25] == 0
    assert canny_image[75, 75] == 0
    assert canny_image[25, 75] == 255
    assert canny_image[75, 25] == 255

def test_region_of_interest():
    # Test region_of_interest() function
    image = np.zeros((100, 100), dtype=np.uint8)
    vertices = np.array([[(20, 50), (80, 50), (80, 80), (20, 80)]])
    masked_image = region_of_interest(image, vertices)
    assert masked_image[40, 40] == 0
    assert masked_image[60, 60] == 0
    assert masked_image[60, 70] == 255
    assert masked_image[40, 70] == 255

def test_display_lines():
    # Test display_lines() function
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    lines = np.array([[[10, 20, 90, 80]], [[20, 10, 80, 90]], [[30, 20, 70, 80]], [[20, 30, 80, 70]]])
    line_image = display_lines(image, lines, line_color=(0, 255, 0), line_thickness=3)
    assert np.array_equal(line_image[50, 50], np.array([0, 255, 0]))
