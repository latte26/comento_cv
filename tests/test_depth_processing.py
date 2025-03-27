import pytest
import cv2
import numpy as np
from image_processing.depth_processing import generate_depth_map, generate_3d_points

def test_generate_depth_map_valid():
    dummy_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    depth_map = generate_depth_map(dummy_image)
    assert depth_map is not None
    assert depth_map.shape == (100, 100, 3)

def test_generate_depth_map_invalid_input():
    with pytest.raises(ValueError):
        invalid_image = np.array([])
        generate_depth_map(invalid_image)

def test_generate_3d_points_valid():
    dummy_gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    points_3d = generate_3d_points(dummy_gray, scale=0.5)
    assert points_3d is not None
    assert points_3d.shape[1] == 3
    assert np.all(points_3d[:, 2] >= 0)
    assert np.all(points_3d[:, 2] <= 127.5)

def test_generate_3d_points_invalid_input():
    with pytest.raises(ValueError):
        dummy_color = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        generate_3d_points(dummy_color)

