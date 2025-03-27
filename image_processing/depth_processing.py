import cv2
import numpy as np

def generate_depth_map(image):
    if image is None or image.size == 0:
        raise ValueError("유효하지 않은 이미지입니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return depth_map

def generate_3d_points(gray_image, scale=1.0):
    if len(gray_image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다.")

    rows, cols = gray_image.shape
    points_3d = []
    for y in range(rows):
        for x in range(cols):
            z = gray_image[y, x] * scale
            points_3d.append((x, y, z))

    return np.array(points_3d, dtype=np.float32)
