from datasets import load_dataset

# food-101 데이터셋에서 샘플 하나 불러오기
dataset = load_dataset("food101", split="train[:1%]")  # 전체 중 1%만 로딩
sample = dataset[0]

# 이미지 저장
sample['image'].save("sample.jpg")  # 이후 OpenCV로 불러올 수 있게 저장


import cv2
import numpy as np

# 이미지로드
image = cv2.imread('sample.jpg')  # 분석할 이미지 파일

# BGR에서 HSV 색상공간으로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색 범위 지정(두 개의 범위를 설정해야 함)
lower_red1 = np.array([0, 120, 70])  
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])  
upper_red2 = np.array([180, 255, 255])

# 마스크 생성
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2  # 두 개의 마스크를 합침

# 원본 이미지에서 빨간색 부분만 추출
result = cv2.bitwise_and(image, image, mask=mask)

# 결과 이미지 출력
cv2.imshow('Original', image)
cv2.imshow('Red Filtered', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과: 빨간색 영역이 검출되며, 다른 색상은 제거된 상태로 표시됨.


import os

os.makedirs("preprocessed_samples", exist_ok=True)

# (1) 크기 조정
resized = cv2.resize(image, (224, 224))
cv2.imwrite("preprocessed_samples/resized.png", resized)

# (2) Grayscale + Normalize
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
normalized = gray / 255.0
norm_img = (normalized * 255).astype(np.uint8)
cv2.imwrite("preprocessed_samples/gray_norm.png", norm_img)

# (3) Blur
blurred = cv2.GaussianBlur(resized, (5, 5), 0)
cv2.imwrite("preprocessed_samples/blurred.png", blurred)

# (4) 데이터 증강
flipped = cv2.flip(resized, 1)
cv2.imwrite("preprocessed_samples/flipped.png", flipped)

rotated = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("preprocessed_samples/rotated.png", rotated)

hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 50)  # 채도값 +50
enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite("preprocessed_samples/enhanced.png", enhanced)
