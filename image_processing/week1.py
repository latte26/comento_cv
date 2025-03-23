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

def is_too_dark(img_bgr, brightness_threshold=50):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    return mean_val < brightness_threshold

def is_too_small(img_bgr, min_area=1000):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True
    max_area = max(cv2.contourArea(cnt) for cnt in contours)
    return max_area < min_area

# 전처리 전에 필터링
if is_too_dark(image):
    print("어두워서 필터링됨!")
else:
    if is_too_small(image):
        print("너무 작아서 필터링됨!")
    else:
        print("정상 이미지! -> 전처리 수행")
        # 여기서 크기 조정, Grayscale, Blur, 증강 등 진행
        # preprocessed_samples 폴더에 저장

