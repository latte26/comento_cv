import cv2
import numpy as np
import os

# 결과 저장 폴더
output_dir = "preprocessed_samples"
os.makedirs(output_dir, exist_ok=True)

# 원본 이미지 로드 (직접 파일명 수정해도 됨)
image = cv2.imread("sample.jpg")

# 1. 크기 조정
resized = cv2.resize(image, (224, 224))

# 2. Grayscale + Normalize
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
normalized = gray / 255.0

# 3. Blur 필터 적용
blurred = cv2.GaussianBlur(resized, (5, 5), 0)

# 4. 좌우 반전
flipped = cv2.flip(resized, 1)

# 5. 색상 변경 (HSV → 색감 강조)
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
hsv[..., 1] = cv2.equalizeHist(hsv[..., 1])
color_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 저장
cv2.imwrite(f"{output_dir}/resized.png", resized)
cv2.imwrite(f"{output_dir}/gray.png", (normalized * 255).astype(np.uint8))
cv2.imwrite(f"{output_dir}/blurred.png", blurred)
cv2.imwrite(f"{output_dir}/flipped.png", flipped)
cv2.imwrite(f"{output_dir}/enhanced.png", color_enhanced)

from datasets import load_dataset
from PIL import Image
import os

# 데이터셋 로드
dataset = load_dataset("food101", split="train[:1%]")  # 전체의 1%만 사용 (샘플용)

# 저장 폴더 설정
save_dir = "preprocessed_samples"
os.makedirs(save_dir, exist_ok=True)

# 이미지 저장
for idx, sample in enumerate(dataset):
    img: Image.Image = sample["image"]
    img.save(os.path.join(save_dir, f"sample_{idx}.jpg"))
    
    if idx == 4:  # 샘플 5개만 저장
        break

