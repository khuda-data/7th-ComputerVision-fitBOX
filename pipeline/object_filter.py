
# pipeline/object_filter.py

import numpy as np
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def filter_object_masks(masks, image_path, threshold=0.5):
    # CLIP 모델 로드
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 원본 이미지 로드 및 RGB 변환
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    prompts = ["a photo of an object", "a photo of floor"]
    filtered = []

    for mask in masks:
        # 마스크 영역 바운딩 박스 계산
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            continue
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        # 크롭 & 리사이즈
        crop = img_rgb[y1:y2+1, x1:x2+1]
        if crop.size == 0:
            continue
        pil_img = Image.fromarray(crop).resize((224, 224))

        # CLIP 인퍼런스
        inputs = clip_processor(text=prompts, images=pil_img, return_tensors="pt", padding=True)
        logits = clip_model(**inputs).logits_per_image[0]
        probs = logits.softmax(dim=0)
        obj_score = probs[0].item()  # 'object' 확률

        if obj_score >= threshold:
            filtered.append((mask, obj_score))

    return filtered


def merge_overlapping_masks(mask_score_list, contain_thresh=0.9):
    """
    filter_object_masks() 결과 리스트에서
    작은 마스크가 큰 마스크에  contain_thresh 이상 포함된 경우에만
    같은 객체로 묶어 합집합(union)으로 반환합니다.

    반환: List[np.ndarray]
    """
    masks = [m for m, _ in mask_score_list]
    n = len(masks)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []

        while stack:
            u = stack.pop()
            comp.append(u)
            area_u = masks[u].sum()
            for v in range(n):
                if visited[v]:
                    continue
                area_v = masks[v].sum()
                inter = np.logical_and(masks[u], masks[v]).sum()
                small_area = min(area_u, area_v)
                if small_area > 0 and (inter / small_area) >= contain_thresh:
                    visited[v] = True
                    stack.append(v)

        union = np.zeros_like(masks[0], dtype=bool)
        for idx in comp:
            union |= masks[idx]
        clusters.append(union)

    return clusters

