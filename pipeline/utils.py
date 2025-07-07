import os
import sys
import exifread
import cv2
import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor

# ─── 1) EXIF 정보 추출 ─────────────────────────────────────────────────────────
def extract_exif(image_path):
    """EXIF에서 focal_length 및 sensor size(mm)를 읽어옵니다."""
    from PIL import Image, ExifTags
    img = Image.open(image_path)
    width, height = img.size
    exif_data = img._getexif() or {}
    exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
    def to_float(val):
        if isinstance(val, tuple):
            num, den = val
            return num / den if den else 0
        return float(val)
    focal = exif.get('FocalLength', None)
    focal_mm = to_float(focal) if focal else None
    if focal_mm is None:
        raise ValueError('필요한 EXIF 데이터가 없습니다.')
    sensor_width_mm = 36.0
    return {
        'focal_length_mm': focal_mm,
        'sensor_width_mm': sensor_width_mm,
        'image_width': width,
        'image_height': height
    }

# ─── 2) DPT 모델 로드 (Hugging Face) ────────────────────────────────────────
def build_midas_model(model_name: str = 'Intel/dpt-large', device: str = 'cpu'):
    """
    Hugging Face에서 DPT 모델과 프로세서를 로드합니다.
    """
    model = DPTForDepthEstimation.from_pretrained(model_name).to(device).eval()
    processor = DPTImageProcessor.from_pretrained(model_name)
    return model, processor

# ─── 3) Depth 맵 예측 ───────────────────────────────────────────────────────
def predict_depth(model, processor, image_path: str, device: str = 'cpu'):
    """
    이미지 경로로부터 DPT 모델을 사용해 depth map을 생성합니다.
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
    h, w = img_bgr.shape[:2]
    depth_map = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    return depth_map

# ─── 4) 3D 크기 추정 ────────────────────────────────────────────────────────
def estimate_sizes(masks, depth_map, exif, card_actual_width_mm: float):
    """카드 기준으로 나머지 마스크 실제 폭(mm)을 추정합니다."""
    H, W = depth_map.shape
    card_mask = masks[0]
    ys, xs = np.where(card_mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError('유효한 카드 마스크 픽셀이 없습니다.')
    px_w = xs.max() - xs.min()
    fp = exif['focal_length_mm'] * (W / exif['sensor_width_mm'])
    z_card = depth_map[ys, xs].mean()
    mm_per_px = (z_card * card_actual_width_mm) / (fp * px_w)
    results = []
    for idx, mask in enumerate(masks[1:], start=1):
        yo, xo = np.where(mask)
        if yo.size == 0 or xo.size == 0:
            continue
        # 픽셀 폭/높이
        px_w_o = xo.max() - xo.min()
        px_h_o = yo.max() - yo.min()
        # 평균 깊이
        z_vals = depth_map[yo, xo]
        z_o_mean    = float(z_vals.mean())  
        z_o_min     = float(z_vals.min())
        z_o_max     = float(z_vals.max())
        thickness_mm = z_o_max - z_o_min
        real_w = px_w_o * mm_per_px * (z_o_mean / z_card)
        real_h = px_h_o * mm_per_px * (z_o_mean / z_card)
        results.append({
            'mask_index':          idx,
            'pixel_width':         int(px_w_o),
            'pixel_height':        int(px_h_o),
            'estimated_width_mm':   real_w,        # 실측 가로(mm)
            'estimated_height_mm':  real_h,        # 실측 세로(mm)
            'estimated_depth_mm':   thickness_mm   # 실측 깊이(두께, mm)


        })
    return results
