# pipeline/detect.py
import os
import sys
import numpy as np

# 1) GroundingDINO 모듈을 import 할 수 있도록 경로 추가
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
GDL_PATH = os.path.join(ROOT, "GroundingDINO")
if GDL_PATH not in sys.path:
    sys.path.insert(0, GDL_PATH)

from groundingdino.util.inference import load_model, load_image, predict

def detect_card(image_path, cfg, weights, device="cpu"):
    # 2) 모델 로드 (device 인자 전달)
    model = load_model(cfg, weights, device=device)
    src, img = load_image(image_path)

    # 3) predict에도 device 전달
    boxes, logits, phrases = predict(
        model=model,
        image=img,
        caption="small creditcard",
        box_threshold=0.2,      # 예시: 0.7로 조정
        text_threshold=0.25,
        device=device
    )

    # 4) "creditcard" phrase 필터링
    cc_idxs = [i for i, p in enumerate(phrases) if "creditcard" in p.lower()]
    if not cc_idxs:
        raise RuntimeError("신용카드를 찾을 수 없습니다.")

    # 5) 그중 confidence(logits)가 가장 높은 박스 하나 선택
    best = cc_idxs[int(np.argmax([logits[i] for i in cc_idxs]))]
    x0, y0, x1, y1 = boxes[best]
    return np.array([x0, y0, x1, y1], dtype=np.float32)
