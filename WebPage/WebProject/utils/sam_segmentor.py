import numpy as np
import torch
import os
import cv2
import matplotlib.cm as cm
from segment_anything import sam_model_registry, SamPredictor

# ------------------ 설정 ------------------
sam_checkpoint = r"C:\Users\khu\Downloads\CHAN\cvSH\cvSH\project_dino_midas_new\segment-anything\checkpoint\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# 전역 상태
confirmed_masks = {
    "front": [],
    "top": []
}
click_count = {
    "front": 0,
    "top": 0
}

confirmed_masks = {
    "front": [],
    "top": []
}
click_count = {
    "front": 0,
    "top": 0
}
MAX_CLICKS = 5  # 필요에 따라 클릭 수 조절

# ------------------ 클릭 처리 함수 ------------------
def handle_click_segment(x, y, view):
    global confirmed_masks, click_count

    image_path = os.path.join('static/uploads', f"{view}.jpg")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"{view} 이미지를 찾을 수 없습니다.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    best_mask = masks[np.argmax(scores)]
    confirmed_masks[view].append(best_mask)
    click_count[view] += 1

    # ✅ 새로운 미리보기 이미지 저장
    color_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    for mask in confirmed_masks[view]:
        color_mask[mask] = [255, 0, 0]
    blended = cv2.addWeighted(image_rgb, 0.6, color_mask, 0.4, 0)
    preview_path = os.path.join("static", "results", f"{view}_preview.jpg")
    cv2.imwrite(preview_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    print(f"\n📍 클릭 좌표: ({x}, {y}) @ {view}")
    print(f"✅ segment 확정: {click_count[view]}개")

    return click_count[view] >= MAX_CLICKS


# ------------------ 결과 생성 함수 ------------------
def generate_final_result(save_path):
    
    """
    front와 top 마스크들을 시각화하고 꼭짓점 추출하여 결과 이미지 생성
    """
    
    all_masks = confirmed_masks["front"] + confirmed_masks["top"]
    image_path = os.path.join('static/uploads', "top.jpg")
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    color_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    cmap = cm.get_cmap('tab20', len(all_masks))

    for idx, mask in enumerate(all_masks):
        rgba = cmap(idx)
        color = [int(255 * c) for c in rgba[:3]]
        for c in range(3):
            color_mask[:, :, c][mask] = color[c]

    alpha = 0.6
    blended = cv2.addWeighted(image_rgb, 1 - alpha, color_mask, alpha, 0)

    # 꼭짓점 시각화
    for idx, mask in enumerate(all_masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 4:
                pts = approx[:, 0, :]
                center = np.mean(pts, axis=0)
                selected_pts = [pts[np.argmax(np.linalg.norm(pts - center, axis=1))]]
                for _ in range(3):
                    dists = np.array([min(np.linalg.norm(p - s) for s in selected_pts) for p in pts])
                    next_pt = pts[np.argmax(dists)]
                    selected_pts.append(next_pt)

                for pt in selected_pts:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(blended, (x, y), 14, (255, 255, 0), -1)

    cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"[💾] 최종 결과 저장 완료: {save_path}")
