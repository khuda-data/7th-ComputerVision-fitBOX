import argparse, json, subprocess, sys
import numpy as np, cv2
import os, glob, csv
from .detect import detect_card
from .segment import segment_all_full, user_select_masks


# Credit card 실제 크기(mm)
CARD_WIDTH_MM = 85.60  # 가로
CARD_HEIGHT_MM = 53.98  # 세로


def main():
    p = argparse.ArgumentParser()
    # 두 개의 이미지: 위에서 찍은 사진과 옆에서 찍은 사진
    p.add_argument("--top_image", required=True, help="Top view input image path")
    p.add_argument("--side_image", required=True, help="Side view input image path")
    p.add_argument("--dino_cfg", required=True, help="GroundingDINO config file")
    p.add_argument("--dino_weights", required=True, help="GroundingDINO weights file")
    p.add_argument("--sam_ckpt", required=True, help="SAM checkpoint file")
    p.add_argument("--sam_model", default="vit_b", help="SAM model type")
    p.add_argument("--device", default="cuda", help="Compute device, e.g. cpu or cuda")
    args = p.parse_args()

    # 가운데 JSON 저장 경로 설정
    top_base, _ = os.path.splitext(args.top_image)
    box_json = top_base + "_box.json"

    # 1) 카드 박스 검출 (Top view)
    box = detect_card(
        args.top_image,
        args.dino_cfg,
        args.dino_weights,
        device=args.device
    )
    with open(box_json, "w", encoding='utf-8') as f:
        json.dump({"box": box.tolist()}, f, ensure_ascii=False, indent=2)



    # ── GroundingDINO 박스로 픽셀→mm 스케일 계산 ──
    # args.top_image 불러와서 w,h 얻고, box=[cx,cy,bw,bh] 정규화 좌표를 픽셀로 환산
    img_for_scale = cv2.imread(args.top_image)
    h, w = img_for_scale.shape[:2]
    cx, cy, bw, bh = box.tolist()
    pix_box_w = bw * w    # 카드의 픽셀 가로 길이
    pix_box_h = bh * h    # 카드의 픽셀 세로 길이
    scale_w = pix_box_w / CARD_WIDTH_MM   # 1mm 당 픽셀 수
    scale_h = pix_box_h / CARD_HEIGHT_MM  # 1mm 당 픽셀 수

     # 2) Top-view 분할 및 사용자가 선택한 마스크
    img_top, masks_top = segment_all_full(
        args.top_image, args.sam_ckpt, args.sam_model, args.device
    )
    selected_top = user_select_masks(img_top, masks_top)

    # 3) 선택된 마스크에 대해 width_mm, height_mm 계산, depth_mm=0 으로 CSV 저장
    csv_path = top_base + "_top_measurements.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mask_num', 'width_mm', 'height_mm', 'depth_mm'])
        for i, m in enumerate(selected_top):
            seg = m['segmentation'].astype(bool)
            ys, xs = np.where(seg)
            w_mm = (xs.max() - xs.min()) / scale_w
            h_mm = (ys.max() - ys.min()) / scale_h
            writer.writerow([i, w_mm, h_mm, 0.0])
    print(f"Top-view measurements (depth=0) saved to {csv_path}")
    #Side-view 분할 및 사용자가 선택한 마스크
    img_side, masks_side = segment_all_full(
        args.side_image, args.sam_ckpt, args.sam_model, args.device
    )
    selected_side = user_select_masks(img_side, masks_side)

    # 6) Top-view CSV 로드
    measurements = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            measurements.append({
                'mask_num': int(row['mask_num']),
                'width_mm': float(row['width_mm']),
                'height_mm': float(row['height_mm']),
                'depth_mm': float(row['depth_mm'])
            })

    # 7) Side-view에서 각 객체별 depth_mm 계산 (x축 비교→y축 변환)
    for i, m in enumerate(selected_side):
        seg = m['segmentation'].astype(bool)
        ys, xs = np.where(seg)
        pix_w = xs.max() - xs.min()
        pix_h = ys.max() - ys.min()
        # side-view x축 픽셀 대비 top-view 실측 여비로 scale
        scale_side = pix_w / measurements[i]['width_mm']
        depth_mm = pix_h / scale_side
        measurements[i]['depth_mm'] = depth_mm

    # 8) 최종 CSV 덮어쓰기
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mask_num', 'width_mm', 'height_mm', 'depth_mm'])
        for m in measurements:
            writer.writerow([m['mask_num'], m['width_mm'], m['height_mm'], m['depth_mm']])
    print(f"Updated with side-view depth in {csv_path}")
if __name__ == "__main__":
    main()
