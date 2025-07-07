import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os


def segment_all_full(image_path, sam_ckpt, sam_model_type="vit_b", device="cuda"):
    """
    위/옆 이미지에 대해 SAM을 이용해 전체 객체 마스크를 생성합니다.

    Args:
        image_path (str): 입력 이미지 파일 경로
        sam_ckpt (str): SAM checkpoint 파일 경로
        sam_model_type (str): SAM 모델 타입 (e.g., 'vit_b')
        device (str): 실행 디바이스 ('cpu' or 'cuda')

    Returns:
        image (np.ndarray): BGR로 로드된 원본 이미지
        masks (list of dict): 각 마스크 정보(dict) 리스트 ('segmentation', 'bbox', ...)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # SAM은 RGB 입력을 기대하므로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # SAM 모델 로드
    sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt)
    sam.to(device=device)
    # 자동 마스크 생성
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image_rgb)
    return image, masks


def user_select_masks(image, masks):
    # headless fallback
    if not os.environ.get('DISPLAY'):
        os.makedirs('mask_previews', exist_ok=True)
        for idx, m in enumerate(masks):
            overlay = image.copy()
            seg = (m['segmentation'].astype(np.uint8) * 255)
            cnts, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 5)
            path = f'mask_previews/mask_{idx}.jpg'
            cv2.imwrite(path, overlay)
            print(f'Mask {idx} preview saved to {path}')
        sel = input('Select mask indices (comma-separated): ')
        idxs = [int(i) for i in sel.split(',') if i.strip().isdigit()]
        # 선택한 인덱스로 마스크 리스트 생성 및 좌->우 정렬
        selected = [masks[i] for i in idxs]
        selected_sorted = sorted(selected, key=lambda m: m['bbox'][0])
        return selected_sorted

    # GUI mode
    selected = []
    mask_array = [m['segmentation'] for m in masks]
    contours = []
    for seg in mask_array:
        seg_uint8 = (seg.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(seg_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.append(cnts)

    overlay = image.copy()
    win = "Select Masks"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, seg in enumerate(mask_array):
                if seg[y, x] and idx not in selected:
                    selected.append(idx)
                    cv2.drawContours(overlay, contours[idx], -1, (0, 255, 0), 5)
                    cv2.imshow(win, overlay)
                    break

    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, overlay)
    print("Click masks then press 'q' to finish.")
    while True:
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    sorted_indices = sorted(selected, key=lambda i: masks[i]['bbox'][0])
    return sorted_indices


def measure_selected_masks(masks, selected_indices, box, image_shape):
    """
    선택된 마스크들(selected_indices)에 대해 실제 크기(width_mm, height_mm)를 계산합니다.
    카드 마스크는 박스(box)와의 IoU로 자동 검출합니다.

    Args:
        masks: segment_all_full이 반환한 마스크 리스트
        selected_indices: user_select_masks가 반환한 원본 인덱스 리스트
        box: detect_card이 반환한 [cx, cy, bw, bh] (정규화된 좌표)
        image_shape: segmentation에 사용된 원본 이미지의 shape

    Returns:
        measurements: list of dicts with keys ['mask_idx','width_mm','height_mm']
    """
    h, w = image_shape[:2]
    cx, cy, bw, bh = box
    cx_px, cy_px = cx * w, cy * h
    bw_px, bh_px = bw * w, bh * h
    x1 = int(cx_px - bw_px/2); y1 = int(cy_px - bh_px/2)
    x2 = int(cx_px + bw_px/2); y2 = int(cy_px + bh_px/2)

    # 카드 영역 마스크 생성
    mask_card = np.zeros((h, w), dtype=bool)
    mask_card[y1:y2, x1:x2] = True

    # IoU로 카드 마스크 인덱스 검출
    ious = []
    for m in masks:
        seg = m['segmentation'].astype(bool)
        inter = np.logical_and(seg, mask_card).sum()
        union = np.logical_or(seg, mask_card).sum()
        ious.append(inter/union if union>0 else 0)
    card_idx = int(np.argmax(ious))

    # 스케일 계산
    scale_w = bw_px / CARD_WIDTH_MM
    scale_h = bh_px / CARD_HEIGHT_MM

    measurements = []
    for idx in selected_indices:
        if idx == card_idx:
            continue
        seg = masks[idx]['segmentation'].astype(bool)
        ys, xs = np.where(seg)
        w_mm = (xs.max() - xs.min()) / scale_w
        h_mm = (ys.max() - ys.min()) / scale_h
        measurements.append({'mask_idx': idx, 'width_mm': w_mm, 'height_mm': h_mm})
    return measurements
