import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

# SAM 초기화
sam_checkpoint = r"C:\Users\khu\Downloads\CHAN\cvSH\cvSH\project_dino_midas_new\segment-anything\checkpoint\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

def run_sam(image_path, click_x, click_y, output_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    input_point = np.array([[click_x, click_y]])
    input_label = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    mask = masks[np.argmax(scores)]

    # 마스크 결과 시각화
    result = image_rgb.copy()
    result[mask] = [0, 255, 0]
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    return output_path
