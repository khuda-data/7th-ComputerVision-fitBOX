# pipeline/backproject.py
import json, cv2, numpy as np, open3d as o3d, argparse

def backproject_and_save(args):
    # (이전 backproject.py 로직을 함수로 옮기고, argparse를 이 모듈에 넣어둡니다.)
    # – depth 읽기
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if depth.ndim == 3: depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    depth = depth.astype(np.float32)/255.0
    # – K, pts3d 계산…
    # – box 로딩, scale 적용…
    # – Octree 변환 & 저장…

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", required=True)
    parser.add_argument("--box",    required=True)
    parser.add_argument("--scale",  type=float, required=True)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--size_expand", type=float, default=0.01)
    args = parser.parse_args()
    backproject_and_save(args)
