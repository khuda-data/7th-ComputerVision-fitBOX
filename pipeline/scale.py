# pipeline/scale.py
import numpy as np
import open3d as o3d

def estimate_scale(depth: np.ndarray, box: np.ndarray, real_width_mm=85.6, samples=200):
    H, W = depth.shape
    x0,y0,x1,y1 = box
    xs = np.linspace(int(x0*W), int(x1*W)-1, samples).astype(int)
    ys = np.linspace(int(y0*H), int(y1*H)-1, samples).astype(int)
    uu, vv = np.meshgrid(xs, ys)
    idx = vv.ravel() * W + uu.ravel()

    fx = fy = 0.5 * W
    cx, cy = W/2, H/2
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
    invK = np.linalg.inv(K)
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([us, vs, np.ones_like(us)], -1).reshape(-1,3)
    pts3d = (invK @ (pix * depth.reshape(-1)) .T).T

    card_pts = pts3d[idx]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(card_pts))
    box3d = pcd.get_axis_aligned_bounding_box()
    Lrec = np.linalg.norm(box3d.get_max_bound() - box3d.get_min_bound())
    return real_width_mm / Lrec
