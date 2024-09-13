import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from modules.midas.utils import write_png, write_pfm, write_depth
from read_write_model import read_model, qvec2rotmat


def pts3d_to_sparse_depth(cam, img, pt3d, depth_scale=None):  # depth_scale = 65535 * 0.1
    def projection(R, t, K, pts3d):
        extrinsic = np.hstack((R, t[:, None]))
        proj_mat = K @ extrinsic
        pts2d = proj_mat @ np.append(pts3d, 1)
        pts2d = (pts2d / pts2d[2])[:2]
        return pts2d.round()
    fx, fy, cx, cy = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R = qvec2rotmat(img.qvec)
    t = img.tvec
    img_pt3dIds = img.point3D_ids[img.point3D_ids != -1]
    img_pt3ds = np.array([pt3d[pt3dId].xyz for pt3dId in img_pt3dIds])
    sparse_depth = np.zeros((cam.height, cam.width))
    for img_pt3d in img_pt3ds:
        u, v = projection(R, t, K, img_pt3d)
        z_depth = (R @ img_pt3d + t)[2]
        if 0 <= u < cam.width and 0 <= v < cam.height:
            sparse_depth[int(v), int(u)] = z_depth
    # depth_scale = 65535 * 0.1
    if depth_scale is not None:
        sparse_depth = sparse_depth * depth_scale
    return sparse_depth


def write_sparse_depth(model_path, output_path):
    if not output_path.exists():
        output_path.mkdir(parents=True)
    cameras, images, points3D = read_model(model_path)
    for img in tqdm(images.values()):
        cam = cameras[img.camera_id]
        sparse_depth = pts3d_to_sparse_depth(cam, img, points3D)
        # sparse_depth_path = output_path / f"{(img.name).replace('frame', 'sparse')[:-4]}"
        sparse_depth_path = output_path / f"{(img.name).replace('frame', 'sparse').replace('.jpg', '.png')}"
        # write_depth(sparse_depth_path.as_posix(), sparse_depth, bits=2)
        cv2.imwrite(sparse_depth_path.as_posix(), (sparse_depth * 255).astype(np.uint16))
        # cv2.imwrite(sparse_depth_path.as_posix(), sparse_depth.astype(np.uint16))
        write_pfm(sparse_depth_path.with_suffix(".pfm").as_posix(), sparse_depth.astype(np.float32))


if __name__ == "__main__":
    model_path = Path("dataset/replica_capture/room_0_traj1_plus2/sparse/loftr/sfm")
    output_path = Path("dataset/replica_capture/room_0_traj1_plus2/sparse_depth_loftr")
    write_sparse_depth(model_path, output_path)