import os
from typing import List, Dict

import numpy as np
import imageio.v2 as imageio

from explorer import Scene


def build_camera_matrix(pos, target, up=np.array([0, 1, 0.0], dtype=np.float32)):
    C = np.array(pos, dtype=np.float32)
    T = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = T - C
    f = f / (np.linalg.norm(f) + 1e-9)
    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)

    R = np.stack([r, u, f], axis=0)   # world -> camera
    t = -R @ C

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def build_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = 0.5 * width / np.tan(fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [[fx, 0,  cx],
         [0,  fy, cy],
         [0,   0,  1]],
        dtype=np.float32,
    )
    return K


class Renderer:
    def __init__(
        self,
        scene: Scene,
        out_dir: str,
        width: int = 960,
        height: int = 544,
        fov_deg: float = 70.0,
    ):
        self.scene = scene
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.width = width
        self.height = height
        self.fov_deg = fov_deg

        # supersampling factor – set to 1 for speed
        self.ss = 1
        self.ss_w = width * self.ss
        self.ss_h = height * self.ss

        self.K = build_intrinsics(self.ss_w, self.ss_h, fov_deg)

        print("[Renderer] FAST point renderer (no supersampling, vectorized Z-buffer)")
        print(f"           Resolution: {width}x{height}, FOV={fov_deg}")

    def render_frame(self, pose: Dict) -> np.ndarray:
        xyz = self.scene.xyz  # (N,3)
        rgb = self.scene.rgb  # (N,3), [0,1]

        C = pose["pos"]
        T = pose["target"]
        up = pose.get("up", np.array([0, 1, 0.0], dtype=np.float32))

        M = build_camera_matrix(C, T, up)

        # world -> camera
        xyz_h = np.concatenate(
            [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)],
            axis=1,
        )   # (N,4)
        cam = (M @ xyz_h.T).T
        Xc, Yc, Zc = cam[:, 0], cam[:, 1], cam[:, 2]

        # keep only points in front of camera
        front_mask = Zc > 0.05
        if not np.any(front_mask):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        Xc = Xc[front_mask]
        Yc = Yc[front_mask]
        Zc = Zc[front_mask]
        rgb_ = rgb[front_mask]

        # project
        pts_cam = np.stack([Xc, Yc, Zc], axis=0)  # 3xN
        pts_im = self.K @ pts_cam
        u = pts_im[0] / pts_im[2]
        v = pts_im[1] / pts_im[2]

        ui = u.astype(np.int32)
        vi = v.astype(np.int32)

        in_bounds = (
            (ui >= 0) & (ui < self.ss_w) &
            (vi >= 0) & (vi < self.ss_h)
        )
        if not np.any(in_bounds):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        ui = ui[in_bounds]
        vi = vi[in_bounds]
        Zc = Zc[in_bounds]
        rgb_ = rgb_[in_bounds]

        # depth weighting (cheap atmospheric effect)
        z_scale = np.median(Zc)
        depth_weight = np.exp(-0.4 * (Zc / (z_scale + 1e-6)))  # (N,)
        rgb_ = np.asarray(rgb_, dtype=np.float32)
        if rgb_.ndim == 3 and rgb_.shape[-1] == 1:
            rgb_ = rgb_[..., 0]
        if rgb_.ndim != 2 or rgb_.shape[1] != 3:
            rgb_ = rgb_.reshape(-1, 3)

        colors = np.clip(rgb_ * depth_weight[:, None] * 1.6, 0.0, 1.0)

        # buffers
        img = np.zeros((self.ss_h, self.ss_w, 3), dtype=np.float32)
        depth = np.full((self.ss_h, self.ss_w), np.inf, dtype=np.float32)

        # --------- FAST vectorized Z-buffer ---------
        # 1) compute nearest depth per pixel
        np.minimum.at(depth, (vi, ui), Zc)

        # 2) keep only points whose depth == per-pixel nearest depth
        #    (Zc and depth[vi, ui] come from same values, so equality is OK)
        nearest_mask = (Zc == depth[vi, ui])
        ui_best = ui[nearest_mask]
        vi_best = vi[nearest_mask]
        colors_best = colors[nearest_mask]

        img[vi_best, ui_best] = colors_best
        # --------------------------------------------

        # no blur, no extra supersampling – very fast

        if self.ss > 1:
            img_small = img.reshape(
                self.height, self.ss,
                self.width, self.ss, 3
            ).mean(axis=(1, 3))
        else:
            img_small = img

        # background + gamma
        bg = 0.02
        img_small = np.clip(img_small + bg, 0.0, 1.0)
        img_small = img_small ** (1.0 / 2.2)
        img_uint8 = (img_small * 255).astype(np.uint8)
        return img_uint8

    def render_path(self, poses: List[Dict], outfile: str, fps: int = 30):
        out_path = os.path.join(self.out_dir, outfile)
        print(f"[Renderer] Writing video to: {out_path}")
        print(f"[Renderer] Total frames: {len(poses)}, FPS={fps}")

        writer = imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )

        for i, pose in enumerate(poses):
            if i % 50 == 0:
                print(f"[Renderer] Rendering frame {i+1}/{len(poses)}")
            frame = self.render_frame(pose)
            writer.append_data(frame)

        writer.close()
        print(f"[Renderer] Saved video: {out_path}")
