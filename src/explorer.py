# src/explorer.py
import numpy as np
import open3d as o3d


class Scene:
    """
    Scene wrapper for a Gaussian Splatting PLY:

    - loads xyz from 'positions' (tensor API)
    - recovers RGB either from:
        * 'colors' attribute, or
        * spherical harmonic DC coeffs f_dc_0..2 (common in 3DGS PLYs)
    - computes:
        * global bbox / center / radius
        * interior "core" bbox / center / radius (for indoor path planning)
    """

    def __init__(self, ply_path: str, max_points: int = 800_000):
        print(f"[Scene] Loading PLY (Gaussian) from: {ply_path}")

        # ---- tensor API keeps all custom attributes ----
        pcd_t = o3d.t.io.read_point_cloud(ply_path)

        # TensorMap is iterable but has no .keys(), so we just list it
        attr_keys = list(pcd_t.point)
        print(f"        Available point attributes: {attr_keys}")

        # ---------- positions ----------
        if "positions" in attr_keys:
            xyz = pcd_t.point["positions"].numpy().astype(np.float32)
        else:
            # fallback to legacy reader if needed
            print("        WARNING: 'positions' not found, falling back to legacy read_point_cloud")
            pcd_legacy = o3d.io.read_point_cloud(ply_path)
            xyz = np.asarray(pcd_legacy.points, dtype=np.float32)

        if xyz.shape[0] == 0:
            raise RuntimeError("Loaded PLY has 0 points – check the file (maybe still compressed?).")

        # ---------- colors ----------
        rgb = None

        # 1) Direct 'colors' attribute
        if "colors" in attr_keys:
            c = pcd_t.point["colors"].numpy().astype(np.float32)
            # normalize if in 0–255
            if c.max() > 2.0:
                c = c / 255.0
            rgb = c
            print("        Using 'colors' attribute for RGB.")

        # 2) Otherwise try SH DC coefficients (typical 3DGS format)
        elif all(k in attr_keys for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
            print("        Using spherical harmonic DC coeffs f_dc_0..2 for RGB.")
            dc0 = pcd_t.point["f_dc_0"].numpy().astype(np.float32)
            dc1 = pcd_t.point["f_dc_1"].numpy().astype(np.float32)
            dc2 = pcd_t.point["f_dc_2"].numpy().astype(np.float32)

            # Standard DC → RGB conversion (similar to 3DGS)
            SH_C0 = 0.28209479177387814  # 1 / (2*sqrt(pi))
            dc = np.stack([dc0, dc1, dc2], axis=1)  # (N,3)
            rgb = 0.5 + SH_C0 * dc
            rgb = np.clip(rgb, 0.0, 1.0)

        # 3) Fallback if nothing color-related exists
        else:
            print("        No 'colors' or 'f_dc_*' found – falling back to neutral gray.")
            rgb = np.full((xyz.shape[0], 3), 0.7, dtype=np.float32)

        print(f"        Raw points: {xyz.shape[0]}")

        # ---------- optional subsampling ----------
        n = xyz.shape[0]
        if n > max_points:
            idx = np.random.choice(n, max_points, replace=False)
            xyz = xyz[idx]
            rgb = rgb[idx]
            print(f"        Subsampled to: {xyz.shape[0]} points")

        self.xyz = xyz
        self.rgb = rgb

        # ---------- global scene stats ----------
        self.bbox_min = xyz.min(axis=0)
        self.bbox_max = xyz.max(axis=0)
        self.center = 0.5 * (self.bbox_min + self.bbox_max)

        extents = self.bbox_max - self.bbox_min
        horiz_extent = np.linalg.norm([extents[0], extents[2]])  # XZ size
        self.radius = horiz_extent * 0.7

        print("        Bounding box min:", self.bbox_min)
        print("        Bounding box max:", self.bbox_max)
        print("        Center:", self.center)
        print("        Approx radius:", self.radius)

        # ---------- interior / "core" region for indoor navigation ----------
        # We ignore extreme outliers and only look at the central 80% of points.
        try:
            x10, x90 = np.percentile(xyz[:, 0], [10, 90])
            y10, y90 = np.percentile(xyz[:, 1], [10, 90])
            z10, z90 = np.percentile(xyz[:, 2], [10, 90])

            self.core_min = np.array([x10, y10, z10], dtype=np.float32)
            self.core_max = np.array([x90, y90, z90], dtype=np.float32)
            self.core_center = 0.5 * (self.core_min + self.core_max)

            core_extents = self.core_max - self.core_min
            # only horizontal (x, z) size for radius
            self.core_radius = 0.5 * np.linalg.norm([core_extents[0], core_extents[2]])

            print("        [Scene] Using interior 'core' stats for path planning:")
            print("            core_min:", self.core_min)
            print("            core_max:", self.core_max)
            print("            core_center:", self.core_center)
            print("            core_radius:", self.core_radius)
        except Exception as e:
            print("        [Scene] Could not compute core bbox, using global center/radius.", e)
            self.core_min = self.bbox_min
            self.core_max = self.bbox_max
            self.core_center = self.center
            self.core_radius = self.radius
