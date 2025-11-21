import numpy as np


def plan_panorama_path(
    scene,
    fps: int = 30,
    duration_sec: float = 70.0,
    d_min: float = 0.1,
    tunnel_min: float = 0.1,
    radii=None,
    heights=None,
):
    """
    Plan a cinematic *indoor* panorama path for a Gaussian scene.

    Improvements vs old version:
      - Still a smooth ellipse in XZ.
      - BUT we now look at the point cloud near camera height and
        *shrink the ellipse* so it stays at least `safety_margin`
        away from dense geometry (walls / obstacles).
    """

    # --- interior stats from Scene ---
    center = getattr(scene, "core_center", scene.center).astype(np.float32)
    base_r = float(getattr(scene, "core_radius", scene.radius))
    if base_r < 1e-3:
        base_r = float(scene.radius)

    core_min = getattr(scene, "core_min", scene.bbox_min).astype(np.float32)
    core_max = getattr(scene, "core_max", scene.bbox_max).astype(np.float32)

    total_frames = int(duration_sec * fps)
    poses = []

    # constant camera height (no vertical bobbing)
    interior_height = center[1]

    # how far from walls we stay (bigger margin → safer, but more central)
    margin = 0.18 * base_r

    # interior extents (rough room box)
    x_min_int = core_min[0] + margin
    x_max_int = core_max[0] - margin
    z_min_int = core_min[2] + margin
    z_max_int = core_max[2] - margin

    # if something is weird, fall back to a small ellipse around center
    if x_max_int <= x_min_int:
        x_min_int = center[0] - 0.3 * base_r
        x_max_int = center[0] + 0.3 * base_r
    if z_max_int <= z_min_int:
        z_min_int = center[2] - 0.3 * base_r
        z_max_int = center[2] + 0.3 * base_r

    # base ellipse radii (in X and Z) – slightly inside the interior box
    half_width = 0.5 * (x_max_int - x_min_int)
    half_depth = 0.5 * (z_max_int - z_min_int)

    # shrink a bit more so we don't graze walls / small obstacles
    a_base = 0.8 * half_width   # radius along X
    b_base = 0.8 * half_depth   # radius along Z

    # if for some reason they are tiny, use a fraction of base_r
    if a_base < 0.1 * base_r:
        a_base = 0.3 * base_r
    if b_base < 0.1 * base_r:
        b_base = 0.2 * base_r

    # -----------------------------
    # Build a crude "wall slice" in XZ from the point cloud
    # -----------------------------
    xyz = scene.xyz  # (N,3)
    y = xyz[:, 1]

    # take points near the camera height (e.g. ±0.7 m)
    # you can tweak the band if needed
    height_band = 0.7 * (core_max[1] - core_min[1])  # somewhat scene-relative
    if height_band < 0.5:
        height_band = 0.5
    mask = np.abs(y - interior_height) < height_band

    wall_points = xyz[mask][:, [0, 2]]  # XZ only

    if wall_points.shape[0] == 0:
        # no info — fall back to old behavior
        print("[Planner] WARNING: no wall slice points – using plain ellipse.")
        wall_points = None
    else:
        # subsample to keep it cheap
        max_wall = 8000
        if wall_points.shape[0] > max_wall:
            idx = np.random.choice(wall_points.shape[0], max_wall, replace=False)
            wall_points = wall_points[idx]

        print(f"[Planner] Using {wall_points.shape[0]} points as wall slice for safety check.")

    # safety distance from "walls"
    safety_margin = 0.20 * base_r  # ~20% of core radius; tweak if needed
    safety_margin = max(safety_margin, 0.25)  # at least 25 cm

    # -----------------------------
    # Find safest scale factor for ellipse radii
    # -----------------------------
    def min_wall_distance_for_scale(scale: float) -> float:
        """
        For a given scale in (0,1], compute the *minimum* distance from
        any point along the ellipse to the nearest "wall point".
        """
        if wall_points is None:
            return np.inf

        a = a_base * scale
        b = b_base * scale

        # sample the ellipse at many angles
        angles = np.linspace(0.0, 2.0 * np.pi, num=180, endpoint=False)
        xs = center[0] + a * np.cos(angles)
        zs = center[2] + b * np.sin(angles)
        ellipse_pts = np.stack([xs, zs], axis=1)  # (M,2)

        # compute distances from ellipse points to all wall points
        # shape: (M, N, 2)
        diff = ellipse_pts[:, None, :] - wall_points[None, :, :]
        dist2 = np.sum(diff ** 2, axis=-1)  # (M, N)
        # nearest wall distance for each ellipse sample
        min_dist_per_sample = np.sqrt(np.min(dist2, axis=1))  # (M,)
        # global minimum distance
        return float(np.min(min_dist_per_sample))

    best_scale = 1.0
    if wall_points is not None:
        # binary search over scale in [min_scale, 1.0]
        min_scale = 0.25  # don't shrink below this
        low, high = min_scale, 1.0
        best_scale = min_scale

        for _ in range(10):  # 10 iterations is enough
            mid = 0.5 * (low + high)
            d = min_wall_distance_for_scale(mid)
            # if we are safely away from walls, we can grow
            if d >= safety_margin:
                best_scale = mid
                low = mid
            else:
                high = mid

        print(
            f"[Planner] Chosen ellipse scale={best_scale:.3f}, "
            f"min distance to walls≈{min_wall_distance_for_scale(best_scale):.3f} m"
        )
    else:
        print("[Planner] No wall info, using scale=1.0")

    a = a_base * best_scale
    b = b_base * best_scale

    # how far ahead we look for the camera target
    look_ahead_dist = 0.25 * base_r  # smaller → smoother, slower apparent turning

    def sample_pos(t_norm: float) -> np.ndarray:
        """
        t_norm in [0, 1] → point on ellipse in XZ plane around 'center'.
        Smooth loop, no zigzags.
        """
        t_norm = (t_norm % 1.0)
        angle = 2.0 * np.pi * t_norm  # one full loop over [0,1]
        x = center[0] + a * np.cos(angle)
        z = center[2] + b * np.sin(angle)
        return np.array([x, z], dtype=np.float32)

    # -----------------------------
    # Build poses along the ellipse
    # -----------------------------
    for idx in range(total_frames):
        t = idx / max(1, total_frames - 1)  # [0, 1]

        # current position on ellipse
        xz = sample_pos(t)
        x, z = xz[0], xz[1]
        y = interior_height

        pos = np.array([x, y, z], dtype=np.float32)

        # ---- forward-look target (camera always faces forward) ----
        dt = 1.0 / total_frames
        t_f = t + 4.0 * dt  # a bit ahead along the loop
        xz_f = sample_pos(t_f)
        x_f, z_f = xz_f[0], xz_f[1]
        y_f = interior_height  # same plane

        forward = np.array([x_f - x, y_f - y, z_f - z], dtype=np.float32)
        norm = np.linalg.norm(forward) + 1e-6
        forward /= norm

        target = pos + forward * look_ahead_dist

        poses.append(
            {
                "pos": pos.astype(np.float32),
                "target": target.astype(np.float32),
                "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            }
        )

    print(f"[Planner] Indoor elliptical path: {len(poses)} frames, {duration_sec:.1f}s @{fps} fps")
    return poses


def plan_object_tour(*args, **kwargs):
    """
    Placeholder for the object-focused cinematic tour (Video 2).
    """
    return []
