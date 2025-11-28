import numpy as np
import heapq
from collections import deque


def plan_panorama_path(
    scene,
    fps: int = 30,
    duration_sec: float = 70.0,
    d_min: float = 0.1,      # kept for compatibility
    tunnel_min: float = 0.1, # kept for compatibility
    radii=None,
    heights=None,
):
    """
    Panorama path planner for Gaussian indoor scene.

    Идея:
      - Работаем только в XZ (камера на фиксированной высоте).
      - Строим 2D occupancy grid по точкам стен / объектов.
      - Distanсe transform — получаем расстояние от каждой клетки до ближайшей стены.
      - Старт/финиш берём из самых "безопасных" (далёких от стен) клеток.
      - A* использует штраф за близость к стенам → путь идёт по центру коридоров.
      - Открытая кривая (а не бесконечный цикл), без самоперезапуска.
      - Если всё ломается → fallback эллиптической дугой (как прогулка по залу).
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

    # single horizontal plane
    interior_height = center[1]

    # чуть стягиваемся внутрь, чтобы не упираться в совсем внешние точки
    margin = 0.20 * base_r

    x_min_int = core_min[0] + margin
    x_max_int = core_max[0] - margin
    z_min_int = core_min[2] + margin
    z_max_int = core_max[2] - margin

    if x_max_int <= x_min_int:
        x_min_int = center[0] - 0.3 * base_r
        x_max_int = center[0] + 0.3 * base_r
    if z_max_int <= z_min_int:
        z_min_int = center[2] - 0.3 * base_r
        z_max_int = center[2] + 0.3 * base_r

    # ------------------------------------------------------------------
    # 1) Собираем "стены" / препятствия около высоты камеры
    # ------------------------------------------------------------------
    xyz = scene.xyz
    y = xyz[:, 1]

    height_band = 0.7 * (core_max[1] - core_min[1])
    if height_band < 0.5:
        height_band = 0.5
    mask = np.abs(y - interior_height) < height_band

    wall_points = xyz[mask][:, [0, 2]]  # XZ только

    if wall_points.shape[0] == 0:
        print("[Planner] No wall slice points → will use ellipse fallback.")
        wall_points = None

    # ------------------------------------------------------------------
    # 2) Occupancy grid + distance transform
    # ------------------------------------------------------------------
    def build_occupancy_and_distance(wall_pts, nx=140, nz=140):
        """
        Возвращает:
          occ       : (nz, nx) bool, True = занято (стена/объект + безопасный радиус)
          dist_m    : (nz, nx) float, расстояние до ближайшей стены в метрах
          x_lin, z_lin: мировые координаты центров ячеек
        """
        x_lin = np.linspace(x_min_int, x_max_int, nx)
        z_lin = np.linspace(z_min_int, z_max_int, nz)

        occ = np.zeros((nz, nx), dtype=bool)
        if wall_pts is not None and wall_pts.shape[0] > 0:
            # попадаем точки в ближайшие клетки
            x_idx = np.searchsorted(x_lin, np.clip(wall_pts[:, 0], x_min_int, x_max_int)) - 1
            z_idx = np.searchsorted(z_lin, np.clip(wall_pts[:, 1], z_min_int, z_max_int)) - 1
            x_idx = np.clip(x_idx, 0, nx - 1)
            z_idx = np.clip(z_idx, 0, nz - 1)
            occ[z_idx, x_idx] = True

        # физический размер клетки
        cell_size_x = (x_max_int - x_min_int) / max(1, nx - 1)
        cell_size_z = (z_max_int - z_min_int) / max(1, nz - 1)
        cell_size = float(min(cell_size_x, cell_size_z))

        # чуть надуваем препятствия, чтобы держаться подальше
        safety_margin = max(0.5, 0.25 * base_r)  # минимум ~0.5м
        steps = int(np.ceil(safety_margin / max(cell_size, 1e-6)))

        occ_dil = occ.copy()
        for _ in range(steps):
            o = occ_dil
            padded = np.pad(o, 1, mode="edge")
            nb = (
                padded[1:-1, 1:-1] |
                padded[:-2, 1:-1] |
                padded[2:, 1:-1] |
                padded[1:-1, :-2] |
                padded[1:-1, 2:] |
                padded[:-2, :-2] |
                padded[:-2, 2:] |
                padded[2:, :-2] |
                padded[2:, 2:]
            )
            occ_dil = nb

        # distance transform (манхэттен в клетках) – BFS от всех occupied
        H, W = occ_dil.shape
        dist_steps = np.full((H, W), np.inf, dtype=np.float32)
        q = deque()

        # источники – занятые клетки
        occ_any = np.where(occ_dil)
        for z, x in zip(occ_any[0], occ_any[1]):
            dist_steps[z, x] = 0.0
            q.append((z, x))

        # если вообще нет стен → всё свободно, расстояние бесконечное
        if len(q) == 0:
            dist_m = np.full((H, W), np.inf, dtype=np.float32)
            return occ_dil, dist_m, x_lin, z_lin

        # BFS 4-связный
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            z, x = q.popleft()
            d = dist_steps[z, x]
            for dz, dx in dirs:
                nz_, nx_ = z + dz, x + dx
                if 0 <= nz_ < H and 0 <= nx_ < W:
                    if dist_steps[nz_, nx_] > d + 1:
                        dist_steps[nz_, nx_] = d + 1
                        q.append((nz_, nx_))

        dist_m = dist_steps * cell_size
        return occ_dil, dist_m, x_lin, z_lin

    # ------------------------------------------------------------------
    # 3) A* с штрафом за близость к стенам
    # ------------------------------------------------------------------
    def astar_with_wall_cost(occ, dist_m, start, goal):
        """
        occ   : (H, W) bool, True = blocked
        dist_m: (H, W) float, расстояние до стены
        start, goal : (z, x) ints
        """
        H, W = occ.shape
        sz, sx = start
        gz, gx = goal

        if occ[sz, sx] or occ[gz, gx]:
            return []

        # 8-соседей
        nbrs = [
            (-1,  0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
            (1, -1, np.sqrt(2)),  (1, 1, np.sqrt(2)),
        ]

        def heuristic(z, x):
            # евклид до цели
            return np.hypot(z - gz, x - gx)

        # хотим избегать маленьких расстояний до стен:
        # penalty ~ 1/dist, но капим
        desired_clearance = max(0.7, 0.3 * base_r)  # хотим хотя бы ~0.7м от стен
        alpha = 4.0  # насколько сильно штрафуем близость к стенам

        def step_cost(z, x, base_step):
            d = dist_m[z, x]
            if not np.isfinite(d) or d <= 1e-6:
                return base_step * (1.0 + alpha)  # очень близко к стене
            # если d >= desired_clearance → штраф почти нулевой
            ratio = np.clip(d / desired_clearance, 0.0, 1.0)
            penalty = alpha * (1.0 - ratio)  # 0..alpha
            return base_step * (1.0 + penalty)

        open_heap = []
        heapq.heappush(open_heap, (heuristic(sz, sx), 0.0, (sz, sx)))
        came_from = {}
        g_score = {(sz, sx): 0.0}
        closed = set()

        while open_heap:
            _, g_cur, (z, x) = heapq.heappop(open_heap)
            if (z, x) in closed:
                continue
            closed.add((z, x))

            if (z, x) == (gz, gx):
                # восстановить путь
                path = [(z, x)]
                while (z, x) in came_from:
                    z, x = came_from[(z, x)]
                    path.append((z, x))
                path.reverse()
                return path

            for dz, dx, base_step in nbrs:
                nz_, nx_ = z + dz, x + dx
                if not (0 <= nz_ < H and 0 <= nx_ < W):
                    continue
                if occ[nz_, nx_]:
                    continue

                c_step = step_cost(nz_, nx_, base_step)
                tentative_g = g_cur + c_step
                if tentative_g < g_score.get((nz_, nx_), np.inf):
                    g_score[(nz_, nx_)] = tentative_g
                    came_from[(nz_, nx_)] = (z, x)
                    f = tentative_g + heuristic(nz_, nx_)
                    heapq.heappush(open_heap, (f, tentative_g, (nz_, nx_)))

        return []

    # ------------------------------------------------------------------
    # 4) Строим grid, выбираем безопасный старт/финиш, планируем A*
    # ------------------------------------------------------------------
    use_grid_path = False
    polyline_world = None
    mode_str = ""

    if wall_points is not None and wall_points.shape[0] > 0:
        occ, dist_m, x_lin, z_lin = build_occupancy_and_distance(wall_points, nx=150, nz=150)
        H, W = occ.shape

        # --- выбираем кандидатов по "наиболее далёким от стен" ---
        free_mask = ~occ
        if not np.any(free_mask):
            print("[Planner] Grid completely occupied → ellipse fallback.")
        else:
            d_free = dist_m.copy()
            d_free[~free_mask] = 0.0
            max_d = float(d_free.max())

            if max_d <= 1e-3:
                print("[Planner] All free cells too close to walls → ellipse fallback.")
            else:
                # кандидаты, у которых расстояние >= 0.7 * max_d
                good_mask = (d_free >= 0.7 * max_d) & free_mask
                good_indices = np.where(good_mask)
                if len(good_indices[0]) == 0:
                    # fallback: просто берём глобальный максимум
                    good_indices = np.where(d_free == max_d)

                # переводим их в список (z, x)
                candidates = list(zip(good_indices[0], good_indices[1]))

                # старт: кандидат, близкий к центру сцены, но далеко от стен
                cx = float(center[0])
                cz = float(center[2])

                def world_from_idx(z_idx, x_idx):
                    wx = x_lin[x_idx]
                    wz = z_lin[z_idx]
                    return wx, wz

                best_start = None
                best_score = np.inf
                for (z_idx, x_idx) in candidates:
                    wx, wz = world_from_idx(z_idx, x_idx)
                    d_center = (wx - cx) ** 2 + (wz - cz) ** 2
                    # хотим поближе к центру ⇒ минимизируем d_center
                    if d_center < best_score:
                        best_score = d_center
                        best_start = (z_idx, x_idx)

                # финиш: далёкий вдоль главной оси + далеко от стен
                best_end = None
                if best_start is not None:
                    # PCA по wall_points, чтобы найти "длинную" ось
                    mean_xz = wall_points.mean(axis=0)
                    centered = wall_points - mean_xz
                    cov = np.cov(centered.T)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    u_vec = eigvecs[:, np.argmax(eigvals)]  # major axis (2,)

                    scored = []
                    for (z_idx, x_idx) in candidates:
                        wx, wz = world_from_idx(z_idx, x_idx)
                        v = np.array([wx, wz], dtype=np.float32) - mean_xz
                        proj = float(v @ u_vec)
                        d_clear = dist_m[z_idx, x_idx]
                        # хотим одновременно далеко по оси и далеко от стен
                        score = proj + 0.5 * d_clear
                        scored.append((score, (z_idx, x_idx)))

                    if scored:
                        scored.sort()
                        # берём самый "дальний"
                        best_end = scored[-1][1]

                if best_start is not None and best_end is not None and best_start != best_end:
                    print(f"[Planner] A* planning: start={best_start}, end={best_end}")
                    path_cells = astar_with_wall_cost(occ, dist_m, best_start, best_end)
                    if len(path_cells) > 10:
                        pts = []
                        for (z_idx, x_idx) in path_cells:
                            wx = x_lin[x_idx]
                            wz = z_lin[z_idx]
                            pts.append([wx, wz])
                        polyline_world = np.asarray(pts, dtype=np.float32)
                        use_grid_path = True
                        mode_str = "grid A* skeleton path"
                        print(f"[Planner] A* path found with {polyline_world.shape[0]} points.")
                    else:
                        print("[Planner] A* path too short or failed → ellipse fallback.")
                else:
                    print("[Planner] Could not choose valid start/end → ellipse fallback.")

        # небольшой принт: расстояние старта до стен
        if use_grid_path and polyline_world is not None:
            start_xz = polyline_world[0]
            if wall_points is not None and wall_points.shape[0] > 0:
                diff = wall_points - start_xz[None, :]
                d = np.sqrt(np.sum(diff ** 2, axis=1)).min()
                print(f"[Planner] Start world position {start_xz}, nearest wall ≈ {d:.3f} m")

    # ------------------------------------------------------------------
    # 5) Сглаживаем A* путь и параметризуем по длине
    # ------------------------------------------------------------------
    def smooth_polyline(pts: np.ndarray, passes: int = 4) -> np.ndarray:
        if pts is None or pts.shape[0] < 5:
            return pts
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
        kernel = kernel / kernel.sum()
        k2 = len(kernel) // 2

        sm = pts.copy()
        for _ in range(passes):
            x = sm[:, 0]
            z = sm[:, 1]
            x_pad = np.pad(x, (k2, k2), mode="edge")
            z_pad = np.pad(z, (k2, k2), mode="edge")
            x_s = np.convolve(x_pad, kernel, mode="same")[k2:-k2]
            z_s = np.convolve(z_pad, kernel, mode="same")[k2:-k2]
            sm = np.stack([x_s, z_s], axis=1)
        return sm.astype(np.float32)

    if use_grid_path and polyline_world is not None:
        polyline_world = smooth_polyline(polyline_world, passes=4)

        diffs = polyline_world[1:] - polyline_world[:-1]
        seglen = np.linalg.norm(diffs, axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(seglen)])
        total_len = float(cumlen[-1] + 1e-6)

        def sample_pos_and_forward(t_norm: float):
            # t_norm ∈ [0,1] → идём один раз по открытой траектории
            s = np.clip(t_norm, 0.0, 1.0) * total_len
            i0 = int(np.searchsorted(cumlen, s, side="right") - 1)
            i0 = max(0, min(i0, len(seglen) - 1))
            i1 = i0 + 1

            seg_len = seglen[i0] + 1e-6
            s0 = cumlen[i0]
            local_t = (s - s0) / seg_len

            p0 = polyline_world[i0]
            p1 = polyline_world[i1]
            pos_xz = (1.0 - local_t) * p0 + local_t * p1

            # направление – усреднённый тангент из соседей → медленные повороты
            i_prev = max(0, i0 - 4)
            i_next = min(len(polyline_world) - 1, i1 + 4)
            tangent = polyline_world[i_next] - polyline_world[i_prev]
            norm = np.linalg.norm(tangent) + 1e-6
            dir_xz = tangent / norm
            return pos_xz, dir_xz

    else:
        # ------------------------------------------------------------------
        # 6) Fallback: гладкая эллиптическая дуга (не полный круг)
        # ------------------------------------------------------------------
        half_width = 0.5 * (x_max_int - x_min_int)
        half_depth = 0.5 * (z_max_int - z_min_int)

        if half_width >= half_depth:
            a = 0.85 * half_width
            b = 0.35 * half_depth
        else:
            a = 0.35 * half_width
            b = 0.85 * half_depth

        if a < 0.1 * base_r:
            a = 0.3 * base_r
        if b < 0.1 * base_r:
            b = 0.2 * base_r

        # "прогулка" от -135° до +135° (а не круг)
        start_angle = -0.75 * np.pi
        end_angle = 0.75 * np.pi

        def sample_pos_and_forward(t_norm: float):
            theta = start_angle + (end_angle - start_angle) * t_norm
            x = center[0] + a * np.cos(theta)
            z = center[2] + b * np.sin(theta)

            dx_dtheta = -a * np.sin(theta)
            dz_dtheta = b * np.cos(theta)
            tangent = np.array([dx_dtheta, dz_dtheta], dtype=np.float32)
            norm = np.linalg.norm(tangent) + 1e-6
            dir_xz = tangent / norm

            return np.array([x, z], dtype=np.float32), dir_xz

        mode_str = "fallback ellipse arc"

    # ------------------------------------------------------------------
    # 7) Строим позы вдоль траектории
    # ------------------------------------------------------------------
    look_ahead_dist = 0.25 * base_r

    for idx in range(total_frames):
        t = idx / max(1, total_frames - 1)
        xz, dir_xz = sample_pos_and_forward(t)
        x, z = float(xz[0]), float(xz[1])
        y = interior_height

        pos = np.array([x, y, z], dtype=np.float32)
        forward3 = np.array([dir_xz[0], 0.0, dir_xz[1]], dtype=np.float32)
        target = pos + forward3 * look_ahead_dist

        poses.append(
            {
                "pos": pos.astype(np.float32),
                "target": target.astype(np.float32),
                "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            }
        )

    print(
        f"[Planner] Indoor path ({mode_str or 'grid A* skeleton path'}): "
        f"{len(poses)} frames, {duration_sec:.1f}s @{fps} fps"
    )
    return poses


def plan_object_tour(*args, **kwargs):
    """
    Placeholder for the object-focused cinematic tour (Video 2).
    """
    return []
