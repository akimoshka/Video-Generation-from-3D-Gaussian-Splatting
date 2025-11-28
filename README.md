# Assignment 4 – Cinematic Navigation in Gaussian Splatting Scene

### Gaussian Museum Tour (Python Renderer)

![GaussianMuseumeTour](vids/GaussianMuseumeTour.gif)  

---

### Realistic Museum Tour (Spark / WebGL Renderer)

![RealisticMuseumeTour](vids/RealisticMuseumeTour.gif)

This project implements an end-to-end pipeline for **cinematic navigation** inside a 3D Gaussian Splatting scene:

- Load a Gaussian PLY into Python (`open3d`) and into a web viewer (Spark + Three.js).
- Plan a **smooth indoor camera path** that avoids walls and obstacles.
- Render a **Gaussian point-based video** (`GaussianMuseumeTour.mp4`) in Python.
- Export the same camera path to JSON and replay it as a **realistic/360-style tour** in the browser using Spark.
- Provide a **real-time preview** of the scene and the camera motion in the browser.

---

## Completed Tasks (Scoring Checklist)

1. ✅ **Render a video from inside the scene**  
   – Implemented in python as `GaussianMuseumeTour.mp4` via `Renderer.render_path`.

4. ✅ **Path planning**  
   – Implemented in `src/path_planner.py` using a 2D occupancy grid and A* with distance transform.

5. ✅ **Obstacle avoidance**  
   – Camera navigates around walls/objects using inflated occupancy grid and a clearance-aware cost.

6. ✅ **Rendered video that covers most of the scene/area**  
   – Planned path traverses the interior “core” of the museum to cover major areas.

7. ✅ **Render a 360° video**  
   – The Spark viewer replays the full tour along the path for 90 seconds, giving a panoramic museum tour.

9. ✅ **Real-time preview of the scene or pipeline**  
   – The `spark_viewer/index.html` loads the splat and the camera path and renders the tour in real time in the browser.

10. ✅ **Artistic / professional / innovative result videos**  
   - The point renderer uses depth-based fading and a smooth, corridor-centric camera path to produce a clean, cinematic museum tour.  
   - A more realistic 360° visualization is produced via Spark (link can be provided in the report).

> **Note:** Object detection (Tasks 2 and 3) is **not implemented**. The file `src/detector.py` is a placeholder only.

---

## 1. Installation Instructions

### 1.1. Requirements

- Python **3.10+** (tested with 3.10 / 3.11)
- Recommended: virtual environment (`venv` or `conda`)
- Libraries (also listed in `requirements.txt`):
  - `numpy`
  - `pyyaml`
  - `imageio`
  - `open3d`
  - (optionally) `tqdm` for progress bars  
- A modern browser (Chrome/Edge/Firefox) for the **Spark + Three.js** viewer.

Install dependencies:

```bash
pip install -r requirements.txt
````
---

## 2. Project Structure

```text
project/
  README.md
  Report.pdf
  requirements.txt

  configs/
    config.yaml

  src/
    main.py
    explorer.py
    path_planner.py
    renderer.py
    detector.py   # placeholder, object detection NOT implemented

  input-data/
    Museume.ply   # Gaussian Splatting scene used in this submission, you do need to add it here, it was too big to import into repository

  outputs/
    scene_1/
      GaussianMuseumeTour.mp4
      panorama_path.json
      RealisticMuseumeTour.webm # added after recording the spark render

  spark_viewer/
    index.html
    Museume.ply
    panorama_path.json      # copy from outputs/scene_1
```

---

## 3. Usage Guide

### 3.1. Configure the Scene

Configuration is defined in `configs/config.yaml`:

```yaml
scene:
  ply_path: "input-data/Museume.ply"
  max_points: 800000

renderer:
  out_dir: "outputs/scene_1"
  width: 960
  height: 544
  fov_deg: 70.0

planner:
  fps: 30
  panorama_duration: 90
  d_min: 0.1
  tunnel_min: 0.1
  panorama_radii: [0.8, 0.6, 0.4]
  panorama_heights: [0.0, 0.1, -0.05]

spark_viewer_dir: "spark_viewer"
```

Update `ply_path` if you want to test a different scene.

---

### 3.2. Render Gaussian Video from Python

From the project root:

```bash
python src/main.py --mode panorama
```

This will:

1. Load the Gaussian scene from `input-data/Museume.ply`.
2. Plan a smooth indoor trajectory that avoids walls.
3. Export the camera path to:

   * `spark_viewer/panorama_path.json` (for the web viewer)
   * `outputs/scene_1/panorama_path.json`
4. Render the video:

   * `outputs/scene_1/GaussianMuseumeTour.mp4`

---

### 3.3. Realistic / 360° Tour with Spark + Three.js

The folder `spark_viewer/` contains a minimal web viewer that uses:

* `@sparkjsdev/spark` to render the Gaussian PLY in the browser.
* The exported `panorama_path.json` to animate the camera.

**Steps:**

1. Ensure these files exist inside `spark_viewer/`:

   * `Museume.ply`
   * `panorama_path.json`
   * `index.html`

   ### ! You do need to add the `Museume.ply` into the folder, the file was too big to put into the repository

2. Run a simple HTTP server in `spark_viewer`:

   ```bash
   cd spark_viewer
   python -m http.server 8000
   ```

3. Open in browser:

   ```text
   http://localhost:8000/index.html
   ```

The viewer will:

* Load `Museume.ply` and the camera path.
* Start a **90-second museum tour** along the same path as in the Python renderer.
* Record the tour using `MediaRecorder` and automatically download a video file
  (e.g. `RealisticMuseumeTour.mp4` or `.webm` depending on browser support).

This can be used as:

* A 360°-style tour for the assignment.
* A **real-time preview** of the scene and the navigation pipeline.

---

## 4. Algorithm Descriptions

### 4.1. Scene Representation (`src/explorer.py`)

* Loads the Gaussian Splatting PLY with `open3d.t.io.read_point_cloud`.
* Reads:

  * `positions` → XYZ coordinates of points.
  * Either:

    * `colors` attribute, or
    * `f_dc_0`, `f_dc_1`, `f_dc_2` (spherical harmonic DC coefficients) converted to RGB.
* Computes:

  * Global bounding box, center, and an approximate scene radius.
  * An interior **core region** (10–90% percentiles along each axis) used for indoor planning, ignoring extreme outliers.

This provides `scene.xyz`, `scene.rgb`, `scene.core_min`, `scene.core_max`, `scene.core_center`, and `scene.core_radius`.

---

### 4.2. Path Planning & Obstacle Avoidance (`src/path_planner.py`)

Core idea: work in the **XZ plane** at a fixed camera height and treat the problem as **2D navigation** in a discretized floor plan.

1. **Extract wall / obstacle points**:

   * Select all points whose height is close to the interior height (a vertical band around `core_center[1]`).
   * Project them to XZ plane.

2. **Build occupancy grid and distance transform**:

   * Discretize the interior XZ region into a grid.
   * Mark grid cells containing wall/obstacle points as occupied.
   * Dilate occupied cells by a **safety margin** so that we keep the camera away from walls.
   * Compute a **distance transform** (via BFS) to get, for each free cell, the distance to the nearest wall in meters.

3. **Start / end selection**:

   * Consider “good” cells that are **far from walls** (distance ≥ 0.7 × max distance).
   * Choose the **start** as the good cell closest to the scene center.
   * Choose the **end** along the dominant axis of the scene (via 2D PCA on wall points) while still preferring large clearance.

4. **A* with clearance-aware cost**:

   * Standard 8-connected grid neighbors (4 cardinal + 4 diagonals).
   * Base step cost = 1 or √2, plus a **penalty** that increases when the distance to the nearest wall is small:

     * `penalty ≈ alpha * (1 - dist / desired_clearance)`, clipped to [0, alpha].
   * This encourages paths that run along the “skeleton” of corridors, not hugging walls.

5. **Smoothing & arc-length parameterization**:

   * The raw A* path is smoothed with a small 1D kernel multiple times to remove jagged corners.
   * We compute cumulative arc-length and resample along it to match the desired number of frames.
   * The camera forward direction is computed from a **wider neighborhood** along the path to produce slow, cinematic turns.

6. **Fallback ellipse**:

   * If the grid is degenerate (no walls / no valid path), we fall back to an **open elliptical arc** around the scene center (from -135° to +135°), which still avoids extremes of the bounding box.

This fulfills both **Path planning** (Task 4) and **Obstacle avoidance** (Task 5).

---

### 4.3. Rendering (`src/renderer.py`)

* Builds a camera extrinsic matrix from `pos`, `target`, and `up`.
* Projects all points in front of the camera into image space with a pinhole intrinsics matrix `K`.
* Applies a **vectorized Z-buffer**:

  * For each pixel, find the nearest depth and keep only points that match this depth.
* Adds a **depth-based fade** (`exp(-0.4 * Z / median_Z)`) to simulate atmospheric perspective and reduce clutter.
* Gamma-corrects the result and writes frames into an `MP4` video using `imageio`.

Result: `GaussianMuseumeTour.mp4`.

---

### 4.4. Spark Viewer (Realistic / 360° Tour)

* Loads the same PLY and camera path in the browser with `SplatMesh` from `@sparkjsdev/spark`.
* Replays the path in **real time** for 90 seconds.
* Uses `captureStream` + `MediaRecorder` to save the tour into a video in the browser.
* Provides on-screen debug info about indices and positions for inspection.

---

## 5. Dependencies and Requirements

Summarized:

* Python: 3.10+
* Python packages:

  * `numpy`, `pyyaml`, `imageio`, `open3d`
* Browser:

  * Modern browser with ES modules and `MediaRecorder` support.
* Files:

  * A 3D Gaussian Splatting PLY (e.g., `Museume.ply`).

---

## 6. Known Limitations

* **No object detection**:

  * `src/detector.py` is a placeholder; YOLO-based 2D/3D detection is not implemented.
  * Tasks 2 and 3 from the bonus checklist are not addressed.


* **Indoor bias**:

  * Path planner is tuned for indoor-style scenes (corridors, rooms).
  * Outdoor or extremely sparse scenes may fall back to the elliptical arc more often.

* **Single semantic style**:

  * Camera style is “smooth museum walk”; no alternative styles (handheld, drone, etc.) are implemented.

For more discussion, results, and future work ideas, please refer to the **Technical Report (Report.pdf)**.