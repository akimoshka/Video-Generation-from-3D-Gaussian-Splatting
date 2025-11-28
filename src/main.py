# src/main.py
import argparse
import yaml
import json
import os

from explorer import Scene
from path_planner import plan_panorama_path, plan_object_tour
from renderer import Renderer


def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def export_path_to_json(poses, out_path: str):
    """Save camera path as JSON for Spark / Three.js."""
    serializable = []
    for p in poses:
        serializable.append(
            {
                "pos": p["pos"].tolist(),
                "target": p["target"].tolist(),
                "up": p["up"].tolist(),
            }
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[Main] Saved camera path to {out_path}")


def build_scene_and_renderer(cfg):
    scene_cfg = cfg["scene"]
    rend_cfg = cfg["renderer"]

    scene = Scene(
        scene_cfg["ply_path"],
        max_points=scene_cfg.get("max_points", 800_000),
    )
    renderer = Renderer(
        scene,
        rend_cfg["out_dir"],
        width=rend_cfg["width"],
        height=rend_cfg["height"],
        fov_deg=rend_cfg["fov_deg"],
    )
    return scene, renderer


def run_panorama(cfg, export_only: bool = False):
    scene, renderer = build_scene_and_renderer(cfg)
    plan_cfg = cfg["planner"]

    poses = plan_panorama_path(
        scene,
        fps=plan_cfg["fps"],
        duration_sec=plan_cfg["panorama_duration"],
    )

    spark_dir = cfg.get("spark_viewer_dir", "spark_viewer")
    spark_json_path = os.path.join(spark_dir, "panorama_path.json")
    export_path_to_json(poses, spark_json_path)

    out_dir = cfg["renderer"]["out_dir"]
    export_path_to_json(poses, os.path.join(out_dir, "panorama_path.json"))

    if not export_only:
        renderer.render_path(
            poses,
            "GaussianMuseumeTour.mp4",
            fps=plan_cfg["fps"],
        )


def run_object_tour(cfg):
    scene, renderer = build_scene_and_renderer(cfg)
    plan_cfg = cfg["planner"]
    poses = plan_object_tour(scene, plan_cfg)

    spark_dir = cfg.get("spark_viewer_dir", "spark_viewer")
    export_path_to_json(
        poses,
        os.path.join(spark_dir, "object_tour_path.json")
    )

    renderer.render_path(poses, "object_tour.mp4", fps=plan_cfg["fps"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["panorama", "objects"], default="panorama")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export path JSON for Spark rendering",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "panorama":
        run_panorama(cfg, export_only=args.export_only)
    else:
        run_object_tour(cfg)


if __name__ == "__main__":
    main()
