"""
RANSAC-based geometric fitting for noisy CAD/freehand sketch datasets.

Fits multiple straight-line models and circle models to Canny edge pixels.
It does NOT use Hough Lines or Hough Circles.

Before running:
1. Set DATASET_ROOT to your dataset directory.
2. Install dependencies:
   pip install opencv-python numpy pandas
3. Run:
   python ransac_cad_sketch_fit.py

Outputs:
- ransac_cad_results/visualizations/<image>_ransac.png
- ransac_cad_results/ransac_fit_summary.csv

Tip: Start with MAX_IMAGES = 30 to check the results, then set it to None
to process every image. The script skips common mask/label folders by default.
"""

from pathlib import Path
import math
import random
import cv2
import numpy as np
import pandas as pd


# --------------------------- CONFIGURATION ---------------------------
DATASET_ROOT = Path(r"C:\flutter_projects\cad\ortho_sketch_dataset_4000")  # CHANGE THIS
OUTPUT_ROOT = Path(r"C:\flutter_projects\cad\analysis")

# Set to 30 for a quick test; set to None to process all discovered images.
MAX_IMAGES = 30

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SKIP_FOLDER_NAMES = {
    "mask", "masks", "label", "labels", "annotations", "annotation",
    "visual_reports", "ransac_results", "ransac_cad_results"
}

# Edge extraction
CANNY_LOW = 40
CANNY_HIGH = 130
BLUR_KERNEL = 3

# RANSAC tuning (pixel units)
LINE_DISTANCE_THRESHOLD = 2.0
CIRCLE_DISTANCE_THRESHOLD = 2.5
MIN_LINE_INLIERS = 25
MIN_CIRCLE_INLIERS = 30
MAX_LINE_MODELS = 12
MAX_CIRCLE_MODELS = 8
LINE_TRIALS = 500
CIRCLE_TRIALS = 700
MAX_EDGE_POINTS = 3500
RANDOM_SEED = 7
# --------------------------------------------------------------------


def list_images(root: Path):
    """Recursively find image files while ignoring common mask/label folders."""
    found = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(part.lower() in SKIP_FOLDER_NAMES for part in p.parts):
            continue
        found.append(p)
    return sorted(found)


def get_edge_points(image_path: Path):
    """Read an image, binarize it, and return Canny edge points as (x, y)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if BLUR_KERNEL and BLUR_KERNEL >= 3:
        k = BLUR_KERNEL if BLUR_KERNEL % 2 else BLUR_KERNEL + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # Otsu threshold is useful for sketches with varying background brightness.
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    edges = cv2.Canny(binary, CANNY_LOW, CANNY_HIGH)

    ys, xs = np.where(edges > 0)
    points = np.column_stack((xs, ys)).astype(np.float32)

    # Bound runtime for high-resolution images while retaining spatial coverage.
    if len(points) > MAX_EDGE_POINTS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(points), MAX_EDGE_POINTS, replace=False)
        points = points[idx]

    return image, edges, points


def fit_line_from_two_points(p1, p2):
    """Return normalized ax + by + c = 0 coefficients, or None if degenerate."""
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    # Normal vector to direction (dx, dy)
    a = -dy / norm
    b = dx / norm
    c = -(a * float(p1[0]) + b * float(p1[1]))
    return np.array([a, b, c], dtype=np.float64)


def line_residuals(points, model):
    return np.abs(points @ model[:2] + model[2])  # a,b are normalized


def ransac_multiple_lines(points, rng):
    """
    Iteratively find line models with RANSAC and remove their inliers.
    Returns dictionaries containing endpoints, angle, slope and inlier count.
    """
    remaining = points.copy()
    models = []

    for _ in range(MAX_LINE_MODELS):
        if len(remaining) < MIN_LINE_INLIERS:
            break

        best_model, best_mask = None, None
        best_count = 0

        for _trial in range(LINE_TRIALS):
            ids = rng.choice(len(remaining), 2, replace=False)
            model = fit_line_from_two_points(remaining[ids[0]], remaining[ids[1]])
            if model is None:
                continue

            mask = line_residuals(remaining, model) <= LINE_DISTANCE_THRESHOLD
            count = int(mask.sum())
            if count > best_count:
                best_model, best_mask, best_count = model, mask, count

        if best_model is None or best_count < MIN_LINE_INLIERS:
            break

        inliers = remaining[best_mask]

        # Refine the line by PCA over its inliers.
        center = inliers.mean(axis=0)
        _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
        direction = vh[0]
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        # Project inliers to obtain endpoints along the fitted line.
        projections = (inliers - center) @ direction
        p_start = center + projections.min() * direction
        p_end = center + projections.max() * direction

        dx = float(p_end[0] - p_start[0])
        dy = float(p_end[1] - p_start[1])
        angle = math.degrees(math.atan2(dy, dx))
        if abs(dx) < 1e-6:
            slope = None  # vertical line: slope is undefined
        else:
            slope = dy / dx

        models.append({
            "p1": tuple(np.round(p_start).astype(int)),
            "p2": tuple(np.round(p_end).astype(int)),
            "inliers": int(best_count),
            "angle_deg": angle,
            "slope": slope,
        })

        remaining = remaining[~best_mask]

    return models


def circle_from_three_points(p1, p2, p3):
    """Calculate circle centre and radius from 3 non-collinear points."""
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    x3, y3 = map(float, p3)

    denom = 2.0 * (
        x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    )
    if abs(denom) < 1e-6:
        return None

    s1 = x1*x1 + y1*y1
    s2 = x2*x2 + y2*y2
    s3 = x3*x3 + y3*y3

    ux = (s1*(y2-y3) + s2*(y3-y1) + s3*(y1-y2)) / denom
    uy = (s1*(x3-x2) + s2*(x1-x3) + s3*(x2-x1)) / denom
    radius = math.hypot(x1 - ux, y1 - uy)

    if not np.isfinite([ux, uy, radius]).all():
        return None
    if radius < 5 or radius > 0.75 * max(1, min(IMAGE_WIDTH, IMAGE_HEIGHT)):
        return None
    return np.array([ux, uy, radius], dtype=np.float64)


def circle_residuals(points, circle):
    distances = np.sqrt(
        (points[:, 0] - circle[0])**2 + (points[:, 1] - circle[1])**2
    )
    return np.abs(distances - circle[2])


def ransac_multiple_circles(points, rng, image_shape):
    """Iteratively find circle models and remove their inlier edge points."""
    global IMAGE_WIDTH, IMAGE_HEIGHT
    IMAGE_HEIGHT, IMAGE_WIDTH = image_shape[:2]

    remaining = points.copy()
    circles = []

    for _ in range(MAX_CIRCLE_MODELS):
        if len(remaining) < MIN_CIRCLE_INLIERS:
            break

        best_circle, best_mask, best_count = None, None, 0

        for _trial in range(CIRCLE_TRIALS):
            ids = rng.choice(len(remaining), 3, replace=False)
            circle = circle_from_three_points(
                remaining[ids[0]], remaining[ids[1]], remaining[ids[2]]
            )
            if circle is None:
                continue

            mask = circle_residuals(remaining, circle) <= CIRCLE_DISTANCE_THRESHOLD
            count = int(mask.sum())
            if count > best_count:
                best_circle, best_mask, best_count = circle, mask, count

        if best_circle is None or best_count < MIN_CIRCLE_INLIERS:
            break

        inliers = remaining[best_mask]

        # Refine centre/radius with least-squares algebraic circle fit.
        # This is a refinement step after robust RANSAC inlier selection.
        x = inliers[:, 0].astype(np.float64)
        y = inliers[:, 1].astype(np.float64)
        A = np.column_stack((2*x, 2*y, np.ones_like(x)))
        b = x*x + y*y
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy, c = sol
            radius_sq = c + cx*cx + cy*cy
            if radius_sq > 0:
                refined = np.array([cx, cy, math.sqrt(radius_sq)])
                if np.isfinite(refined).all():
                    best_circle = refined
        except np.linalg.LinAlgError:
            pass

        circles.append({
            "center": (int(round(best_circle[0])), int(round(best_circle[1]))),
            "radius": int(round(best_circle[2])),
            "inliers": int(best_count),
        })

        # Remove only the inliers supporting this circle, then find another.
        remaining = remaining[~best_mask]

    return circles


def process_image(image_path: Path, root: Path, output_root: Path, index: int):
    image, edges, points = get_edge_points(image_path)
    if image is None:
        return None, f"Could not read image: {image_path}"

    relative = image_path.relative_to(root)
    safe_stem = "__".join(relative.with_suffix("").parts)
    vis = image.copy()

    rng = np.random.default_rng(RANDOM_SEED + index)
    lines = ransac_multiple_lines(points, rng)
    circles = ransac_multiple_circles(points, rng, image.shape)

    # Draw RANSAC-fitted lines in green and circles in red.
    for i, line in enumerate(lines, start=1):
        cv2.line(vis, line["p1"], line["p2"], (0, 220, 0), 2)
        mid = (
            (line["p1"][0] + line["p2"][0]) // 2,
            (line["p1"][1] + line["p2"][1]) // 2,
        )
        cv2.putText(
            vis, f"L{i}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 120, 0), 1, cv2.LINE_AA
        )

    for i, circle in enumerate(circles, start=1):
        cv2.circle(vis, circle["center"], circle["radius"], (0, 0, 255), 2)
        cv2.circle(vis, circle["center"], 3, (255, 0, 0), -1)
        cv2.putText(
            vis, f"C{i}", circle["center"], cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 200), 1, cv2.LINE_AA
        )

    # Add compact diagnostic text.
    cv2.putText(
        vis, f"RANSAC lines: {len(lines)} | circles: {len(circles)}",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA
    )

    visual_dir = output_root / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(visual_dir / f"{safe_stem}_ransac.png"), vis)

    record = {
        "image": str(relative),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "edge_points_used": int(len(points)),
        "ransac_line_count": len(lines),
        "ransac_circle_count": len(circles),
        "line_angles_deg": ";".join(f"{m['angle_deg']:.2f}" for m in lines),
        "line_slopes": ";".join(
            "vertical" if m["slope"] is None else f"{m['slope']:.4f}"
            for m in lines
        ),
        "line_inlier_counts": ";".join(str(m["inliers"]) for m in lines),
        "circle_centres_xy": ";".join(
            f"{c['center'][0]},{c['center'][1]}" for c in circles
        ),
        "circle_radii_px": ";".join(str(c["radius"]) for c in circles),
        "circle_inlier_counts": ";".join(str(c["inliers"]) for c in circles),
    }
    return record, None


def main():
    if not DATASET_ROOT.exists():
        print(f"ERROR: Dataset folder not found:\n{DATASET_ROOT}")
        print("Edit DATASET_ROOT near the top of this script and run it again.")
        return

    images = list_images(DATASET_ROOT)
    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]

    if not images:
        print("No image files found. Check DATASET_ROOT and IMAGE_EXTENSIONS.")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records, errors = [], []
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Images selected: {len(images)}")
    print("Running RANSAC line and circle fitting; no Hough transform is used.")

    for i, path in enumerate(images, start=1):
        try:
            record, error = process_image(path, DATASET_ROOT, OUTPUT_ROOT, i)
            if record is not None:
                records.append(record)
            if error:
                errors.append(error)
            if i % 10 == 0 or i == len(images):
                print(f"Processed {i}/{len(images)}")
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            print(f"Skipped {path.name}: {exc}")

    csv_path = OUTPUT_ROOT / "ransac_fit_summary.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")

    if errors:
        (OUTPUT_ROOT / "errors.txt").write_text("\n".join(errors), encoding="utf-8")

    print("\nFinished.")
    print(f"Successful images: {len(records)}")
    print(f"Errors: {len(errors)}")
    print(f"CSV summary: {csv_path}")
    print(f"Visualizations: {OUTPUT_ROOT / 'visualizations'}")
    print("Review visualizations: RANSAC may fit false circles where line edges overlap.")


if __name__ == "__main__":
    main()
