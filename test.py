"""
Compare preprocessing options for CAD sketch images.

What it compares:
1. Grayscale baseline
2. Gaussian blur + global Otsu threshold
3. Median blur + global Otsu threshold
4. Adaptive Gaussian threshold
5. Adaptive mean threshold
6. Canny edges
7. Morphological opening (remove small noise)
8. Morphological closing (connect small stroke gaps)

It saves a comparison grid per selected image and a CSV with basic, measurable
properties. These metrics help screen options, but they do NOT by themselves
prove which preprocessing is best for segmentation. Inspect the visual grid
and, if ground-truth masks are available, compare against those masks too.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import math

# ------------------------- EDIT THESE SETTINGS -------------------------
DATASET_ROOT = Path(r"C:\flutter_projects\cad\ortho_sketch_dataset_4000")
OUTPUT_ROOT = Path(r"C:\flutter_projects\cad\preprocessing_comparison")

# Start small. Set to 0 to process every image.
MAX_IMAGES_TO_PROCESS = 100

# Save comparison grids for only the first N images; set 0 to disable.
MAX_VISUAL_REPORTS = 20

# Assumes dark strokes on a light background.
# Set True if images instead have light strokes on a dark background.
INVERT_STROKES = False

# Adaptive threshold parameters (block_size must be odd and > 1).
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 7

# Morphology kernel: compare gently; large kernels may destroy sketch details.
MORPH_KERNEL_SIZE = 3
# ----------------------------------------------------------------------


def read_image(path: Path):
    """Read an image robustly, including paths containing non-ASCII characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def as_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def make_grid(items, target_height=260):
    """Create a labelled grid with two columns."""
    panels = []
    for title, image in items:
        if image is None:
            continue
        image = as_bgr(image)
        scale = target_height / image.shape[0]
        width = max(1, int(image.shape[1] * scale))
        image = cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)
        banner = np.full((32, width, 3), 35, dtype=np.uint8)
        cv2.putText(banner, title[:45], (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.vstack([banner, image]))

    if not panels:
        return None

    cell_w = max(p.shape[1] for p in panels)
    cell_h = max(p.shape[0] for p in panels)
    cols = 2
    rows = math.ceil(len(panels) / cols)
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        y, x = r * cell_h, c * cell_w
        canvas[y:y + panel.shape[0], x:x + panel.shape[1]] = panel
    return canvas


def foreground_mask(binary):
    """Return a mask where strokes are white (255), background black (0)."""
    if INVERT_STROKES:
        return cv2.bitwise_not(binary)
    return binary


def process_image(path: Path, save_visual: bool):
    image = read_image(path)
    if image is None:
        return {"image": str(path), "status": "read_error"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Candidate 1: Gaussian smoothing + Otsu threshold
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_gaussian = cv2.threshold(
        gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Candidate 2: Median smoothing + Otsu threshold
    median = cv2.medianBlur(gray, 5)
    _, otsu_median = cv2.threshold(
        median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Candidate 3 & 4: Adaptive thresholds
    block_size = max(3, ADAPTIVE_BLOCK_SIZE | 1)
    adaptive_gaussian = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block_size, ADAPTIVE_C
    )
    adaptive_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        block_size, ADAPTIVE_C
    )

    # Make the foreground convention consistent: strokes white, background black.
    candidates = {
        "otsu_gaussian": otsu_gaussian,
        "otsu_median": otsu_median,
        "adaptive_gaussian": adaptive_gaussian,
        "adaptive_mean": adaptive_mean,
    }
    if INVERT_STROKES:
        candidates = {k: cv2.bitwise_not(v) for k, v in candidates.items()}

    # Canny is an edge map, not a replacement for a filled foreground mask.
    edges = cv2.Canny(gaussian, 50, 150)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    morphology = {}
    for name, mask in candidates.items():
        morphology[f"{name}_opening"] = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )
        morphology[f"{name}_closing"] = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel
        )

    # Use a consistent stroke-white mask for metrics.
    # Thresholds above are normalized to strokes white unless INVERT_STROKES is True.
    # Compute simple screening metrics; none is a ground-truth accuracy measure.
    metric_images = dict(candidates)
    metric_images.update(morphology)

    row = {
        "image": str(path),
        "status": "ok",
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "canny_edge_ratio": float(np.count_nonzero(edges) / edges.size),
    }

    for name, mask in metric_images.items():
        # Foreground ratio, number of connected components, and component sizes.
        foreground_ratio = float(np.count_nonzero(mask) / mask.size)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8
        )
        component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([])
        row[f"{name}_foreground_ratio"] = foreground_ratio
        row[f"{name}_component_count"] = int(max(0, count - 1))
        row[f"{name}_tiny_component_ratio"] = (
            float(np.sum(component_areas <= 3) / len(component_areas))
            if len(component_areas) else 0.0
        )

    if save_visual:
        items = [
            ("Original", image),
            ("Grayscale", gray),
            ("Gaussian + Otsu", candidates["otsu_gaussian"]),
            ("Median + Otsu", candidates["otsu_median"]),
            ("Adaptive Gaussian", candidates["adaptive_gaussian"]),
            ("Adaptive Mean", candidates["adaptive_mean"]),
            ("Canny edges (edge map only)", edges),
            ("Opening after adaptive Gaussian", morphology["adaptive_gaussian_opening"]),
            ("Closing after adaptive Gaussian", morphology["adaptive_gaussian_closing"]),
        ]
        grid = make_grid(items)
        if grid is not None:
            visual_dir = OUTPUT_ROOT / "visual_reports"
            visual_dir.mkdir(parents=True, exist_ok=True)
            try:
                rel = path.relative_to(DATASET_ROOT)
                stem = "__".join(rel.with_suffix("").parts)
            except ValueError:
                stem = path.stem
            out_path = visual_dir / f"{stem}_preprocessing.png"
            cv2.imwrite(str(out_path), grid)
            row["visual_report"] = str(out_path)

    return row


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        p for p in DATASET_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
        }
        and OUTPUT_ROOT not in p.parents
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {DATASET_ROOT}")

    print(f"Found {len(image_paths)} images.")
    if MAX_IMAGES_TO_PROCESS > 0:
        image_paths = image_paths[:MAX_IMAGES_TO_PROCESS]
        print(f"Quick-test mode: processing {len(image_paths)} images.")
    else:
        print("Full-dataset mode: processing all images.")

    records = []
    for i, path in enumerate(image_paths, start=1):
        save_visual = i <= MAX_VISUAL_REPORTS
        records.append(process_image(path, save_visual))
        if i % 25 == 0 or i == len(image_paths):
            print(f"Processed {i}/{len(image_paths)}")

    df = pd.DataFrame(records)
    csv_path = OUTPUT_ROOT / "preprocessing_metrics.csv"
    df.to_csv(csv_path, index=False)

    print("\nFinished.")
    print(f"Metrics CSV: {csv_path}")
    print(f"Visual reports: {OUTPUT_ROOT / 'visual_reports'}")
    print("\nImportant: choose based on the saved comparison images and, if available, mask-based metrics.")
    print("The numeric metrics are screening indicators, not proof of segmentation quality.")


if __name__ == "__main__":
    main()
