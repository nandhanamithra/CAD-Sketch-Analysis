"""
OpenCV dataset inspection: thresholding, contours, and Hough line transforms.

What it does:
1. Recursively finds image files under DATASET_ROOT.
2. Runs grayscale conversion + Gaussian blur + Otsu thresholding.
3. Finds contours on the foreground mask.
4. Runs Canny + Probabilistic Hough Line Transform.
5. Writes per-image measurements to a CSV.
6. Saves a small set of visual reports so you can inspect the results without
   generating thousands of large images.

Install:
    pip install opencv-python numpy pandas

Before running:
    Set DATASET_ROOT below to the folder containing your dataset images.
    Example: r"C:\\path\\to\\dataset"

Notes:
- The script processes image files only; CSV and .npz annotation files are ignored.
- For dark strokes on a light background, the default inverted Otsu threshold is
  appropriate. If your images have the opposite polarity, change THRESHOLD_MODE.
"""

from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd

# --------------------------- USER SETTINGS ---------------------------

DATASET_ROOT = r"C:\flutter_projects\cad\ortho_sketch_dataset_4000"   # CHANGE THIS
OUTPUT_ROOT = r"C:\flutter_projects\cad\analysis"  # CHANGE if desired

# "dark_strokes" = dark lines on a light background (typical paper sketch)
# "light_strokes" = light lines on a dark background
THRESHOLD_MODE = "dark_strokes"

# Save visual reports for only this many images, while analysing every image.
# Set to 0 to disable visual reports. Set to -1 to save reports for all images.
MAX_VISUAL_REPORTS = 40

# For a quick test, set this to 100 or another small number.
# Set to 0 to analyse the entire dataset.
MAX_IMAGES_TO_PROCESS = 100

# Hough transform settings. You may tune these after inspecting the reports.
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 25
MIN_LINE_LENGTH = 12
MAX_LINE_GAP = 5

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# --------------------------------------------------------------------


def read_image(path: Path):
    """Read image safely, including paths that may contain non-ASCII characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def make_panel(images, labels, target_height=300):
    """Resize images to a common height and combine them into a labelled panel."""
    resized = []
    for img, label in zip(images, labels):
        if img is None:
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scale = target_height / img.shape[0]
        width = max(1, int(img.shape[1] * scale))
        view = cv2.resize(img, (width, target_height), interpolation=cv2.INTER_AREA)
        banner = np.zeros((32, view.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            banner, label[:42], (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        resized.append(np.vstack([banner, view]))

    if not resized:
        return None

    # Put panels on a grid with two columns.
    cols = 2
    rows = math.ceil(len(resized) / cols)
    cell_w = max(img.shape[1] for img in resized)
    cell_h = max(img.shape[0] for img in resized)
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)

    for i, img in enumerate(resized):
        r, c = divmod(i, cols)
        y, x = r * cell_h, c * cell_w
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img

    return canvas


def process_image(path: Path, output_root: Path, save_visual: bool):
    image = read_image(path)
    if image is None:
        return {
            "image": str(path),
            "status": "read_error",
            "width": None,
            "height": None,
            "contour_count": None,
            "largest_contour_area": None,
            "hough_line_count": None,
            "foreground_ratio": None,
            "edge_pixel_ratio": None,
        }

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if THRESHOLD_MODE == "dark_strokes":
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Small morphological opening can remove isolated speckles.
    # Keep the original threshold mask for comparison; use clean_mask for contours.
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        clean_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_areas = [float(cv2.contourArea(c)) for c in contours]
    largest_area = max(contour_areas, default=0.0)

    contour_view = image.copy()
    cv2.drawContours(contour_view, contours, -1, (0, 180, 0), 1)

    # Canny + probabilistic Hough transform
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    hough_view = image.copy()
    line_count = 0
    total_line_length = 0.0
    if lines is not None:
        # OpenCV normally returns (N, 1, 4), but some versions/wrappers
        # may return (N, 4). Flattening to (-1, 4) handles both formats.
        line_segments = np.asarray(lines).reshape(-1, 4)
        line_count = len(line_segments)

        for segment in line_segments:
            x1, y1, x2, y2 = map(int, segment)
            cv2.line(hough_view, (x1, y1), (x2, y2), (0, 0, 255), 2)
            total_line_length += math.hypot(x2 - x1, y2 - y1)

    foreground_ratio = float(np.count_nonzero(clean_mask)) / (width * height)
    edge_ratio = float(np.count_nonzero(edges)) / (width * height)

    row = {
        "image": str(path),
        "status": "ok",
        "width": width,
        "height": height,
        "contour_count": len(contours),
        "largest_contour_area": largest_area,
        "hough_line_count": line_count,
        "total_hough_line_length_px": total_line_length,
        "foreground_ratio": foreground_ratio,
        "edge_pixel_ratio": edge_ratio,
    }

    if save_visual:
        # Keep a folder per image based on its relative path, avoiding filename collisions.
        try:
            relative = path.relative_to(Path(DATASET_ROOT))
            safe_stem = "__".join(relative.with_suffix("").parts)
        except ValueError:
            safe_stem = path.stem

        report = make_panel(
            [image, binary, contour_view, edges, hough_view],
            ["Original", "Otsu threshold", "Contours", "Canny edges", "Hough lines"],
        )
        if report is not None:
            visual_dir = output_root / "visual_reports"
            visual_dir.mkdir(parents=True, exist_ok=True)
            out_path = visual_dir / f"{safe_stem}_opencv_report.jpg"
            cv2.imwrite(str(out_path), report)
            row["visual_report"] = str(out_path)

    return row


def main():
    dataset_root = Path(DATASET_ROOT)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"DATASET_ROOT does not exist: {dataset_root}\n"
            "Edit DATASET_ROOT near the top of this script."
        )

    image_paths = sorted(
        p for p in dataset_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and output_root not in p.parents
    )

    if not image_paths:
        raise RuntimeError(
            f"No image files found under {dataset_root}. "
            "Check the dataset path and image extensions."
        )

    print(f"Found {len(image_paths)} image files.")

    if MAX_IMAGES_TO_PROCESS > 0:
        image_paths = image_paths[:MAX_IMAGES_TO_PROCESS]
        print(f"Quick-test mode: processing the first {len(image_paths)} images.")
    else:
        print("Full-dataset mode: processing all images.")

    print("Analysing thresholding, contours, and Hough lines...")

    records = []
    for index, path in enumerate(image_paths, start=1):
        if MAX_VISUAL_REPORTS == -1:
            save_visual = True
        else:
            save_visual = index <= MAX_VISUAL_REPORTS

        records.append(process_image(path, output_root, save_visual))

        if index % 100 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)}")

    df = pd.DataFrame(records)
    csv_path = output_root / "opencv_dataset_summary.csv"
    df.to_csv(csv_path, index=False)

    successful = df[df["status"] == "ok"]
    summary = {
        "total_images_found": len(image_paths),
        "successfully_processed": len(successful),
        "read_errors": int((df["status"] != "ok").sum()),
        "mean_contour_count": (
            float(successful["contour_count"].mean()) if len(successful) else None
        ),
        "median_contour_count": (
            float(successful["contour_count"].median()) if len(successful) else None
        ),
        "mean_hough_line_count": (
            float(successful["hough_line_count"].mean()) if len(successful) else None
        ),
        "median_hough_line_count": (
            float(successful["hough_line_count"].median()) if len(successful) else None
        ),
        "mean_foreground_ratio": (
            float(successful["foreground_ratio"].mean()) if len(successful) else None
        ),
        "mean_edge_pixel_ratio": (
            float(successful["edge_pixel_ratio"].mean()) if len(successful) else None
        ),
    }

    summary_path = output_root / "opencv_overall_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("\nFinished.")
    print(f"Per-image results: {csv_path}")
    print(f"Overall summary:   {summary_path}")
    print(f"Visual reports:    {output_root / 'visual_reports'}")
    print("\nOverall summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
