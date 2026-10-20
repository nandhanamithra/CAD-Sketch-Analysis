"""
Improved geometric detection for clean/freehand CAD sketch images.

Key enhancements:
- Skeletonization before LSD to prevent parallel double-edge lines on thick strokes.
- Contour circularity check to reliably detect hand-drawn circles/ellipses.
- Endpoint snapping to snap nearby line corners into clean vertices.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path("consolidated_geometry")

# Parameter tuning
ANGLE_TOLERANCE_DEG = 8.0
PERPENDICULAR_DISTANCE_TOLERANCE_PX = 5.0
SEGMENT_GAP_TOLERANCE_PX = 15.0
MIN_LINE_LENGTH_PX = 12.0
MIN_CONTOUR_AREA_PX = 50.0
CORNER_SNAP_RADIUS_PX = 10.0


def skeletonize_binary(binary):
    """Reduces thick strokes to a 1-pixel thin skeleton."""
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    img = binary.copy()
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            done = True
    return skel


def angle_diff_undirected(a, b):
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def line_params(segment):
    x1, y1, x2, y2 = map(float, segment)
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    offset = nx * x1 + ny * y1
    t1 = x1 * ux + y1 * uy
    t2 = x2 * ux + y2 * uy
    return {
        "segment": [x1, y1, x2, y2],
        "length": length,
        "angle": angle,
        "ux": ux, "uy": uy,
        "nx": nx, "ny": ny,
        "offset": offset,
        "tmin": min(t1, t2),
        "tmax": max(t1, t2),
    }


def can_merge(a, b):
    if angle_diff_undirected(a["angle"], b["angle"]) > ANGLE_TOLERANCE_DEG:
        return False

    ax1, ay1, ax2, ay2 = a["segment"]
    dx, dy = ax2 - ax1, ay2 - ay1
    length = max(math.hypot(dx, dy), 1e-9)
    distances = [
        abs(dx * (y - ay1) - dy * (x - ax1)) / length
        for x, y in [
            (b["segment"][0], b["segment"][1]),
            (b["segment"][2], b["segment"][3]),
        ]
    ]
    if max(distances) > PERPENDICULAR_DISTANCE_TOLERANCE_PX:
        return False

    a0 = np.array([ax1, ay1], dtype=float)
    ux, uy = a["ux"], a["uy"]
    b_points = [
        np.array([b["segment"][0], b["segment"][1]], dtype=float),
        np.array([b["segment"][2], b["segment"][3]], dtype=float),
    ]
    b_proj = [float((p[0] - a0[0]) * ux + (p[1] - a0[1]) * uy) for p in b_points]
    a_proj = [0.0, length]
    gap = max(0.0, max(min(a_proj), min(b_proj)) - min(max(a_proj), max(b_proj)))
    return gap <= SEGMENT_GAP_TOLERANCE_PX


def merge_group(group):
    points = []
    for item in group:
        x1, y1, x2, y2 = item["segment"]
        points.extend([[x1, y1], [x2, y2]])
    pts = np.asarray(points, dtype=np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    projections = centered @ direction
    p1 = mean + direction * projections.min()
    p2 = mean + direction * projections.max()
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    length = math.hypot(x2 - x1, y2 - y1)
    return {
        "type": "line",
        "start": [round(x1, 3), round(y1, 3)],
        "end": [round(x2, 3), round(y2, 3)],
        "length": round(length, 3),
        "angle_deg": round(math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180, 3),
        "sources": ["LSD_Skeleton"],
        "merged_segment_count": len(group),
    }


def merge_lines(raw_segments):
    params = []
    for seg in raw_segments:
        try:
            p = line_params(seg)
            if p["length"] >= MIN_LINE_LENGTH_PX:
                params.append(p)
        except (ValueError, ZeroDivisionError):
            continue

    groups = []
    for candidate in sorted(params, key=lambda p: p["length"], reverse=True):
        matching = [i for i, group in enumerate(groups)
                    if any(can_merge(candidate, existing) for existing in group)]
        if not matching:
            groups.append([candidate])
        else:
            first = matching[0]
            groups[first].append(candidate)
            for idx in reversed(matching[1:]):
                groups[first].extend(groups.pop(idx))

    merged = [merge_group(group) for group in groups]
    return [line for line in merged if line["length"] >= MIN_LINE_LENGTH_PX]


def snap_corners(lines, radius=CORNER_SNAP_RADIUS_PX):
    """Snaps endpoints of distinct lines together if they are within a small radius."""
    endpoints = []
    for i, line in enumerate(lines):
        endpoints.append((i, "start", np.array(line["start"])))
        endpoints.append((i, "end", np.array(line["end"])))

    n = len(endpoints)
    for i in range(n):
        for j in range(i + 1, n):
            idx1, key1, pt1 = endpoints[i]
            idx2, key2, pt2 = endpoints[j]
            if idx1 == idx2:
                continue
            dist = np.linalg.norm(pt1 - pt2)
            if dist <= radius:
                midpoint = ((pt1 + pt2) / 2.0).round(3).tolist()
                lines[idx1][key1] = midpoint
                lines[idx2][key2] = midpoint
    return lines


def main():
    if len(sys.argv) < 2:
        print('Usage: python geometric_result_consolidation.py "path_to_image.png"')
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"ERROR: Image not found: {image_path}")
        return 1

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"ERROR: Could not read image: {image_path}")
        return 1

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold image to separate sketch lines from background
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Skeletonize to reduce thick/double stroke lines to a single center pixel
    skel = skeletonize_binary(binary)

    # Detect lines on skeleton image using LSD
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected, _, _, _ = lsd.detect(skel)
    raw_segments = detected.reshape(-1, 4).tolist() if detected is not None else []
    merged_lines = merge_lines(raw_segments)
    merged_lines = snap_corners(merged_lines)

    # Contour and Circularity Detection for Circles/Ellipses
    contours_cv, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    contour_features = []

    for contour in contours_cv:
        area = float(cv2.contourArea(contour))
        if area < MIN_CONTOUR_AREA_PX:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)

        # Classify highly circular loops as Circles
        if circularity > 0.55:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            circles.append({
                "type": "circle",
                "center": [round(float(cx), 3), round(float(cy), 3)],
                "radius": round(float(radius), 3),
                "circularity": round(circularity, 3),
                "sources": ["Contour_Circularity"],
            })
        else:
            x, y, w, h = cv2.boundingRect(contour)
            contour_features.append({
                "type": "contour",
                "points": contour.reshape(-1, 2).astype(float).tolist(),
                "area": round(area, 3),
                "perimeter": round(perimeter, 3),
                "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "sources": ["OpenCV_contours"],
            })

    features = []
    for item in merged_lines + circles + contour_features:
        item["id"] = f"F{len(features) + 1}"
        features.append(item)

    result = {
        "image_id": image_path.stem,
        "image_path": str(image_path.resolve()),
        "image_width": int(gray.shape[1]),
        "image_height": int(gray.shape[0]),
        "feature_count": len(features),
        "features": features,
        "detection_counts": {
            "raw_lsd_segments": len(raw_segments),
            "merged_lines": len(merged_lines),
            "circles": len(circles),
            "contours": len(contour_features),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    json_path = OUTPUT_DIR / f"{stem}_geometry.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Save visualization preview
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Draw lines in red
    for item in merged_lines:
        x1, y1 = map(lambda v: int(round(v)), item["start"])
        x2, y2 = map(lambda v: int(round(v)), item["end"])
        cv2.line(preview, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Draw detected circles in green
    for circle in circles:
        cx, cy = map(lambda v: int(round(v)), circle["center"])
        r = int(round(circle["radius"]))
        cv2.circle(preview, (cx, cy), r, (0, 255, 0), 2)

    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_skeleton.png"), skel)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_detections.png"), preview)

    print(f"Processed image: {image_path.resolve()}")
    print(f"Raw LSD segments: {len(raw_segments)}")
    print(f"Merged lines: {len(merged_lines)}")
    print(f"Circles detected: {len(circles)}")
    print(f"Saved preview: {(OUTPUT_DIR / (stem + '_detections.png')).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())