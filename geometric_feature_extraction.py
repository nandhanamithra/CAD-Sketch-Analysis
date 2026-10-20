"""
Geometric feature and relationship extraction for hand-drawn CAD sketches.

Input:
    consolidated_geometry/<image_id>_geometry.json

Output:
    extracted_features/<image_id>_features.json

Run:
    python geometric_feature_extraction.py
    python geometric_feature_extraction.py YOUR_IMAGE_ID

Install dependencies if needed:
    pip install opencv-python numpy
"""

import json
import math
import sys
from itertools import combinations
from pathlib import Path

INPUT_DIR = Path("consolidated_geometry")
OUTPUT_DIR = Path("extracted_features")

# Pixel-space tolerances. Tune these to match your sketch resolution.
ANGLE_TOLERANCE_DEGREES = 8.0
POINT_TOLERANCE_PIXELS = 6.0


def distance(p1, p2):
    return math.hypot(
        float(p1[0]) - float(p2[0]),
        float(p1[1]) - float(p2[1]),
    )


def angle_difference_degrees(angle_a, angle_b):
    """Smallest difference between undirected line angles, from 0 to 90 degrees."""
    difference = abs((float(angle_a) - float(angle_b)) % 180.0)
    return min(difference, 180.0 - difference)


def line_angle(line):
    """Return the orientation of a line in degrees, in the range [0, 180)."""
    x1, y1 = line["start"]
    x2, y2 = line["end"]
    return math.degrees(math.atan2(float(y2) - float(y1),
                                   float(x2) - float(x1))) % 180.0


def is_parallel(line_a, line_b):
    # IMPORTANT: pass numerical angles, not the line dictionaries.
    angle_a = line_angle(line_a)
    angle_b = line_angle(line_b)
    difference = angle_difference_degrees(angle_a, angle_b)
    return difference <= ANGLE_TOLERANCE_DEGREES


def is_perpendicular(line_a, line_b):
    # IMPORTANT: pass numerical angles, not the line dictionaries.
    angle_a = line_angle(line_a)
    angle_b = line_angle(line_b)
    difference = angle_difference_degrees(angle_a, angle_b)
    return abs(difference - 90.0) <= ANGLE_TOLERANCE_DEGREES


def lines_connected(line_a, line_b):
    endpoints_a = [line_a["start"], line_a["end"]]
    endpoints_b = [line_b["start"], line_b["end"]]
    return any(
        distance(point_a, point_b) <= POINT_TOLERANCE_PIXELS
        for point_a in endpoints_a
        for point_b in endpoints_b
    )


def circles_concentric(circle_a, circle_b):
    return distance(circle_a["center"], circle_b["center"]) <= POINT_TOLERANCE_PIXELS


def circle_inside_contour_bbox(circle, contour):
    """Approximate test: checks whether the full circle fits inside the contour bounding box."""
    box = contour.get("bounding_box")
    if not box:
        return False

    cx, cy = circle["center"]
    radius = float(circle["radius"])

    return (
        cx - radius >= box["x"]
        and cy - radius >= box["y"]
        and cx + radius <= box["x"] + box["width"]
        and cy + radius <= box["y"] + box["height"]
    )


def extract_features(data):
    raw_features = data.get("features", [])
    lines = [feature for feature in raw_features if feature.get("type") == "line"]
    circles = [feature for feature in raw_features if feature.get("type") == "circle"]
    contours = [feature for feature in raw_features if feature.get("type") == "contour"]

    features = []
    relationships = []

    # Calculate properties for each line.
    for line in lines:
        start = line["start"]
        end = line["end"]
        features.append({
            "id": line.get("id"),
            "type": "line_segment",
            "start": start,
            "end": end,
            "length_px": round(distance(start, end), 3),
            "orientation_deg": round(line_angle(line), 3),
            "sources": line.get("sources", []),
        })

    # Calculate properties for each circle.
    for circle in circles:
        radius = float(circle["radius"])
        features.append({
            "id": circle.get("id"),
            "type": "circle",
            "center": circle["center"],
            "radius_px": round(radius, 3),
            "diameter_px": round(2.0 * radius, 3),
            "area_px2": round(math.pi * radius * radius, 3),
            "sources": circle.get("sources", []),
        })

    # Keep contour measurements produced by the consolidation script.
    for contour in contours:
        features.append({
            "id": contour.get("id"),
            "type": "contour",
            "area_px2": contour.get("area"),
            "perimeter_px": contour.get("perimeter"),
            "bounding_box": contour.get("bounding_box"),
            "sources": contour.get("sources", []),
        })

    # Compare pairs of lines to infer geometric relationships.
    for line_a, line_b in combinations(lines, 2):
        id_a = line_a.get("id")
        id_b = line_b.get("id")

        if is_parallel(line_a, line_b):
            relationships.append({
                "type": "parallel",
                "between": [id_a, id_b],
                "confidence": "heuristic",
            })

        if is_perpendicular(line_a, line_b):
            relationships.append({
                "type": "perpendicular",
                "between": [id_a, id_b],
                "confidence": "heuristic",
            })

        if lines_connected(line_a, line_b):
            relationships.append({
                "type": "connected_endpoints",
                "between": [id_a, id_b],
                "confidence": "heuristic",
            })

    # Compare circles to find approximately concentric pairs.
    for circle_a, circle_b in combinations(circles, 2):
        if circles_concentric(circle_a, circle_b):
            relationships.append({
                "type": "concentric",
                "between": [circle_a.get("id"), circle_b.get("id")],
                "confidence": "heuristic",
            })

    # Approximate circle-in-contour relationships using bounding boxes.
    for circle in circles:
        for contour in contours:
            if circle_inside_contour_bbox(circle, contour):
                relationships.append({
                    "type": "circle_inside_contour_bbox",
                    "between": [circle.get("id"), contour.get("id")],
                    "confidence": "approximate_bbox_test",
                })

    return {
        "image_id": data.get("image_id", "unknown"),
        "input_feature_count": len(raw_features),
        "extracted_feature_count": len(features),
        "relationship_count": len(relationships),
        "features": features,
        "relationships": relationships,
        "notes": [
            "Measurements are in image pixels, not real-world CAD units.",
            "Relationships are heuristic and depend on the configured tolerances.",
            "Circle containment is checked against a contour bounding box, not the exact contour.",
            "These are candidate relationships for later CAD-constraint inference.",
        ],
    }


def main():
    image_id = sys.argv[1] if len(sys.argv) > 1 else "sample_sketch"
    input_path = INPUT_DIR / f"{image_id}_geometry.json"

    if not input_path.exists():
        print(f"ERROR: Input JSON not found:\n{input_path.resolve()}")
        print("\nRun geometric_result_consolidation.py first, or pass your image ID:")
        print("python geometric_feature_extraction.py your_image_id")
        return 1

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: Could not read input JSON: {exc}")
        return 1

    if not isinstance(data.get("features"), list):
        print("ERROR: Input JSON must contain a 'features' list.")
        return 1

    try:
        result = extract_features(data)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        print(f"ERROR: Invalid feature data in input JSON: {exc}")
        print("Check that line features contain 'start'/'end' and circles contain 'center'/'radius'.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_id}_features.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Image ID: {result['image_id']}")
    print(f"Input features: {result['input_feature_count']}")
    print(f"Extracted features: {result['extracted_feature_count']}")
    print(f"Relationships found: {result['relationship_count']}")
    print(f"Saved results to: {output_path.resolve()}")

    for relationship in result["relationships"]:
        print(f"  {relationship['type']}: {relationship['between']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
