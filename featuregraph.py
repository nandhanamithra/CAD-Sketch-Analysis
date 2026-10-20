import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

GEOMETRY_DIR = Path("consolidated_geometry")
GRAPH_OUTPUT_DIR = Path("graph_output")


class CADGraphOptimizer:

    def __init__(
        self, json_features, distance_threshold=12.0, angle_threshold_deg=8.0
    ):
        # Filter out tiny line noise
        self.raw_features = [
            f
            for f in json_features
            if f.get("type") != "line" or f.get("length", 0) >= 15.0
        ]
        self.dist_tol = distance_threshold
        self.angle_tol = angle_threshold_deg
        self.graph = nx.Graph()

    def build_optimized_graph(self):
        # 1. Add primitive nodes
        for feature in self.raw_features:
            fid = feature["id"]
            self.graph.add_node(fid, **feature)

        lines = [
            (n, d)
            for n, d in self.graph.nodes(data=True)
            if d.get("type") == "line"
        ]
        circles = [
            (n, d)
            for n, d in self.graph.nodes(data=True)
            if d.get("type") == "circle"
        ]
        contours = [
            (n, d)
            for n, d in self.graph.nodes(data=True)
            if d.get("type") == "contour"
        ]

        # 2. Extract COINCIDENT relationships (Topological Connections)
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                id1, d1 = lines[i]
                id2, d2 = lines[j]
                if self._are_coincident(d1, d2):
                    self.graph.add_edge(id1, id2, constraint="COINCIDENT")

        # 3. Directional Axis Grouping (Prevents N^2 PARALLEL edge explosion)
        self._add_minimal_parallel_constraints(lines)

        # 4. Containment constraints (Circles inside Outer Loops)
        for c_id, c_data in circles:
            cx, cy = c_data["center"]
            for cont_id, cont_data in contours:
                bbox = cont_data.get("bounding_box", {})
                if (
                    bbox
                    and (bbox["x"] <= cx <= bbox["x"] + bbox["width"])
                    and (bbox["y"] <= cy <= bbox["y"] + bbox["height"])
                ):
                    self.graph.add_edge(
                        c_id, cont_id, constraint="INNER_CUTOUT"
                    )

        return self.graph

    def _are_coincident(self, d1, d2):
        pts1 = [np.array(d1["start"]), np.array(d1["end"])]
        pts2 = [np.array(d2["start"]), np.array(d2["end"])]
        for p1 in pts1:
            for p2 in pts2:
                if np.linalg.norm(p1 - p2) <= self.dist_tol:
                    return True
        return False

    def _add_minimal_parallel_constraints(self, lines):
        """Groups lines by angle into direction buckets to avoid dense all-to-all edges."""
        buckets = {}
        for line_id, data in lines:
            angle = data.get("angle_deg", 0.0) % 180.0
            matched = False
            for bucket_angle in buckets:
                diff = abs(angle - bucket_angle) % 180.0
                if min(diff, 180.0 - diff) <= self.angle_tol:
                    buckets[bucket_angle].append(line_id)
                    matched = True
                    break
            if not matched:
                buckets[angle] = [line_id]

        # Add PARALLEL edges only along a linear chain per angle group
        for group in buckets.values():
            if len(group) > 1:
                for k in range(len(group) - 1):
                    # Do not duplicate if already connected by COINCIDENT
                    if not self.graph.has_edge(group[k], group[k + 1]):
                        self.graph.add_edge(
                            group[k], group[k + 1], constraint="PARALLEL"
                        )


def render_clean_cad_graph(graph, output_path):
    plt.figure(figsize=(14, 10))

    # Spring layout with repulsion forces to prevent overlapping nodes (like F12 & F22)
    pos = nx.spring_layout(graph, seed=42, k=1.2, iterations=100)

    edge_labels = nx.get_edge_attributes(graph, "constraint")

    color_map = []
    for _, attrs in graph.nodes(data=True):
        t = attrs.get("type", "")
        if t == "line":
            color_map.append("#4A90E2")  # Blue
        elif t == "circle":
            color_map.append("#50E3C2")  # Teal Green
        else:
            color_map.append("#F5A623")  # Orange

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=color_map,
        node_size=1100,
        edgecolors="#222222",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=9,
        font_weight="bold",
        font_color="white",
        font_family="sans-serif",
    )

    # Draw edges with clear label styling
    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.6, edge_color="#7F8C8D")
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    )

    plt.title(
        "CAD Feature Graph (Optimized Topology)", fontsize=14, fontweight="bold"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    json_files = list(GEOMETRY_DIR.glob("*_geometry.json"))
    if not json_files:
        print(f"No geometry JSON found in {GEOMETRY_DIR.resolve()}")
        return 1

    json_path = sorted(json_files, key=lambda p: p.stat().st_mtime)[-1]
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    optimizer = CADGraphOptimizer(data.get("features", []))
    graph = optimizer.build_optimized_graph()

    GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = GRAPH_OUTPUT_DIR / f"{json_path.stem}_optimized_graph.png"

    render_clean_cad_graph(graph, out_png)
    print(f"Saved optimized graph image to: {out_png.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())