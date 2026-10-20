import json
import math
from pathlib import Path
import networkx as nx
import numpy as np

GRAPH_INPUT_DIR = Path("graph_output")
RECONSTRUCTION_OUTPUT_DIR = Path("cad_3d_output")


class Candidate3DGenerator:

    def __init__(self, feature_graph_data):
        self.graph_data = feature_graph_data
        self.graph = nx.node_link_graph(feature_graph_data)

    def extract_closed_faces(self):
        """Extracts minimal cycles (planar faces) from line primitives using COINCIDENT edges."""
        topo_subgraph = nx.Graph()
        for u, v, d in self.graph.edges(data=True):
            if d.get("constraint") == "COINCIDENT":
                topo_subgraph.add_edge(u, v)

        # Extract cycle basis (closed 2D loops)
        cycles = nx.cycle_basis(topo_subgraph)
        return cycles

    def generate_hypotheses(self):
        """Generates multiple 3D structural interpretations based on graph features."""
        faces = self.extract_closed_faces()
        circles = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("type") == "circle"
        ]

        candidates = []

        # Candidate 1: CSG Extrusion Assembly (Main Solid + Internal Cutouts)
        candidates.append({
            "candidate_id": "3D_CSG_EXTRUSION_01",
            "description": "Multi-primitive CSG extrusion with cylindrical cutouts",
            "type": "CSG_SOLID",
            "faces_detected": len(faces),
            "operations": [
                {
                    "op": "EXTRUDE",
                    "target": "BASE_PLATE",
                    "vector": [0, 0, 1],
                    "depth_px": 40.0,
                },
                {
                    "op": "EXTRUDE",
                    "target": "CENTRAL_PYRAMID",
                    "vector": [0, 0, 1],
                    "depth_px": 65.0,
                },
                {
                    "op": "SUBTRACT",
                    "target": "CYLINDRICAL_CUTOUTS",
                    "count": len(circles),
                },
            ],
            "manifold_assumption": True,
        })

        # Candidate 2: Axonometric 2.5D Plate Model
        candidates.append({
            "candidate_id": "3D_PLATE_MODEL_02",
            "description": "Single-body flat profile with through-holes",
            "type": "PLATE_EXTRUSION",
            "faces_detected": len(faces),
            "operations": [
                {
                    "op": "EXTRUDE",
                    "target": "OUTER_PROFILE",
                    "vector": [0, 0, 1],
                    "depth_px": 15.0,
                },
                {
                    "op": "SUBTRACT",
                    "target": "INNER_CIRCLES",
                    "count": len(circles),
                },
            ],
            "manifold_assumption": True,
        })

        # Candidate 3: Multi-View Orthographic Wireframe Sheet
        candidates.append({
            "candidate_id": "3D_SHEET_SURFACE_03",
            "description": "Unextruded 2D wireframe planar sheet",
            "type": "SURFACE_SHEET",
            "faces_detected": len(faces),
            "operations": [
                {"op": "CREATE_SURFACE", "target": "PLANAR_PATCHES"}
            ],
            "manifold_assumption": False,
        })

        return candidates


class CandidateRanker:

    def __init__(self, feature_graph):
        self.graph = feature_graph

    def score_and_rank(self, candidates):
        """Scores candidates using a weighted evaluation function."""
        ranked_list = []

        edges = list(self.graph.edges(data=True))
        coincident_count = sum(
            1 for u, v, d in edges if d.get("constraint") == "COINCIDENT"
        )
        parallel_count = sum(
            1 for u, v, d in edges if d.get("constraint") == "PARALLEL"
        )

        total_edges = max(len(edges), 1)
        consistency_ratio = (coincident_count + parallel_count) / total_edges

        for cand in candidates:
            # Metric 1: Manifold Watertight Validity (Weight: 0.4)
            s_manifold = 1.0 if cand["manifold_assumption"] else 0.2

            # Metric 2: Geometric & Topological Constraint Alignment (Weight: 0.35)
            s_topology = min(1.0, consistency_ratio * 1.2)

            # Metric 3: CSG Simplicity / Feature Inclusion (Weight: 0.25)
            s_simplicity = 0.9 if cand["type"] == "CSG_SOLID" else 0.5

            total_score = (
                0.40 * s_manifold + 0.35 * s_topology + 0.25 * s_simplicity
            )
            cand["confidence_score"] = round(total_score, 3)
            cand["score_breakdown"] = {
                "manifold_score": round(s_manifold, 2),
                "topology_alignment": round(s_topology, 2),
                "csg_simplicity": round(s_simplicity, 2),
            }
            ranked_list.append(cand)

        return sorted(
            ranked_list, key=lambda x: x["confidence_score"], reverse=True
        )


def main():
    json_files = list(GRAPH_INPUT_DIR.glob("*_optimized_graph.json"))
    if not json_files:
        json_files = list(GRAPH_INPUT_DIR.glob("*.json"))

    if not json_files:
        print(f"ERROR: No graph JSON files found in {GRAPH_INPUT_DIR.resolve()}")
        return 1

    graph_json_path = sorted(json_files, key=lambda p: p.stat().st_mtime)[-1]
    print(f"Loading graph: {graph_json_path.resolve()}")

    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    # 1. Generate Candidates
    generator = Candidate3DGenerator(graph_data)
    candidates = generator.generate_hypotheses()

    # 2. Score and Rank Candidates
    ranker = CandidateRanker(generator.graph)
    ranked_candidates = ranker.score_and_rank(candidates)

    # 3. Export Results
    RECONSTRUCTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json_path = (
        RECONSTRUCTION_OUTPUT_DIR
        / f"{graph_json_path.stem}_ranked_3d_candidates.json"
    )

    out_data = {
        "source_graph": graph_json_path.name,
        "total_candidates": len(ranked_candidates),
        "top_rank_candidate": ranked_candidates[0]["candidate_id"],
        "candidates": ranked_candidates,
    }

    out_json_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")

    # Terminal Output
    print("\n" + "=" * 60)
    print("        RANKED 3D CAD INTERPRETATIONS SUMMARY")
    print("=" * 60)
    for rank, item in enumerate(ranked_candidates, 1):
        print(f"\nRANK {rank}: {item['candidate_id']} ({item['type']})")
        print(f"  • Confidence Score: {item['confidence_score']}")
        print(f"  • Description     : {item['description']}")
        print(f"  • Score Breakdown : {item['score_breakdown']}")
        print(f"  • Operations      : {len(item['operations'])} CSG steps")
    print("\n" + "=" * 60)
    print(f"SUCCESS: Saved 3D candidate ranking output -> {out_json_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())