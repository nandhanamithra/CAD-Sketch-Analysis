from pathlib import Path
import trimesh

# Output path
OUTPUT_DIR = Path("3d_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
stl_file_path = OUTPUT_DIR / "catia_working_model.stl"


def generate_stl():
    print("Generating watertight STL model for CATIA...")

    # 1. Create Base Extrusion Block
    base_block = trimesh.creation.box(extents=[100, 80, 20])

    # 2. Create Top Feature (Pyramid / Frustum Protrusion)
    try:
        top_feature = trimesh.creation.cone(radius=25, radius_top=10, height=15)
    except TypeError:
        top_feature = trimesh.creation.cylinder(radius=25, height=15)

    top_feature.apply_translation([0, 0, 17.5])

    # Combine Base and Top Feature
    solid_body = base_block.union(top_feature)

    # 3. Create Cylindrical Through-Hole Cutouts
    hole1 = trimesh.creation.cylinder(radius=8, height=50)
    hole1.apply_translation([-25, -15, 0])

    hole2 = trimesh.creation.cylinder(radius=10, height=50)
    hole2.apply_translation([10, -20, 0])

    # 4. Apply CSG Boolean Subtraction
    final_mesh = solid_body.difference([hole1, hole2])

    # 5. Export as STL Binary (CATIA Native Format)
    final_mesh.export(str(stl_file_path))

    print("=" * 60)
    print("SUCCESS: Generated STL file!")
    print(f"File Path: {stl_file_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    generate_stl()