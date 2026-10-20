// Auto-generated CAD model from Hand-Drawn Sketch Pipeline
$fn = 50;

difference() {
    union() {
        // Base Extrusion Block
        cube([100, 80, 20], center=true);
        // Top Pyramid / Frustum Protrusion
        translate([0, 0, 15])
            cylinder(h=15, r1=25, r2=10, center=true);
    }
    // Through-Hole Cutouts
    translate([-25, -15, 0])
        cylinder(h=50, r=8, center=true);
    translate([10, -20, 0])
        cylinder(h=50, r=10, center=true);
}
