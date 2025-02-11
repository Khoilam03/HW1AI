import numpy as np

def read_pqr(file_path):
    """Reads atomic data from a PQR file and extracts (x, y, z, radius)."""
    atoms = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("ATOM"):
                parts = line.strip().split()
                try:
                    x, y, z = map(float, parts[5:8])  # Extract coordinates
                    r = float(parts[9])  # Extract radius
                    atoms.append((x, y, z, r))
                except ValueError:
                    print(f"Skipping malformed line: {line.strip()}")
    return atoms

def compute_bounding_box(atoms):
    """Computes the bounding box dimensions for the protein."""
    min_x = min(atom[0] - atom[3] for atom in atoms)
    max_x = max(atom[0] + atom[3] for atom in atoms)
    min_y = min(atom[1] - atom[3] for atom in atoms)
    max_y = max(atom[1] + atom[3] for atom in atoms)
    min_z = min(atom[2] - atom[3] for atom in atoms)
    max_z = max(atom[2] + atom[3] for atom in atoms)
    
    return (min_x, max_x, min_y, max_y, min_z, max_z)

def is_inside_protein(x, y, z, atoms):
    """Checks if a point (x, y, z) is inside any atomic sphere."""
    for ax, ay, az, r in atoms:
        if (x - ax) ** 2 + (y - ay) ** 2 + (z - az) ** 2 <= r ** 2:
            return True
    return False

def estimate_protein_volume(file_path, delta):
    """Estimates protein volume using a voxel-based approach."""
    atoms = read_pqr(file_path)
    if not atoms:
        raise ValueError("No atoms found in the input file.")

    min_x, max_x, min_y, max_y, min_z, max_z = compute_bounding_box(atoms)
    a, b, c = max_x - min_x, max_y - min_y, max_z - min_z  # Box dimensions

    # Define voxel grid
    x_steps = np.arange(min_x, max_x, delta)
    y_steps = np.arange(min_y, max_y, delta)
    z_steps = np.arange(min_z, max_z, delta)

    total_voxels = len(x_steps) * len(y_steps) * len(z_steps)
    inside_voxels = 0
    
    # Iterate over voxel centers
    for x in x_steps:
        for y in y_steps:
            for z in z_steps:
                if is_inside_protein(x + delta/2, y + delta/2, z + delta/2, atoms):
                    inside_voxels += 1
    
    # Compute estimated volume
    estimated_volume = (a * b * c) * (inside_voxels / total_voxels)
    return estimated_volume

# Example usage
if __name__ == "__main__":
    pqr_files = [
        "protein.pqr",
        "single_sphere.pqr",
        "two_nonoverlapping.pqr",
        "two_overlapping_spheres.pqr",
        "cube_approx.pqr"
    ]
    delta_value = 0.4  # Adjust voxel size
    
    for pqr_file in pqr_files:
        try:
            volume = estimate_protein_volume(pqr_file, delta_value)
            print(f"Estimated Protein Volume for {pqr_file}: {volume:.3f} cubic Angstroms")
        except ValueError as e:
            print(f"Error processing {pqr_file}: {str(e)}")
