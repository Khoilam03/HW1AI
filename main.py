import numpy as np

def read_pqr_atoms_fixed(file_path):
    """ Reads atomic data from a PQR file and extracts (x, y, z, radius). """
    atoms = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("ATOM"):
                parts = line.strip().split()
                try:
                    x, y, z = map(float, parts[5:8])  # Extract x, y, z coordinates
                    r = float(parts[-1])  # Extract radius (always the last column)
                    atoms.append((x, y, z, r))
                except ValueError:
                    print(f"Skipping malformed line: {line.strip()}")
    return atoms

def compute_bounding_box(atoms, margin=2.0):
    """ Computes the bounding box dimensions for the protein. """
    min_x = min(atom[0] - atom[3] for atom in atoms)
    max_x = max(atom[0] + atom[3] for atom in atoms)
    min_y = min(atom[1] - atom[3] for atom in atoms)
    max_y = max(atom[1] + atom[3] for atom in atoms)
    min_z = min(atom[2] - atom[3] for atom in atoms)
    max_z = max(atom[2] + atom[3] for atom in atoms)

    return (min_x - margin, max_x + margin, 
            min_y - margin, max_y + margin, 
            min_z - margin, max_z + margin)

def is_inside_atom(x, y, z, atoms):
    """ Checks if a point (x, y, z) is inside any atomic sphere. """
    for atom in atoms:
        ax, ay, az, r = atom
        if (x - ax) ** 2 + (y - ay) ** 2 + (z - az) ** 2 <= r ** 2:
            return True
    return False

def estimate_protein_volume_fixed(file_path, num_samples=100000):
    """ Estimates protein volume using a Monte Carlo voxel-based approach. """
    atoms = read_pqr_atoms_fixed(file_path)
    if not atoms:
        raise ValueError("No atoms found in the input file.")

    min_x, max_x, min_y, max_y, min_z, max_z = compute_bounding_box(atoms)

    # Compute bounding box volume
    box_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

    # Generate random sample points
    inside_count = 0
    for _ in range(num_samples):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        z = np.random.uniform(min_z, max_z)
        if is_inside_atom(x, y, z, atoms):
            inside_count += 1

    # Compute estimated protein volume
    protein_volume = box_volume * (inside_count / num_samples)
    return protein_volume

# Run the corrected volume estimation
file_path = "protein.pqr"  # Replace with the actual file path if needed
estimated_volume_fixed = estimate_protein_volume_fixed(file_path, num_samples=100000)

print(f"Estimated protein volume: {estimated_volume_fixed:.2f} Å³")
