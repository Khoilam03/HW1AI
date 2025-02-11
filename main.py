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
    """ Checks if a point (x, y, z) is inside any atomic sphere without double counting overlaps. """
    inside_any = False
    for atom in atoms:
        ax, ay, az, r = atom
        if (x - ax) ** 2 + (y - ay) ** 2 + (z - az) ** 2 <= r ** 2:
            inside_any = True
            break  # Exit early if a point is inside any sphere
    return inside_any

def estimate_protein_volume_fixed(file_path, num_samples=100000):
    """ Estimates protein volume using a Monte Carlo method with improved overlap handling. """
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

# Run the corrected volume estimation with improved overlap handling
pqr_filenames = [
    "protein.pqr", "single_sphere.pqr", "two_nonoverlapping.pqr", "two_overlapping_spheres.pqr", "cube_approx.pqr", "encapsulate_atom.pqr"
]
num_samples = 100000  # Monte Carlo sample size

for file_path in pqr_filenames:
    try:
        estimated_volume_fixed = estimate_protein_volume_fixed(file_path, num_samples=num_samples)
        print(f"Estimated protein volume for {file_path}: {estimated_volume_fixed:.2f} Å³")
    except ValueError as e:
        print(f"Error processing {file_path}: {str(e)}")