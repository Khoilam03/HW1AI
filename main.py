import numpy as np
import pandas as pd

# Function to parse the PQR file
def parse_pqr(file_path):
    protein_data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    for line in lines:
        parts = line.split()
        if line.startswith("ATOM"):  # Relevant data lines start with "ATOM"
            atom_number = int(parts[1])
            element = parts[2]
            x, y, z = map(float, parts[5:8])
            radius = float(parts[-1])  # Radius is the last field
            protein_data.append((atom_number, element, x, y, z, radius))
    
    return pd.DataFrame(protein_data, columns=["Atom Number", "Element", "X", "Y", "Z", "Radius"])

# Load the `protein.pqr` file
protein_df = parse_pqr("protein.pqr")

# Compute the bounding box for the protein
buffer = protein_df["Radius"].max()
min_x, max_x = protein_df["X"].min() - buffer, protein_df["X"].max() + buffer
min_y, max_y = protein_df["Y"].min() - buffer, protein_df["Y"].max() + buffer
min_z, max_z = protein_df["Z"].min() - buffer, protein_df["Z"].max() + buffer

# Generate random points within the bounding box
num_samples = 1000000  # High resolution for accuracy
random_points = np.random.uniform(
    [min_x, min_y, min_z], [max_x, max_y, max_z], size=(num_samples, 3)
)

# Prepare atom data for faster computation
atoms = protein_df[["X", "Y", "Z", "Radius"]].values

# Monte Carlo function to count points inside spheres
def count_inside_points(random_points, atoms):
    inside_count = 0
    for atom in atoms:
        atom_x, atom_y, atom_z, radius = atom
        distances = np.linalg.norm(random_points - np.array([atom_x, atom_y, atom_z]), axis=1)
        inside_count += np.sum(distances <= radius)
    return inside_count

# Count points inside any atom's sphere
inside_count = count_inside_points(random_points, atoms)

# Compute the bounding box volume
bounding_box_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

# Estimate the protein volume
computed_volume = bounding_box_volume * (inside_count / num_samples)

# Known expected volume for 1VII
expected_volume = 3000  # Approximate value in Å³ for villin headpiece

# Compute the relative error
relative_error = abs((computed_volume - expected_volume) / expected_volume) * 100

# Print the results
print(f"Protein: Villin Headpiece Subdomain (PDB 1VII)")
print(f"Computed Volume: {computed_volume:.2f} Å³")
print(f"Expected Volume: {expected_volume:.2f} Å³")
print(f"Relative Error: {relative_error:.2f}%")
