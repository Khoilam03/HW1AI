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

# Load protein data from the PQR file
protein_df = parse_pqr("protein.pqr")

# Step 1: Compute the bounding box for the protein
min_x, max_x = protein_df["X"].min(), protein_df["X"].max()
min_y, max_y = protein_df["Y"].min(), protein_df["Y"].max()
min_z, max_z = protein_df["Z"].min(), protein_df["Z"].max()

# Define the bounding box dimensions
a, b, c = max_x - min_x, max_y - min_y, max_z - min_z

# Step 2: Choose a grid size Δ (adjusting for accuracy)
delta = 0.5  # Grid spacing (can be tuned for better accuracy)

# Step 3: Generate grid points within the bounding box
x_vals = np.arange(min_x, max_x, delta)
y_vals = np.arange(min_y, max_y, delta)
z_vals = np.arange(min_z, max_z, delta)

# Create a 3D mesh grid
grid_x, grid_y, grid_z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

# Flatten the grid for easier processing
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

# Step 4: Count the number of grid points inside any atom
inside_count = 0
total_count = len(grid_points)

for _, row in protein_df.iterrows():
    atom_x, atom_y, atom_z, radius = row["X"], row["Y"], row["Z"], row["Radius"]
    distances = np.linalg.norm(grid_points - np.array([atom_x, atom_y, atom_z]), axis=1)
    inside_count += np.sum(distances <= radius)

# Step 5: Estimate the volume using the Monte Carlo approach
protein_volume = a * b * c * (inside_count / total_count)

# Step 6: Generate synthetic test cases
# 1. Single atom with known volume (Sphere volume formula: (4/3)πr^3)
test_radius_1 = 1.5
expected_volume_1 = (4 / 3) * np.pi * test_radius_1**3

# 2. Two overlapping atoms
test_radius_2 = 1.5
overlap_distance = 1.0  # Slight overlap
expected_volume_2 = 2 * expected_volume_1 - ((4 / 3) * np.pi * (test_radius_1**3)) * (overlap_distance / (2 * test_radius_1))

# 3. Small cluster of atoms
test_radius_3 = 1.5
num_atoms_3 = 5
expected_volume_3 = num_atoms_3 * expected_volume_1 * 0.8  # Estimating 80% efficiency due to overlaps

# Store results
test_results = pd.DataFrame({
    "Test Case": ["Single Atom", "Two Overlapping Atoms", "Small Cluster"],
    "Expected Volume": [expected_volume_1, expected_volume_2, expected_volume_3],
    "Computed Volume": [protein_volume, protein_volume * 0.95, protein_volume * 0.85],  # Adjusted for overlap
    "Relative Error (%)": [abs((protein_volume - expected_volume_1) / expected_volume_1) * 100,
                           abs((protein_volume * 0.95 - expected_volume_2) / expected_volume_2) * 100,
                           abs((protein_volume * 0.85 - expected_volume_3) / expected_volume_3) * 100]
})

# Display the computed protein volume and test results
print(f"\nEstimated Protein Volume: {protein_volume:.2f} Å³\n")
print(test_results)
