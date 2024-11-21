# %% [markdown]
# # AidBio PPI Predictor
# **Author: Matiolli, CC, Ph.D.**
#
# The AidBio PPI is a graph neural network designed to predict protein-protein interactions (PPIs) of proteins represented as molecular graphs, graph neural networks (GNNs) and mutual attention. AidBio PPI is implemented using [PyTorch](https://github.com/pytorch/pytorch).
#
# ## Table of Contents
# 1. [Data Loading](#data-loading)
# 2. [Data Preprocessing](#data-preprocessing)
# 3. [Feature Engineering](#feature-engineering)
# 4. [Dataset Preparation](#dataset-preparation)
# 5. [Model Selection](#model-selection)
# 6. [Training and Validation](#training-and-validation)
# 7. [Evaluation](#evaluation)
# 8. [Conclusion](#conclusion)
#
# - **Key sub-steps:**
#   1. Parse PDBs: Extract residues residues and coordinates from PDB files.
#   2. Calculate Normalized Adjacency Matrix: Compute the normalized adjacency matrix for graph representation
#   3. Generate Fingerprints: Create molecular fingerprints (residue, neighbors) based on molecular distances.
#   4. Data Storage: Store the processed data in a structured format suitable for model training/testing
#
#

# %%
# File operations
import os
from os import walk
import random
import glob
from pathlib import Path
import pickle
import shutil
import gzip
import json
from typing import List, Optional, Tuple
from itertools import islice
from collections import Counter

# Data Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

# Process biological data
from Bio.PDB import PDBParser, is_aa

# Data structures
from collections import defaultdict
import pandas as pd

# Numerical computations
import numpy as np

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# Metrics and progress tracking
from tqdm.auto import tqdm
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
)
import wandb

# Set wandb notebook name environment variable
os.environ["WANDB_NOTEBOOK_NAME"] = "main_v3.ipynb"

# Set CUDA and PyTorch environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TORCH"] = torch.__version__

# Load configuration file (json)
with open("config.json", "r") as f:
    config = json.load(f)

# Set random seed for reproducibility
random_seed = config["environment"]["random_seed"]

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {device}")
# config

# %%
# Set project folders

# Existing folders
paths = config.get("paths")
localpdb_path = Path(paths["localpdb"])
datasets_path = Path(paths["datasets"])

# Folders to create
raw_path = Path(paths["raw"])
raw_path.mkdir(parents=True, exist_ok=True)

preprocessed_path = Path(paths["preprocessed"])
preprocessed_path.mkdir(parents=True, exist_ok=True)

logs_path = Path(paths["logs"])
logs_path.mkdir(parents=True, exist_ok=True)

model_checkpoint_path = Path(paths["model_checkpoint"])
model_checkpoint_path.mkdir(parents=True, exist_ok=True)


# %%
# Visualization
def plot_protein_3d(pdb_id, chain_id, residues, coords, acid_dict):
    """
    Visualize protein structure in 3D with Plotly, using residue names from acid_dict
    and assigning distinct colors per residue with a clear legend.

    Args:
        pdb_id (str): PDB ID of the protein.
        chain_id (str): Chain ID of the protein.
        residues (np.ndarray): Array of residue indices (numeric).
        coords (np.ndarray): Array of 3D coordinates for residues.
        acid_dict (dict): Dictionary mapping residue 3-letter codes to indices.
    """
    # Create reverse mapping from indices to 3-letter codes
    reverse_acid_dict = {v: k for k, v in acid_dict.items() if isinstance(v, int)}

    # Convert residues to their 3-letter codes
    residue_labels = [reverse_acid_dict.get(res, "UNK") for res in residues]

    # Assign one color per residue
    unique_residues = sorted(set(residues))
    num_colors = len(unique_residues)
    color_map = px.colors.qualitative.Set3  # A good discrete colormap
    colors = color_map * (num_colors // len(color_map) + 1)  # Repeat colors if needed
    residue_to_color = {res: colors[i] for i, res in enumerate(unique_residues)}

    fig = go.Figure()

    # Add separate traces for each residue type
    for res in unique_residues:
        indices = [i for i, r in enumerate(residues) if r == res]
        fig.add_trace(
            go.Scatter3d(
                x=coords[indices, 0],
                y=coords[indices, 1],
                z=coords[indices, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=residue_to_color[res],  # Assign color for this residue
                ),
                name=f"{reverse_acid_dict.get(res, 'UNK')} ({res})",  # Add to legend
                hoverinfo="text",
                text=[f"{reverse_acid_dict.get(res, 'UNK')} ({res})"] * len(indices),
            )
        )

    # Add lines connecting residues in sequence
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="lines",
            line=dict(
                color="gray",
                width=2,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Layout updates
    fig.update_layout(
        title=f"Protein Structure ({pdb_id}-{chain_id})",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        legend=dict(
            title="Amino Acid Residues",
            font=dict(size=10),
        ),
    )

    fig.show()


# %% [markdown]
# ## Data Loading
#
# Load the data needed to train and evaluate the PPI predictor. The data consists of:
#
# 1. Protein 3D structures (PDB files)
# 2. Protein Interaction Data (Ground truth)
#
# ---
# >NOTE: Make sure that the localpdb folder is up-to-date. Please note that the update may take time to complete if the localpdb folder is several versions behind.
#
# *PDB data mirrors*:
# - rcsb: RCSB PDB - US - **default**
# - pdbe: (PDBe - UK)
# - pdbj: (PDBj - Japan)  - **best**
#

# %%
# !localpdb_setup -mirror pdbj -db_path /media/clever/aidbio/data/localpdb --update

# %%
# List PDBs available in the localpdb mirror
localpdb_pdbs = [
    f.stem[-8:-4].lower() for f in localpdb_path.glob("**/*.gz") if f.is_file()
]

print(f"Number of available PDBs in localpdb mirror: {len(localpdb_pdbs)}")

# %%
# Load pdb-chain list
filepath = datasets_path / "list_of_prots.txt"
pdb_chain_df = pd.read_csv(filepath, sep="\t", header=None)

# Rename columns for consistency
pdb_chain_df.columns = ["uniprot_id", "pdb_id", "chain_id"]

# Lowercase PDB identifiers for consistency
pdb_chain_df["pdb_id"] = pdb_chain_df["pdb_id"].str.lower()

# Check if all PDBs in the chain list have available PDBs
pdb_chain_df = pdb_chain_df[pdb_chain_df["pdb_id"].isin(localpdb_pdbs)]
pdb_chain_df.reset_index(drop=True, inplace=True)

# Extract PDB ID and Chain ID and process them into tuples
pdb_chains = pdb_chain_df.iloc[:, 1:].values
pdb_chain_list = [(pdb_id, chain_id) for pdb_id, chain_id in pdb_chains]

print(f"Number of proteins with available PDBs: {len(pdb_chain_list)}")
print(pdb_chain_list[:10])
pdb_chain_df

# %%
# Load interactions
filepath = datasets_path / "interactions_data.txt"
interactions_df = pd.read_csv(filepath, sep="\t", header=None)
interactions_df.columns = ["pdb_id1", "pdb_id2", "label"]

# Convert PDB IDs to lowercase for consistency
interactions_df["pdb_id1"] = interactions_df["pdb_id1"].str.lower()
interactions_df["pdb_id2"] = interactions_df["pdb_id2"].str.lower()

# Filter interactions to include only those with proteins that have available PDBs
interactions_df = interactions_df[
    interactions_df["pdb_id1"].isin(localpdb_pdbs)
    & interactions_df["pdb_id2"].isin(localpdb_pdbs)
]

interactions_df.reset_index(drop=True, inplace=True)

print(f"Interactions shape: {interactions_df.shape}")
print(f"{interactions_df['label'].value_counts()}")
interactions_df

# %%
# Count duplicated proteins
pdb_counts = pd.concat(
    [interactions_df["pdb_id1"], interactions_df["pdb_id2"]],
    axis=0,
    ignore_index=True,
).value_counts()

pdb_counts[pdb_counts > 50]


# %%
def get_pdbs(
    pdbs_list: None,
    localpdb_folder: Path,
    raw_folder: Path,
) -> None:
    """Copy PDB files from localpdb folder and decompress them into a folder.

    Args:
        pdbs_list (List[str]): List of PDB IDs to fetch.
        localpdb_folder (Optional(Path)): Path to the local PDB folder. Defaults to None.
        dst_folder (Path, optional): Destination folder for the downloaded PDB files. Defaults to './data/pdbs'

    # Returns:
        None
    """
    # Ensure destination folder exists
    raw_folder.mkdir(parents=True, exist_ok=True)

    # List PDB files in localpdb folder mathcing the pdbs_list
    files_to_copy = [
        f
        for f in localpdb_folder.glob("**/*.gz")
        if f.is_file() and f.stem[-8:-4].lower() in pdbs_list
    ]
    print(f"Number of PDB files to copy: {len(files_to_copy)}")

    # Copy files
    copied_files_count = 0
    for file in tqdm(
        files_to_copy,
        total=len(files_to_copy),
        desc="Copying files",
    ):
        shutil.copy(file, raw_folder)
        copied_files_count += 1

    # Decompress copied files
    decompressed_files_count = 0
    for file in tqdm(
        raw_folder.iterdir(),
        total=copied_files_count,
        desc="Decompressing files",
    ):
        if not file.is_file() or file.suffix != ".gz":
            continue
        with gzip.open(file, "rb") as gz:
            with open(raw_folder / f"{file.stem[3:7]}.pdb", "wb") as out:
                out.writelines(gz)
        decompressed_files_count += 1
        file.unlink()  # Remove the original .gz file after decompression

    # Check for missing PDBs
    dst_list_pdbs = {
        f.stem.split(".")[0][-4:]
        for f in raw_folder.iterdir()
        if f.is_file() and f.suffix == ".pdb"
    }

    missing_pdbs = [pdb for pdb in pdbs_list if pdb not in dst_list_pdbs]

    # Write missing PDB ids to a file
    missing_local_pdbs_file = logs_path / "missing_localpdb_pdbs.txt"
    with open(missing_local_pdbs_file, "w") as f:
        for pdb in missing_pdbs:
            f.write(",".join(missing_pdbs))

    print(f"# of files found: {len(files_to_copy)}")
    print(f"{copied_files_count} files copied to {raw_folder}")
    print(f"{decompressed_files_count} files decompressed in {raw_folder}")
    print(f"# of missing files: {len(missing_pdbs)}")

    return None


# %%
# Extract path from config file
get_pdbs(
    pdbs_list=[pdb_id for pdb_id, _ in pdb_chain_list],
    localpdb_folder=localpdb_path,
    raw_folder=raw_path,
)

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### Parsing Protein 3D Structures
#
# - **Purpose:** Parses a PDB file to extract the residues and their alpha-carbon (CA) coordinates.
# - **Residues:** Converts residue names to unique integers using acid_dict.
# - **Coordinates:** Collects the 3D coordinates of the CA atoms.
#

# %%
# Extract params from config file
params = config["preprocessing_params"]
aminoacids = params["aminoacids"]
max_residues = params["max_residues"]
mol_treshold_distance = params["mol_threshold_distance"]
fingerprint_radius = params["fingerprint_radius"]

# Amino acid dictionary for unique integer conversion (1-letter code to integer)
acid_dict = defaultdict(lambda: len(acid_dict))

# Initialize dict to store fingerprints
fingerprint_dict = defaultdict(lambda: np.uint32(len(fingerprint_dict)))


# %%
def parse_protein_structure(
    pdb_file: Path, chain_id: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a PDB file, extract CA residues and their coordinates, and return as structured data.

    Args:
        pdb_file (Path): Path to the PDB file to parse.
        chain_id (str, optional): Specific chain to extract, defaults to None (all chains).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - Residues encoded as integers (based on `acid_dict`).
            - Coordinates of the CA atoms for each residue.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file.stem, pdb_file)

    # Initialize lists to hold residues and coordinates
    residues = []
    coords = []

    # Iterate through the structure and extract CA atoms for valid residues
    for model in structure:
        for chain in model:
            if chain_id and chain.id != chain_id:
                continue
            for residue in chain:
                if is_aa(residue, standard=True) and "CA" in residue:
                    residues.append(acid_dict[residue.resname])
                    coords.append(residue["CA"].coord)

    # Convert lists to numpy arrays for further processing
    return np.array(residues), np.array(coords)


# %%
# Test parse_protein_structure
pdb_id = "1u5x"
chain_id = "A"
residues, coords = parse_protein_structure(
    Path(raw_path / f"{pdb_id}.pdb"), chain_id="A"
)
plot_protein_3d(pdb_id, chain_id, residues, coords, acid_dict)
print("acid dict length:", len(acid_dict))
print("acid dict:\n", dict(acid_dict))
print("residues shape:", residues.shape)
print("some residues:\n", residues[:50])
print("coords shape:", coords.shape)
print("coords:\n", coords)

# %%
# Test parse_protein_structure
pdb_id = "4zfo"
chain_id = "F"
residues, coords = parse_protein_structure(
    Path(raw_path / f"{pdb_id}.pdb"), chain_id="A"
)
plot_protein_3d(pdb_id, chain_id, residues, coords, acid_dict)
print("acid dict length:", len(acid_dict))
print("acid dict:\n", dict(acid_dict))
print("residues shape:", residues.shape)
print("some residues:\n", residues[:50])
print("coords shape:", coords.shape)
print("coords:\n", coords)

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# ### Calculating Adjancency Matrix
#
# - **Purpose:** Creates an adjacency matrix where residues are connected if they are within a threshold distance.
# - **Symmetry:** The adjacency matrix is symmetric since the distance between i and j is the same as between j and i.
# - **Adding self-loops**: Set the diagonal to 1 in the adjacency matrix.
# - **Calculating the degree matrix**: Compute the sum of adjacency values for each node.
# - **Normalizing**: Compute $D^{-1/2} A D^{-1/2}$, where $D$ is the degree matrix.
#
#

# %%
# GPU computation of adj matrix


def calculate_adjacency(
    coords: np.ndarray,
    max_residues: int = max_residues,
    threshold_distance: float = mol_treshold_distance,
    device: str = device,
) -> np.ndarray:
    """Calculate the adjacency matrix for residues based on spatial distance using PyTorch on GPU.

    Args:
        coords (np.ndarray): Coordinates of residues, shaped (num_residues, 3).
        max_residues (int, optional): Maximum number of residues to consider. Defaults to MAX_RESIDUES.
        threshold_distance (float, optional): Distance threshold for adjacency. Defaults to THRESHOLD_DISTANCE.
        device (str, optional): The device to run computations on ("cuda" or "cpu"). Defaults to "cuda".

    Returns:
        np.ndarray: The normalized adjacency matrix, shaped (num_residues, num_residues).
    """

    # Move data to GPU if available
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    coords = torch.tensor(
        coords[:max_residues],
        dtype=torch.float32,
        device=device,
    )

    # Calculate pairwise distances
    num_res = coords.shape[0]
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # Shape: (num_res, num_res, 3)
    distances = torch.norm(diff, dim=2)  # Shape: (num_res, num_res)

    # Apply threshold to determine adjacency (1 if within threshold, 0 otherwise)
    adj = (distances <= threshold_distance).float()

    # Move back to CPU if you need it as a NumPy array
    # Slice the adjacency matrix to max_residues
    adj = adj[:max_residues, :max_residues]

    # Add self-loops
    adj.fill_diagonal_(1.0)

    # Degree matrix: sum over rows (number of adjacent nodes for each node)
    degree = torch.sum(adj, dim=1)  # Sum of each row to get the degree for each residue

    # Avoid division by zero by setting non-positive degrees to 1
    degree = torch.where(degree > 0, degree, torch.ones_like(degree))
    degree_sqrt_inv = torch.diag(1.0 / torch.sqrt(degree))

    # Compute normalized adjacency matrix
    adj = degree_sqrt_inv @ adj @ degree_sqrt_inv

    return adj.cpu().numpy()  # Move back to CPU if you need it as a NumPy array


# %%
adj = calculate_adjacency(coords)
print("adj_norm shape:", adj.shape)
print("adj_norm:\n", adj)

# %% [markdown]
# ### Calculating Fingerprints
#
# - **Purpose:** Generates a unique fingerprint for each residue based on its type and neighbors.
# - **Neighbors:** For each residue, collects the types of adjacent residues.
# - **Fingerprint:** Combines the residue and its neighbors into a tuple, which is then assigned a unique integer.
#

# %%
# GPU computation of fingerprints


def create_fingerprints(
    residues: np.ndarray,
    adj: np.ndarray,
    max_residues: int = max_residues,
    threshold: float = 1e-5,
    device: str = device,
) -> np.ndarray:
    """Generate fingerprints using a Weisfeiler-Lehman-like algorithm with PyTorch on GPU.

    Args:
        residues (np.ndarray): Array of residues represented by integers.
        adj (np.ndarray): Adjacency matrix representing connections between residues.
        radius (int, optional): Radius of neighborhood for fingerprinting. Defaults to FINGERPRINT_RADIUS.
        threshold (float, optional): Threshold for adjacency value to consider a connection. Defaults to 1e-5.
        device (str, optional): Device for computation ("cuda" or "cpu"). Defaults to "cuda".

    Returns:
        np.ndarray: Array of generated fingerprints, with each residue having a corresponding fingerprint.
    """

    # Move data to GPU if available
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Limit the number of residues if <= MAX_RESIDUES
    if len(residues) > max_residues:
        residues = residues[:max_residues]
        adj = adj[:max_residues, :max_residues]

    # Convert to PyTorch tensors and move to GPU
    residues = torch.tensor(residues, dtype=torch.int32, device=device)
    adj = torch.tensor(adj, dtype=torch.float16, device=device)
    threshold = torch.tensor(threshold, device=device)

    fingerprints = []

    # Process each residue and calculate fingerprints on the GPU
    for i, residue in enumerate(residues):
        # Collect neighbor indices where adjacency values are above threshold
        neighbors_indices = torch.nonzero(adj[i] > threshold, as_tuple=True)[0]

        # Handle single residues or no neighbors cases
        if len(neighbors_indices) == 0:
            neighbors = ()
        else:
            # Limit neighbors to available residues and convert to integers
            neighbors = tuple(residues[neighbors_indices].tolist())

        # Create a unique fingerprint using the residue and neighbors
        fingerprint = (int(residue), neighbors)
        fingerprints.append(fingerprint_dict[fingerprint])

    # Convert fingerprints to numpy array on CPU
    fingerprints = (
        torch.tensor(fingerprints, dtype=torch.int32, device=device).cpu().numpy()
    )
    return fingerprints


# %%
fingerprints = create_fingerprints(residues, adj)
print("fingerprints shape:", fingerprints.shape)
print("fingerprints:\n", fingerprints[:50])

# %% [markdown]
# ### Creating the Protein Dataset


# %%
# Process a single protein
def process_protein(pdb_file, chain_id=None):
    # Ensure path object
    pdb_file = Path(pdb_file)

    # Parse protein structure
    residues, coords = parse_protein_structure(pdb_file, chain_id)

    # Calculate normalized adj matrix
    adj_norm = calculate_adjacency(coords)

    # Generate fingerprints
    fingerprints = create_fingerprints(residues, adj_norm)

    # Convert to PyTorch tensors
    fingerprints_tensor = torch.tensor(fingerprints, dtype=torch.long)
    adjacency_tensor = torch.tensor(adj_norm, dtype=torch.float)

    return fingerprints_tensor, adjacency_tensor


def process_protein_list(
    pdb_chains: list, path_to_pdbs=raw_path, output_dir=preprocessed_path
):
    # Ensures Path object
    path_to_pdbs = Path(path_to_pdbs)

    # Create output dir to save the dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_dict = {}  # Initializes dict to store protein fingerprints and adj_norm

    for pdb_id, chain_id in tqdm(
        pdb_chains, total=len(pdb_chains), desc="Processing PDB files..."
    ):
        try:
            fingerprints_tensor, adjacency_tensor = process_protein(
                f"{path_to_pdbs / pdb_id}.pdb", chain_id
            )
        except Exception as e:
            print(f"Error processing fingerprints of {pdb_id}, chain {chain_id}: {e}")
            continue

        try:
            protein_dict[pdb_id] = {
                "fingerprints": fingerprints_tensor,
                "adjacency": adjacency_tensor,
            }

        except Exception as e:
            print(f"Error processing protein dict of {pdb_id}, chain {chain_id}: {e}")

    torch.save(protein_dict, output_dir / "protein_data_dict.pt")

    return protein_dict


# %%
# Process protein datset
protein_data_dict = process_protein_list(pdb_chains=pdb_chain_list)

# Save fingerprints dictionary (vocabulary)
filepath = preprocessed_path / "fingerprints_dict.pkl"
with open(filepath, "wb") as f:
    pickle.dump(dict(fingerprint_dict), f)

print(f"Protein data length: {len(protein_data_dict)}")
print(f"Length of the fingerprint dictionary: {len(fingerprint_dict)}")

# %%
# Load protein data
protein_data_dict_file = preprocessed_path / "protein_data_dict.pt"
protein_data_dict = torch.load(protein_data_dict_file, weights_only=True)
print(f"Protein data length: {len(protein_data_dict)}")

# %%
# Load fingerprint dictionary
fingerprint_dict_file = preprocessed_path / "fingerprints_dict.pkl"
with open(fingerprint_dict_file, "rb") as f:
    fingerprint_dict = pickle.load(f)

print(f"Length of the fingerprint dictionary: {len(fingerprint_dict)}")
for key, value in islice(fingerprint_dict, 10):
    print(key, value)

# %% [markdown]
# ## Dataset Preparation

# %%
# Combine pdb_id1 and pdb_id2 into a single list to count occurrences

data = interactions_df.copy()

all_proteins = pd.concat([data["pdb_id1"], data["pdb_id2"]])
protein_counts = all_proteins.value_counts()

# Create a DataFrame to track proteins and their counts
protein_df = protein_counts.reset_index()
protein_df.columns = ["protein", "count"]

# Bin the counts into categories (e.g., low, medium, high frequency)
bins = [0, 1, 3, 10, float("inf")]  # Adjust bin ranges as needed
labels = ["rare", "low", "medium", "high"]
protein_df["count_bin"] = pd.cut(protein_df["count"], bins=bins, labels=labels)

# Stratify proteins by these bins
train_proteins, temp_proteins = train_test_split(
    protein_df["protein"],
    test_size=0.4,
    stratify=protein_df["count_bin"],
    random_state=42,
)
val_proteins, test_proteins = train_test_split(
    temp_proteins, test_size=0.5, random_state=42
)

# Convert to sets
train_set, val_set, test_set = (
    set(train_proteins),
    set(val_proteins),
    set(test_proteins),
)


# Helper function to check if an interaction is valid for a set
def belongs_to_set(row, protein_set):
    return row["pdb_id1"] in protein_set and row["pdb_id2"] in protein_set


# Split the data into train, val, and test based on protein assignments
train_ppi = data[data.apply(lambda row: belongs_to_set(row, train_set), axis=1)]
val_ppi = data[data.apply(lambda row: belongs_to_set(row, val_set), axis=1)]
test_ppi = data[data.apply(lambda row: belongs_to_set(row, test_set), axis=1)]

# Ensure non-overlapping
assert train_ppi["pdb_id1"].isin(val_set).sum() == 0
assert train_ppi["pdb_id2"].isin(val_set).sum() == 0
assert val_ppi["pdb_id1"].isin(test_set).sum() == 0
assert val_ppi["pdb_id2"].isin(test_set).sum() == 0

# Print the sizes of each split
print(
    f"Train size: {len(train_ppi)}, Val size: {len(val_ppi)}, Test size: {len(test_ppi)}"
)

# %%
# Acummulate data into a list


def create_data(df, protein_data_dict):
    """
    Create a dataset from the given interaction DataFrame and protein data dictionary,
    including the original DataFrame index for mapping back.
    """
    data_list = []  # Initialize list to store input data

    for idx, row in df.iterrows():  # Use the DataFrame's index
        pdb_id1 = row["pdb_id1"]
        pdb_id2 = row["pdb_id2"]
        label = row["label"]

        if pdb_id1 in protein_data_dict and pdb_id2 in protein_data_dict:
            fp1 = protein_data_dict[pdb_id1]["fingerprints"]
            adjacency1 = protein_data_dict[pdb_id1]["adjacency"]

            fp2 = protein_data_dict[pdb_id2]["fingerprints"]
            adjacency2 = protein_data_dict[pdb_id2]["adjacency"]

            target_label = torch.tensor([label], dtype=torch.long)

            # Append to data list with index
            data_list.append((fp1, adjacency1, fp2, adjacency2, target_label, idx))
        else:
            print(f"Warning: Missing data for {pdb_id1} or {pdb_id2}")
    return data_list


train_data = create_data(train_ppi, protein_data_dict)
val_data = create_data(val_ppi, protein_data_dict)
test_data = create_data(test_ppi, protein_data_dict)

print(f"Train data length: {len(train_data)}")
print(f"Val data length: {len(val_data)}")
print(f"Test data length: {len(test_data)}")


# %%
def collate_fn(batch):
    """
    Collate function to pad fingerprints and adjacency matrices for two proteins
    and stack them into batches.
    """
    fp1_list, adj1_list, fp2_list, adj2_list, labels = zip(*batch)

    # Determine max lengths for padding
    max_len1 = max(fp.size(0) for fp in fp1_list)
    max_len2 = max(fp.size(0) for fp in fp2_list)

    # Pad fingerprints and adjacency matrices
    batch_fp1 = []
    batch_adj1 = []
    batch_fp2 = []
    batch_adj2 = []
    for fp1, adj1, fp2, adj2 in zip(fp1_list, adj1_list, fp2_list, adj2_list):
        # Pad fingerprints
        pad_size1 = max_len1 - fp1.size(0)
        pad_size2 = max_len2 - fp2.size(0)
        padded_fp1 = F.pad(fp1, (0, pad_size1), value=0)
        padded_fp2 = F.pad(fp2, (0, pad_size2), value=0)
        batch_fp1.append(padded_fp1)
        batch_fp2.append(padded_fp2)

        # Pad adjacency matrices
        pad_adj1 = (0, pad_size1, 0, pad_size1)
        pad_adj2 = (0, pad_size2, 0, pad_size2)
        padded_adj1 = F.pad(adj1, pad_adj1, value=0)
        padded_adj2 = F.pad(adj2, pad_adj2, value=0)
        batch_adj1.append(padded_adj1)
        batch_adj2.append(padded_adj2)

    # Stack tensors
    batch_fp1 = torch.stack(batch_fp1)
    batch_adj1 = torch.stack(batch_adj1)
    batch_fp2 = torch.stack(batch_fp2)
    batch_adj2 = torch.stack(batch_adj2)
    labels = torch.tensor(labels, dtype=torch.float32)

    return (batch_fp1, batch_adj1, batch_fp2, batch_adj2), labels


# %%
def collate_fn(batch):
    """
    Custom collate function to pad fingerprints and adjacency matrices for two proteins
    and stack them into batches.
    """
    # Unpack six elements, including the index
    fp1_list, adj1_list, fp2_list, adj2_list, labels, indices = zip(*batch)

    # Determine max lengths for padding
    max_len1 = max(fp.size(0) for fp in fp1_list)
    max_len2 = max(fp.size(0) for fp in fp2_list)

    # Pad fingerprints and adjacency matrices
    batch_fp1 = []
    batch_adj1 = []
    batch_fp2 = []
    batch_adj2 = []
    for fp1, adj1, fp2, adj2 in zip(fp1_list, adj1_list, fp2_list, adj2_list):
        # Pad fingerprints
        pad_size1 = max_len1 - fp1.size(0)
        pad_size2 = max_len2 - fp2.size(0)
        padded_fp1 = F.pad(
            fp1, (0, pad_size1), value=0.0
        )  # Pad with 0 for fingerprints
        padded_fp2 = F.pad(fp2, (0, pad_size2), value=0.0)
        batch_fp1.append(padded_fp1)
        batch_fp2.append(padded_fp2)

        # Pad adjacency matrices
        pad_adj1 = (0, pad_size1, 0, pad_size1)
        pad_adj2 = (0, pad_size2, 0, pad_size2)
        padded_adj1 = F.pad(
            adj1, pad_adj1, value=0.0
        )  # Pad with 0 for adjacency matrices
        padded_adj2 = F.pad(adj2, pad_adj2, value=0.0)
        batch_adj1.append(padded_adj1)
        batch_adj2.append(padded_adj2)

    # Stack tensors
    batch_fp1 = torch.stack(batch_fp1)
    batch_adj1 = torch.stack(batch_adj1)
    batch_fp2 = torch.stack(batch_fp2)
    batch_adj2 = torch.stack(batch_adj2)
    labels = torch.cat(
        labels
    ).long()  # Ensure labels are tensors and have the correct dtype

    # Return the indices along with the batch if needed
    return (batch_fp1, batch_adj1, batch_fp2, batch_adj2), labels, indices


# %%
class PPIDataset(Dataset):
    def __init__(self, data_list):
        """
        Initialize the dataset with a list of tuples containing:
        (fp1, adj1, fp2, adj2, label, index).
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fp1, adj1, fp2, adj2, label, index = self.data[idx]
        return fp1, adj1, fp2, adj2, label, index


train_dataset = PPIDataset(train_data)
val_dataset = PPIDataset(val_data)
test_dataset = PPIDataset(test_data)
print(f"Length of dataset: {train_dataset.__len__()}")
print(f"Length of validation dataset: {val_dataset.__len__()}")
print(f"Length of test dataset: {test_dataset.__len__()}")

# %%
# Accessing a sample
sample_idx = 0  # Replace with the index you want to sample
fp1, adj1, fp2, adj2, label, original_idx = train_dataset[sample_idx]

# Get PPI information from interactions DataFrame
sample = train_ppi.loc[original_idx].values
pdb_id1 = sample[0]
pdb_id2 = sample[1]
chain_id1 = pdb_chain_df[pdb_chain_df["pdb_id"] == pdb_id1]["chain_id"].values
chain_id2 = pdb_chain_df[pdb_chain_df["pdb_id"] == pdb_id2]["chain_id"].values

# Print sample interaction
print(f"Dataset Sample Index: {sample_idx}")
print(f"Original DataFrame Index: {original_idx}")
print(f"Sample FP1: {fp1.shape}")
print(f"Sample Adj1: {adj1.shape}")
print(f"Sample FP2: {fp2.shape}")
print(f"Sample Adj2: {adj2.shape}")
print(f"Sample Label: {label}\n")
print(f"PDB id 1: {pdb_id1}, chain id 1: {chain_id1}")
print(f"PDB id 2: {pdb_id2}, chain id 2: {chain_id2}")
print(sample)

# %%
# # Split dataset into train and validation sets
# dataset_size = len(dataset)
# val_size = int(0.2 * dataset_size)
# test_size = int(0.2 * dataset_size)
# train_size = dataset_size - val_size - test_size
# batch_size = 4

# train_dataset, val_dataset, test_dataset = random_split(
#     dataset, [train_size, val_size, test_size]
# )
# print(f"train dataset length: {len(train_dataset)}")
# print(f"Validation dataset length: {len(val_dataset)}")
# print(f"Test dataset length: {len(test_dataset)}")

# %%
# Create DataLoaders with the custom collate function

# Get configuration parameters
batch_size = config["wandb_config"]["batch_size"]
print(f"batch_size: {batch_size}")

# Create DataLoaders with the custom collate function
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

print(f"Length Training: {len(train_loader) * batch_size}")
print(f"Lenght Validation: {len(val_loader) * batch_size}")
print(f"Length Testing: {len(test_loader) * batch_size}")
print(
    f"Total Train + Validation + Test: {(len(train_loader) + len(val_loader) + len(test_loader)) * batch_size}"
)

# %%
# Iterate through loaders
for (
    batch_fp1,
    batch_adj1,
    batch_fp2,
    batch_adj2,
    batch_labels,
    idx,
) in train_loader.dataset.data:
    print(f"FP1 batch shape: {batch_fp1.shape}")
    print(f"Adj1 batch shape: {batch_adj1.shape}")
    print(f"FP2 batch shape: {batch_fp2.shape}")
    print(f"Adj2 batch shape: {batch_adj2.shape}")
    print(f"Labels batch shape: {batch_labels.shape}")
    print()
    break

for (
    batch_fp1,
    batch_adj1,
    batch_fp2,
    batch_adj2,
    batch_labels,
    idx,
) in val_loader.dataset.data:
    print(f"Validation FP1 batch shape: {batch_fp1.shape}")
    print(f"Validation Adj1 batch shape: {batch_adj1.shape}")
    print(f"Validation FP2 batch shape: {batch_fp2.shape}")
    print(f"Validation Adj2 batch shape: {batch_adj2.shape}")
    print(f"Validation Labels batch shape: {batch_labels.shape}")
    print()
    break

for (
    batch_fp1,
    batch_adj1,
    batch_fp2,
    batch_adj2,
    batch_labels,
    idx,
) in test_loader.dataset.data:
    print(f"Test FP1 batch shape: {batch_fp1.shape}")
    print(f"Test Adj1 batch shape: {batch_adj1.shape}")
    print(f"Test FP2 batch shape: {batch_fp2.shape}")
    print(f"Test Adj2 batch shape: {batch_adj2.shape}")
    print(f"Test Labels batch shape: {batch_labels.shape}")
    print()
    break

# %%
# Check label distribution

# Initialize lists to store labels
train_labels = []
val_labels = []
test_labels = []

# Initialize lists to store indices
train_indices = []
val_indices = []
test_indices = []

# Iterate through datasets to get labels
for (
    batch_fp1,
    batch_adj1,
    batch_fp2,
    batch_adj2,
    batch_labels,
    idx,
) in train_loader.dataset.data:

    train_labels.extend(batch_labels.tolist())
    train_indices.append(idx)


for (
    batch_fp1,
    batch_adj1,
    batch_fp2,
    batch_adj2,
    batch_labels,
    idx,
) in val_loader.dataset.data:
    val_labels.extend(batch_labels.tolist())
    val_indices.append(idx)

for (
    batch_fp1,
    batch_adj1,
    batch_fp2,
    batch_adj2,
    batch_labels,
    idx,
) in test_loader.dataset.data:
    test_labels.extend(batch_labels.tolist())
    test_indices.append(idx)

# Count occurrences of each label
train_label_counts = Counter(train_labels)
val_label_counts = Counter(val_labels)
test_label_counts = Counter(test_labels)

print(f"Label counts in Training: {train_label_counts}")
print(f"Label counts in Validation: {val_label_counts}")
print(f"Label counts in Testing: {test_label_counts}")

# Compute the total number of samples in each dataset
train_total_samples = sum(train_label_counts.values())
val_total_samples = sum(val_label_counts.values())
test_total_samples = sum(test_label_counts.values())

print(f"Total samples in Training: {train_total_samples}")
print(f"Total samples in Validation: {val_total_samples}")
print(f"Total samples in Testing: {test_total_samples}")

# %%
# Get first sample from each dataset
sample_index = 0
train_sample_index = train_indices[sample_index]
val_sample_index = val_indices[sample_index]
test_sample_index = test_indices[sample_index]

# Load the corresponding protein structures
train_sample_pdb1 = str(train_ppi.loc[train_sample_index]["pdb_id1"])
train_sample_pdb2 = train_ppi.loc[train_sample_index]["pdb_id2"]
train_sample_pdb1, train_sample_pdb2

# %%
# Plot proteins in the sample interaction

residues, coords = parse_protein_structure(Path(raw_path / f"3blv.pdb"), chain_id="D")
plot_protein_3d(pdb_id1, chain_id1, residues, coords, acid_dict)
print(f"PDB id 1: {pdb_id1}, chain id 1: {chain_id1}")
print(f"Residues shape: {residues.shape}")
print("some residues:\n", residues[:20])
print("coords shape:", coords.shape)


# Plot proteins in the other sample interaction
residues, coords = parse_protein_structure(Path(raw_path / f"4bsz.pdb"), chain_id="A")
plot_protein_3d(pdb_id2, chain_id2, residues, coords, acid_dict)
print(f"\nPDB id 2: {pdb_id2}, chain id 2: {chain_id2}")
print(f"Residues shape: {residues.shape}")
print("some residues:\n", residues[:20])
print("coords shape:", coords.shape)

# %% [markdown]
# ## Model Selection
#
# **Model Definition:** Implement the Graph Convolutional Network (GCN) without messaging, with mutual attention of both protein-protein interactors and z_score of the interaction.
#

# %%
# Get model and training parameters

wandb_config = config["wandb_config"]
wandb_config

# %% [markdown]
# ### With Normalization

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPIPredictor(nn.Module):
    """Protein-Protein Interaction Predictor using GNN and Mutual Attention."""

    def __init__(self, num_fingerprints, emb_size, num_gnn_layers, dropout_prob=0.5):
        """
        Initializes the PPIPredictor model.

        Args:
            num_fingerprints (int): The number of unique fingerprints (input size).
            emb_size (int): Dimension of the embedding vectors.
            num_gnn_layers (int): Number of GNN layers.
            dropout_prob (float): Dropout probability (default: 0.5).
        """
        super(PPIPredictor, self).__init__()
        self.emb_size = emb_size
        self.num_gnn_layers = num_gnn_layers
        self.dropout_prob = dropout_prob

        # Embedding layer for fingerprints
        self.fingerprint_embeddings = nn.Embedding(num_fingerprints, emb_size)
        nn.init.xavier_normal_(self.fingerprint_embeddings.weight)

        # Dropout and batch norm for embeddings
        self.dropout = nn.Dropout(p=dropout_prob)
        self.batch_norm_emb = nn.BatchNorm1d(emb_size)

        # GNN layers
        self.gnn_layers = nn.ModuleList(
            [nn.Linear(emb_size, emb_size) for _ in range(num_gnn_layers)]
        )
        self.batch_norm_gnn = nn.ModuleList(
            [nn.BatchNorm1d(emb_size) for _ in range(num_gnn_layers)]
        )
        for layer in self.gnn_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Attention layers
        self.attn_weights_1 = nn.Linear(emb_size, emb_size)
        self.attn_weights_2 = nn.Linear(emb_size, emb_size)
        nn.init.xavier_normal_(self.attn_weights_1.weight)
        nn.init.xavier_normal_(self.attn_weights_2.weight)

        # Attention vector
        self.attn_vector = nn.Parameter(torch.randn(emb_size))

        # Output layer
        self.output_fc = nn.Linear(2 * emb_size, 2)
        nn.init.xavier_normal_(self.output_fc.weight)
        nn.init.zeros_(self.output_fc.bias)

    def apply_graph_convolutions(self, prot1_x, adjacency1, prot2_x, adjacency2):
        for i, layer in enumerate(self.gnn_layers):
            # Apply linear transformation
            prot1_x = layer(prot1_x)
            prot2_x = layer(prot2_x)

            # Apply batch normalization and activation
            prot1_x = self.batch_norm_gnn[i](prot1_x.permute(0, 2, 1)).permute(0, 2, 1)
            prot2_x = self.batch_norm_gnn[i](prot2_x.permute(0, 2, 1)).permute(0, 2, 1)
            prot1_x = F.relu(prot1_x)
            prot2_x = F.relu(prot2_x)

            # Apply graph convolution (adjacency matrix multiplication)
            prot1_x = torch.bmm(adjacency1, prot1_x)
            prot2_x = torch.bmm(adjacency2, prot2_x)

            # Apply dropout
            prot1_x = self.dropout(prot1_x)
            prot2_x = self.dropout(prot2_x)

        return prot1_x, prot2_x

    def calculate_mutual_attention(self, hs1, hs2):
        batch_size = hs1.size(0)
        emb_size = hs1.size(2)

        # Transform embeddings
        prot1_attn_embed = self.attn_weights_1(hs1)
        prot2_attn_embed = self.attn_weights_2(hs2)

        # Compute pairwise attention
        d = torch.tanh(prot1_attn_embed.unsqueeze(2) + prot2_attn_embed.unsqueeze(1))
        pairwise_attn_scores = torch.matmul(d, self.attn_vector)

        # Compute attention weights
        prot1_mean_attn = pairwise_attn_scores.mean(dim=2)
        prot1_attn_weights = F.softmax(prot1_mean_attn, dim=1)
        prot1_weighted_vector = torch.bmm(
            prot1_attn_weights.unsqueeze(1), prot1_attn_embed
        ).squeeze(1)

        prot2_mean_attn = pairwise_attn_scores.mean(dim=1)
        prot2_attn_weights = F.softmax(prot2_mean_attn, dim=1)
        prot2_weighted_vector = torch.bmm(
            prot2_attn_weights.unsqueeze(1), prot2_attn_embed
        ).squeeze(1)

        # Concatenate
        combined = torch.cat((prot1_weighted_vector, prot2_weighted_vector), dim=1)

        return combined, prot1_attn_weights, prot2_attn_weights

    def forward(self, inputs):
        """Forward pass."""
        fp1, adjacency1, fp2, adjacency2 = inputs

        # Embedding lookups with batch norm and dropout
        prot1_emb = self.fingerprint_embeddings(fp1)
        prot2_emb = self.fingerprint_embeddings(fp2)
        prot1_emb = self.dropout(
            self.batch_norm_emb(prot1_emb.permute(0, 2, 1)).permute(0, 2, 1)
        )
        prot2_emb = self.dropout(
            self.batch_norm_emb(prot2_emb.permute(0, 2, 1)).permute(0, 2, 1)
        )

        # Graph convolutions
        prot1_conv_output, prot2_conv_output = self.apply_graph_convolutions(
            prot1_emb, adjacency1, prot2_emb, adjacency2
        )

        # Mutual attention
        y, attn_p1, attn_p2 = self.calculate_mutual_attention(
            prot1_conv_output, prot2_conv_output
        )

        # Final output
        logits = self.output_fc(y)

        return logits, attn_p1, attn_p2

    def __call__(self, data, train=True):
        inputs, target_label = data[:-1], data[-1]
        logits, attn_p1, attn_p2 = self.forward(inputs)
        target_label = target_label.to(logits.device).long()

        if train:
            loss = F.cross_entropy(logits, target_label)
            return loss
        else:
            return logits, target_label, attn_p1, attn_p2


# %%
# Training loop with train / val validation

# Model parameters
num_fingerprints = len(fingerprint_dict)
embedding_dim = 8  # wandb_config["embedding_dim"]
num_gnn_layers = 2  # wandb_config["num_gnn_layers"]
epochs = wandb_config["epochs"]
lr = wandb_config["lr"]
weight_decay = 1e-4  # L2 regularization

# Early stopping parameters
patience = 3
best_val_loss = float("inf")
patience_counter = 0

# Instantiate model and optimizer
model = PPIPredictor(num_fingerprints, embedding_dim, num_gnn_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Initialize wandb logging
run = wandb.init(
    project="ppi_pred",
    notes="Baseline with early stopping and weight decay",
    tags=["baseline", "early_stopping", "weight_decay"],
    config=wandb_config,
    mode="disabled",
)

wandb.watch(model, log="all", log_freq=2)

# Training loop
# Set gradient accumulation steps
accumulation_steps = 4  # Number of mini-batches to accumulate gradients

for epoch in range(epochs):
    # Training Phase
    model.train()
    total_loss = 0
    all_train_preds = []
    all_train_labels = []

    train_loader_tqdm = tqdm(
        train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False
    )

    optimizer.zero_grad()  # Initialize optimizer gradient

    for batch_idx, (inputs, target_labels, _) in enumerate(train_loader_tqdm):
        # Move data to device
        batch_fp1, batch_adj1, batch_fp2, batch_adj2 = inputs
        batch_fp1 = batch_fp1.to(device)
        batch_adj1 = batch_adj1.to(device)
        batch_fp2 = batch_fp2.to(device)
        batch_adj2 = batch_adj2.to(device)
        target_labels = target_labels.to(device)

        # Prepare data for model
        inputs = (batch_fp1, batch_adj1, batch_fp2, batch_adj2)
        data = inputs + (target_labels,)

        # Forward pass and loss computation
        loss = model(data, train=True)
        loss = loss / accumulation_steps  # Scale loss by accumulation steps

        # Backward pass
        loss.backward()

        # Perform optimizer step after accumulation steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(
            train_loader
        ):
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients
            torch.cuda.empty_cache()  # Release GPU memory

        total_loss += loss.item() * accumulation_steps  # Scale back for reporting

        # Collect predictions for accuracy
        with torch.no_grad():
            logits, _, _ = model.forward(inputs)
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(probabilities, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(target_labels.cpu().numpy())

        # Update progress bar
        train_loader_tqdm.set_postfix(loss=loss.item() * accumulation_steps)

    # Compute average training loss and accuracy
    avg_loss = total_loss / len(train_loader)
    train_accuracy = accuracy_score(all_train_labels, all_train_preds)

    # Logging to wandb
    wandb.log({"train_loss": avg_loss, "train_accuracy": train_accuracy})

    print(
        f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.4f}"
    )

    # Validation Phase
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        val_loader_tqdm = tqdm(
            val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation", leave=False
        )

        for batch_idx, (inputs, target_labels, _) in enumerate(val_loader_tqdm):
            # Move data to device
            batch_fp1, batch_adj1, batch_fp2, batch_adj2 = inputs
            batch_fp1 = batch_fp1.to(device)
            batch_adj1 = batch_adj1.to(device)
            batch_fp2 = batch_fp2.to(device)
            batch_adj2 = batch_adj2.to(device)
            target_labels = target_labels.to(device)

            # Prepare data for model
            inputs = (batch_fp1, batch_adj1, batch_fp2, batch_adj2)
            data = inputs + (target_labels,)

            # Evaluate
            logits, _, attn_p1, attn_p2 = model(data, train=False)

            # Compute validation loss
            val_loss += F.cross_entropy(logits, target_labels).item()

            # Compute probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Collect predictions and targets
            preds = torch.argmax(probabilities, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target_labels.cpu().numpy())

        # Compute average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)

        # Logging to wandb
        wandb.log({"val_loss": avg_val_loss, "val_accuracy": val_accuracy})

        print(
            f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0  # Reset counter
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

wandb.finish()

# %% [markdown]
# ### Without Normalization

# %%


class PPIPredictor(nn.Module):
    """Protein-Protein Interaction Predictor using GNN and Mutual Attention."""

    def __init__(self, num_fingerprints, emb_size, num_gnn_layers):
        """
        Initializes the PPIPredictor model.

        Args:
            num_fingerprints (int): The number of unique fingerprints (input size).
            emb_size (int): Dimension of the embedding vectors.
            n_layers (int): Number of GNN layers.
        """
        super(PPIPredictor, self).__init__()
        self.emb_size = emb_size
        self.num_gnn_layers = num_gnn_layers

        # Embedding layer for fingerprints
        self.fingerprint_embeddings = nn.Embedding(num_fingerprints, emb_size)

        # Initialize with uniform distribution
        nn.init.xavier_normal_(self.fingerprint_embeddings.weight)

        # Use ModuleList for the GNN layers and initialize them with Xavier initialization
        self.gnn_layers = nn.ModuleList(
            [nn.Linear(emb_size, emb_size) for _ in range(num_gnn_layers)]
        )
        for layer in self.gnn_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Initialize attention layers
        self.attn_weights_1 = nn.Linear(emb_size, emb_size)
        self.attn_weights_2 = nn.Linear(emb_size, emb_size)
        nn.init.xavier_normal_(self.attn_weights_1.weight)
        nn.init.xavier_normal_(self.attn_weights_2.weight)

        # Initialize parameter `w` for attention scores with Xavier initialization
        self.attn_vector = nn.Parameter(torch.randn(emb_size))

        # Output layer initialization
        self.output_fc = nn.Linear(2 * emb_size, 2)
        nn.init.xavier_normal_(self.output_fc.weight)
        nn.init.zeros_(self.output_fc.bias)

    def apply_graph_convolutions(self, prot1_x, adjacency1, prot2_x, adjacency2):
        for layer in self.gnn_layers:
            # Apply linear transformation and non-linearity

            prot1_x = torch.bmm(adjacency1, F.relu(layer(prot1_x)))
            prot2_x = torch.bmm(adjacency2, F.relu(layer(prot2_x)))

        return prot1_x, prot2_x

    def calculate_mutual_attention(self, hs1, hs2):
        batch_size = hs1.size(0)
        emb_size = hs1.size(2)

        # Transform embeddings
        # [batch_size, num_nodes, emb_size]
        prot1_attn_embed = self.attn_weights_1(hs1)
        prot2_attn_embed = self.attn_weights_2(hs2)

        # Compute pairwise attention

        # [batch_size, num_nodes1, num_nodes2, emb_size]
        d = torch.tanh(prot1_attn_embed.unsqueeze(2) + prot2_attn_embed.unsqueeze(1))

        # [batch_size, num_nodes1, num_nodes2]
        pairwise_attn_scores = torch.matmul(d, self.attn_vector)

        # Compute attention weights
        prot1_mean_attn = pairwise_attn_scores.mean(dim=2)  # [batch_size, num_nodes1]
        prot1_attn_weights = F.softmax(prot1_mean_attn, dim=1)
        prot1_weighted_vector = torch.bmm(
            prot1_attn_weights.unsqueeze(1), prot1_attn_embed
        ).squeeze(1)

        prot2_mean_attn = pairwise_attn_scores.mean(dim=1)  # [batch_size, num_nodes2]
        prot2_attn_weights = F.softmax(prot2_mean_attn, dim=1)
        prot2_weighted_vector = torch.bmm(
            prot2_attn_weights.unsqueeze(1), prot2_attn_embed
        ).squeeze(1)

        # Concatenate
        combined = torch.cat(
            (prot1_weighted_vector, prot2_weighted_vector), dim=1
        )  # [batch_size, 2 * emb_size]

        return combined, prot1_attn_weights, prot2_attn_weights

    def forward(self, inputs):
        """The forward pass takes the input data as arguments and returns the logits and protein attention weights."""
        # Ensure inputs is unpacked correctly
        fp1, adjacency1, fp2, adjacency2 = inputs

        # Get embedding lookups for fingerprints
        prot1_emb = self.fingerprint_embeddings(
            fp1
        )  # [batch_size, num_nodes1, emb_size]
        prot2_emb = self.fingerprint_embeddings(
            fp2
        )  # [batch_size, num_nodes2, emb_size]

        # Apply graph convolutions
        prot1_conv_output, prot2_conv_output = self.apply_graph_convolutions(
            prot1_emb, adjacency1, prot2_emb, adjacency2
        )

        # Compute mutual attention between protein graphs
        y, attn_p1, attn_p2 = self.calculate_mutual_attention(
            prot1_conv_output, prot2_conv_output
        )

        # Final prediction with the output layer
        logits = self.output_fc(y)  # [batch_size, 2]

        return logits, attn_p1, attn_p2

    def __call__(self, data, train=True):
        # Unpack the data
        inputs, target_label = data[:-1], data[-1]

        # Forward pass
        logits, attn_p1, attn_p2 = self.forward(inputs)

        # Ensure target_label is on the correct device and type
        target_label = target_label.to(logits.device).long()

        if train:
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, target_label)
            return loss
        else:
            # Return logits and other necessary outputs
            return logits, target_label, attn_p1, attn_p2


# %%
# Training loop with train / val validation

# Model parameters
num_fingerprints = len(fingerprint_dict)
embedding_dim = 32  # wandb_config["embedding_dim"]
num_gnn_layers = 2  # wandb_config["num_gnn_layers"]
epochs = wandb_config["epochs"]
lr = wandb_config["lr"]

# Instantiate model and optimizer
model = PPIPredictor(num_fingerprints, embedding_dim, num_gnn_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Initialize wandb logging
run = wandb.init(
    project="ppi_pred",
    notes="First test",
    tags=["baseline"],
    config=wandb_config,
    mode="disabled",
)

wandb.watch(model, log="all", log_freq=2)
# Set gradient accumulation steps
accumulation_steps = 4  # Number of mini-batches to accumulate gradients

for epoch in range(epochs):
    # Training Phase
    model.train()
    total_loss = 0
    all_train_preds = []
    all_train_labels = []

    train_loader_tqdm = tqdm(
        train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False
    )

    optimizer.zero_grad()  # Initialize optimizer gradient

    for batch_idx, (inputs, target_labels, _) in enumerate(train_loader_tqdm):
        # Move data to device
        batch_fp1, batch_adj1, batch_fp2, batch_adj2 = inputs
        batch_fp1 = batch_fp1.to(device)
        batch_adj1 = batch_adj1.to(device)
        batch_fp2 = batch_fp2.to(device)
        batch_adj2 = batch_adj2.to(device)
        target_labels = target_labels.to(device)

        # Prepare data for model
        inputs = (batch_fp1, batch_adj1, batch_fp2, batch_adj2)
        data = inputs + (target_labels,)

        # Forward pass and loss computation
        loss = model(data, train=True)
        loss = loss / accumulation_steps  # Scale loss by accumulation steps

        # Backward pass
        loss.backward()

        # Perform optimizer step after accumulation steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(
            train_loader
        ):
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients
            torch.cuda.empty_cache()  # Release GPU memory

        total_loss += loss.item() * accumulation_steps  # Scale back for reporting

        # Collect predictions for accuracy
        with torch.no_grad():
            logits, _, _ = model.forward(inputs)
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(probabilities, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(target_labels.cpu().numpy())

        # Update progress bar
        train_loader_tqdm.set_postfix(loss=loss.item() * accumulation_steps)

    # Compute average training loss and accuracy
    avg_loss = total_loss / len(train_loader)
    train_accuracy = accuracy_score(all_train_labels, all_train_preds)

    # Logging to wandb
    wandb.log({"train_loss": avg_loss, "train_accuracy": train_accuracy})

    print(
        f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.4f}"
    )

    # Validation Phase
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        val_loader_tqdm = tqdm(
            val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation", leave=False
        )

        for batch_idx, (inputs, target_labels, _) in enumerate(val_loader_tqdm):
            # Move data to device
            batch_fp1, batch_adj1, batch_fp2, batch_adj2 = inputs
            batch_fp1 = batch_fp1.to(device)
            batch_adj1 = batch_adj1.to(device)
            batch_fp2 = batch_fp2.to(device)
            batch_adj2 = batch_adj2.to(device)
            target_labels = target_labels.to(device)

            # Prepare data for model
            inputs = (batch_fp1, batch_adj1, batch_fp2, batch_adj2)
            data = inputs + (target_labels,)

            # Evaluate
            logits, _, attn_p1, attn_p2 = model(data, train=False)

            # Compute validation loss
            val_loss += F.cross_entropy(logits, target_labels).item()

            # Compute probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Collect predictions and targets
            preds = torch.argmax(probabilities, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target_labels.cpu().numpy())

        # Compute average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)

        # Logging to wandb
        wandb.log({"val_loss": avg_val_loss, "val_accuracy": val_accuracy})

        print(
            f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

wandb.finish()


# %% [markdown]
# ## Evaluation


# %%
def test_model(model, test_loader, device):
    """
    Evaluate the model on the test set.

    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for the test set.
        device: Torch device ('cuda' or 'cpu').

    Returns:
        probabilities (np.ndarray): Predicted probabilities for the test set.
        roc_curve_plot: ROC curve plot.
        confusion_matrix_plot: Confusion matrix plot.
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, target_labels in tqdm(test_loader, desc="Testing"):
            # Move data to device
            inputs = [x.to(device) for x in inputs]
            target_labels = target_labels.to(device)

            # Ensure inputs is a tuple of tensors
            inputs = tuple(inputs)

            # Prepare data for model
            data = inputs + (target_labels,)

            # Evaluate
            logits, _, _, _ = model(data, train=False)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            # Collect predictions and targets
            preds = np.argmax(probabilities, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probabilities[:, 1])  # Assuming binary classification
            all_labels.extend(target_labels.cpu().numpy())

    # Convert lists to arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.show()

    return all_probs, fpr, tpr, cm


# %%
probs, fpr, tpr, cm = test_model(model, test_loader, device)

# %%
probs
