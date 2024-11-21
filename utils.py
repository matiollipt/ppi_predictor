# Visualization
import plotly.express as px
import plotly.graph_objects as go


def plot_protein_3D(pdb_id, chain_id, residues, coords, acid_dict):
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
