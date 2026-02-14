import numpy as np
import matplotlib.pyplot as plt

def transform_label(L:np.ndarray)->list: 
    """Transfom binary mask to a list of start and ends

    Args:
        L (np.ndarray): binary mask, shape (n_label,length_time_series)

    Returns:
        list: start and end list. 
    """
    lst = []
    for line in L: 
        if np.count_nonzero(line)!=0:
            line = np.hstack(([0],line,[0]))
            diff = np.diff(line)
            start = np.where(diff==1)[0]+1
            end = np.where(diff==-1)[0]
            lst.append(np.array(list(zip(start,end))))
    return np.array(lst,dtype=object)

import matplotlib.pyplot as plt
import numpy as np

def plot_signal_and_motifs(
    signal,
    motifs,
    dimension_names=None,
    motif_names=None,
    row_height=2,
    col_width=8,
    show_axes=True,
    save_svg=False,
    svg_name="signal_motifs.svg"
):
    """
    Plot un signal multivarié et colore les occurrences de motifs.
    """

    n_samples, n_dimensions = signal.shape
    motifs = motifs.T
    n_motifs = motifs.shape[1]

    # Couleurs par motif
    colors = {m: plt.cm.tab10(m % 10) for m in range(n_motifs)}

    fig, axes = plt.subplots(
        n_dimensions,
        1,
        figsize=(col_width, row_height * n_dimensions),
        sharex=True
    )

    if n_dimensions == 1:
        axes = [axes]

    for dim in range(n_dimensions):
        ax = axes[dim]
        if not show_axes:
            ax.set_xticks([])
            ax.set_yticks([])
        if dimension_names is not None:
            ax.set_ylabel(dimension_names[dim])

        ax.plot(signal[:, dim], color="black", linewidth=1)

        if dimension_names is not None:
            ax.set_ylabel(dimension_names[dim])

        # Coloration des motifs
        for m in range(n_motifs):
            mask = motifs[:, m] == 1
            ax.fill_between(
                np.arange(n_samples),
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                where=mask,
                color=colors[m],
                alpha=0.3
            )

        ax.grid(False)

    axes[-1].set_xlabel("Time")

    # Légende globale
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[m], alpha=0.3)
        for m in range(n_motifs)
    ]
    labels = (
        motif_names
        if motif_names is not None
        else [f"Motif {m+1}" for m in range(n_motifs)]
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(n_motifs, 4),
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_svg:
        plt.savefig(svg_name, format="svg", bbox_inches="tight")
        print(f"✅ Figure SVG sauvegardée : {svg_name}")

    plt.show()



import math
import matplotlib.pyplot as plt

import math
import matplotlib.pyplot as plt

def plot_signal_and_submotifs(
    signal,
    motifs,
    motif_dims,
    dimension_names=None,
    motifs_names=None,
    show_axes=True,
    n_cols=1,
    row_height=2.0,
    col_width=4.0,
    save_svg=False,
    svg_name="signal_motifs.svg"
):
    n_samples, n_dimensions = signal.shape
    motifs = motifs.T
    n_motifs = motifs.shape[1]

    # Couleurs pour chaque motif
    colors = {motif: plt.cm.tab10(motif % 10) for motif in range(n_motifs)}

    # Grille
    n_rows = math.ceil(n_dimensions / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(col_width * n_cols, row_height * n_rows),
        sharex=True
    )

    # Forcer axes en tableau 2D
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Placement colonne-par-colonne
    for dim in range(n_dimensions):
        col = dim // n_rows
        row = dim % n_rows

        ax = axes[row][col]
        if not show_axes:
            ax.set_xticks([])
            ax.set_yticks([])
        if dimension_names is not None:
            ax.set_ylabel(dimension_names[dim])

        ax.plot(signal[:, dim], color="black", linewidth=1)

        for motif in range(n_motifs):
            if dim in motif_dims[motif]:
                mask = motifs[:, motif] == 1
                ax.fill_between(
                    range(n_samples),
                    ax.get_ylim()[0],
                    ax.get_ylim()[1],
                    where=mask,
                    color=colors[motif],
                    alpha=0.3
                )

    # Désactiver les axes inutilisés
    for dim in range(n_dimensions, n_rows * n_cols):
        col = dim // n_rows
        row = dim % n_rows
        axes[row][col].axis("off")

    #plt.xlabel("Time")
    for col in range(n_cols):
        axes[-1][col].set_xlabel("Time")
    plt.tight_layout()
    if save_svg:
        plt.savefig(svg_name, format="svg", bbox_inches="tight")
        print(f"✅ Figure SVG sauvegardée : {svg_name}")
    plt.show()



import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_signal_and_motifs_by_dimension(signal, estimated_masks_per_dim, true_masks=None):
    """
    Affiche un signal multivarié avec coloration des motifs estimés par dimension.
    Si des motifs réels sont fournis, trace des lignes verticales colorées pour indiquer leur début et leur fin.

    Parameters:
    - signal: np.ndarray de forme (n_samples, n_dimensions)
    - estimated_masks_per_dim: liste de np.ndarrays (n_motifs_est, n_samples) pour chaque dimension
    - true_masks: (optionnel) np.ndarray de forme (n_true_motifs, n_samples), global à toutes les dimensions
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_samples, n_dimensions = signal.shape
    assert len(estimated_masks_per_dim) == n_dimensions, "Un masque estimé par dimension est requis."
    if true_masks is not None:
        assert true_masks.ndim == 2 and true_masks.shape[1] == n_samples, "true_masks doit être (n_true_motifs, n_samples)"

    fig, axes = plt.subplots(n_dimensions, 1, figsize=(12, 2.5 * n_dimensions), sharex=True)
    if n_dimensions == 1:
        axes = [axes]

    # Couleurs pour les motifs réels
    if true_masks is not None:
        true_colors = [plt.cm.Set1(i % 9) for i in range(true_masks.shape[0])]  # Set1 a 9 couleurs distinctes

    for dim in range(n_dimensions):
        ax = axes[dim]
        ax.plot(signal[:, dim], label=f"Dimension {dim+1}", color='black', linewidth=1)

        # Estimés
        est_masks = estimated_masks_per_dim[dim]
        est_colors = [plt.cm.tab10(i % 10) for i in range(est_masks.shape[0])]
        for j, mask in enumerate(est_masks):
            ax.fill_between(
                np.arange(n_samples),
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                where=mask.astype(bool),
                color=est_colors[j],
                alpha=0.3,
                label=f"Motif estimé {j+1}"
            )

        # Vrais motifs : lignes pointillées colorées
        if true_masks is not None:
            for j, mask in enumerate(true_masks):
                start_idxs = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
                end_idxs = np.where(np.diff(mask.astype(int)) == -1)[0] + 1

                if mask[0]:
                    start_idxs = np.insert(start_idxs, 0, 0)
                if mask[-1]:
                    end_idxs = np.append(end_idxs, n_samples - 1)

                for start, end in zip(start_idxs, end_idxs):
                    ax.axvline(start, color=true_colors[j], linestyle='--', linewidth=2.5)
                    ax.axvline(end, color=true_colors[j], linestyle='--', linewidth=2.5)

        ax.set_ylabel(f"Dim {dim+1}")
        ax.legend(loc="upper right")

    plt.xlabel("Temps")
    plt.tight_layout()
    plt.show()


def plot_signal_and_motifs_by_group(signal, estimated_masks_per_group, dimension_groups, true_masks=None):
    """
    Affiche un signal multivarié avec coloration des motifs estimés par groupe de dimensions.
    Si des motifs réels sont fournis, trace des lignes verticales colorées pour indiquer leur début et leur fin.

    Parameters:
    - signal: np.ndarray de forme (n_samples, n_dimensions)
    - estimated_masks_per_group: liste de np.ndarrays (n_motifs_est, n_samples) pour chaque groupe
    - dimension_groups: liste de listes, chaque sous-liste contient les indices des dimensions du groupe
    - true_masks: (optionnel) np.ndarray de forme (n_true_motifs, n_samples), global à toutes les dimensions
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_samples, n_dimensions = signal.shape
    assert len(estimated_masks_per_group) == len(dimension_groups), "Un masque estimé par groupe est requis."

    total_plots = sum(len(group) for group in dimension_groups)
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 2.5 * total_plots), sharex=True)
    if total_plots == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    if true_masks is not None:
        assert true_masks.ndim == 2 and true_masks.shape[1] == n_samples, "true_masks doit être (n_true_motifs, n_samples)"
        true_colors = [plt.cm.Set1(i % 9) for i in range(true_masks.shape[0])]

    ax_idx = 0
    for group_idx, group in enumerate(dimension_groups):
        est_masks = estimated_masks_per_group[group_idx]
        est_colors = [plt.cm.tab10(i % 10) for i in range(est_masks.shape[0])]

        for dim in group:
            ax = axes[ax_idx]
            ax.plot(signal[:, dim], label=f"Dimension {dim+1}", color='black', linewidth=1)

            for j, mask in enumerate(est_masks):
                ax.fill_between(
                    np.arange(n_samples),
                    ax.get_ylim()[0],
                    ax.get_ylim()[1],
                    where=mask.astype(bool),
                    color=est_colors[j],
                    alpha=0.3,
                    label=f"Motif estimé {j+1}"
                )

            if true_masks is not None:
                for j, mask in enumerate(true_masks):
                    start_idxs = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
                    end_idxs = np.where(np.diff(mask.astype(int)) == -1)[0] + 1
                    if mask[0]:
                        start_idxs = np.insert(start_idxs, 0, 0)
                    if mask[-1]:
                        end_idxs = np.append(end_idxs, n_samples - 1)

                    for start, end in zip(start_idxs, end_idxs):
                        ax.axvline(start, color=true_colors[j], linestyle='--', linewidth=2.5)
                        ax.axvline(end, color=true_colors[j], linestyle='--', linewidth=2.5)

            ax.set_ylabel(f"Dim {dim+1}")
            ax.legend(loc="upper right")
            ax_idx += 1

    plt.xlabel("Temps")
    plt.tight_layout()
    plt.show()

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def plot_signal_and_submotifs_by_dim_apparition(signal, motifs_list):
    """
    Affiche un signal multivarié avec un subplot par dimension,
    et des boutons pour filtrer les motifs selon leur nombre de dimensions.
    
    motifs_list : liste de dicts du type
        {'motif_counter': int, 'dims': [int], 'segs': [[start, end], ...]}
    """
    n_samples, n_dimensions = signal.shape

    # --- Préparation de la figure ---
    fig = make_subplots(rows=n_dimensions, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        subplot_titles=[f"Dimension {i}" for i in range(n_dimensions)])

    # --- Tracer le signal (toujours visible) ---
    for dim in range(n_dimensions):
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_samples),
                y=signal[:, dim],
                mode="lines",
                line=dict(color="black", width=1),
                name=f"Signal dim {dim}",
                legendgroup="signal",
                showlegend=(dim == 0),
                visible=True
            ),
            row=dim + 1, col=1
        )

    # --- Palette de couleurs ---
    cmap = plt.cm.get_cmap("tab10", 20)
    motif_colors = [f"rgba{tuple((np.array(cmap(i)[:3]) * 255).astype(int)) + (0.3,)}" for i in range(20)]

    # --- Ajouter les zones de motifs ---
    trace_info = []  # (trace_idx, nb_dims)
    for motif_idx, motif in enumerate(motifs_list):
        dims = motif["dims"]
        segs = motif["segs"]
        dim_count = len(dims)
        color = motif_colors[motif_idx % 20]

        for dim in dims:
            for start, end in segs:
                # Tracer une zone colorée sur le segment correspondant
                trace = go.Scatter(
                    x=[start, end, end, start],
                    y=[signal[:, dim].min(), signal[:, dim].min(),
                       signal[:, dim].max(), signal[:, dim].max()],
                    fill="toself",
                    fillcolor=color,
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    visible=True
                )
                fig.add_trace(trace, row=dim + 1, col=1)
                trace_info.append((len(fig.data) - 1, dim_count))

    # --- Boutons interactifs pour filtrer par nombre de dimensions ---
    unique_dim_counts = sorted(set(len(m["dims"]) for m in motifs_list))
    buttons = []

    for k in unique_dim_counts:
        visible = []
        for i, trace in enumerate(fig.data):
            if trace.legendgroup == "signal":  # Le signal reste toujours visible
                visible.append(True)
            else:
                visible.append(any(k == dim_count for tidx, dim_count in trace_info if tidx == i))
        buttons.append(dict(
            label=f"{k}D",
            method="update",
            args=[{"visible": visible},
                  {"title": f"Motifs sur {k} dimensions"}]
        ))

    # Bouton "Tous"
    visible_all = [True] * len(fig.data)
    buttons.insert(0, dict(
        label="Tous",
        method="update",
        args=[{"visible": visible_all},
              {"title": "Tous les motifs"}]
    ))

    # --- Layout ---
    fig.update_layout(
        height=250 * n_dimensions,
        template="plotly_white",
        title="Visualisation des motifs par nombre de dimensions",
        xaxis_title="Temps",
        updatemenus=[{
            "type": "buttons",
            "x": 1.05,
            "y": 1,
            "buttons": buttons,
            "showactive": True
        }]
    )

    for i in range(n_dimensions):
        fig.update_yaxes(title_text=f"Dimension {i}", row=i + 1, col=1)

    fig.show()

